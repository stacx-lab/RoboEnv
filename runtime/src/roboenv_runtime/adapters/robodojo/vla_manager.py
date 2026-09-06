"""Bounded, source-locked VLA-as-a-tool integration owned by the worker.

This deliberately does not call ``RoboDojoEnv.vla_run``: that legacy helper
re-registers the reward program through ``compute_reward`` and writes policy
logs outside this campaign.  The manager below uses RoboDojo's native
``EvalEnv.take_action`` path, which already advances the official reward clock
once per action frame.
"""

# Moved from codex_dojo_alltasks/harness/vla_manager.py on 2026-08-26 (refactor P3).

from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import subprocess
import time
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

# The largest canonical official episode is 1,900 action frames. Keep a small
# hard ceiling above that while the live remaining-budget check provides the
# tighter per-task bound.
MAX_VLA_ACTIONS = 2_000
MAX_VLA_CHUNK = 110
SUPPORTED_POLICIES = frozenset({"G05", "Pi_05"})


def _validated_policy(policy: Any) -> str:
    """Return an exact allowlisted policy identifier or fail before spawning."""

    if not isinstance(policy, str) or policy not in SUPPORTED_POLICIES:
        allowed = ", ".join(sorted(SUPPORTED_POLICIES))
        raise ValueError(f"unsupported policy {policy!r}; locked policies: {allowed}")
    return policy


JOINT_ACTION_FIELDS = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)


def _validated_joint_action(action: Any) -> dict[str, np.ndarray]:
    """Normalize finite flat or structured dual-X5 joint actions.

    XPolicyLab adapters return the native EvalEnv mapping, while some policy
    clients expose the same values as one packed 14-vector.  Accept exactly
    those two representations and always hand Isaac the canonical mapping.
    """

    if isinstance(action, Mapping):
        expected = {name for name, _ in JOINT_ACTION_FIELDS}
        actual = set(action)
        if actual != expected:
            raise RuntimeError(
                "structured policy action keys must be exactly "
                f"{sorted(expected)}, received {sorted(map(str, actual))}"
            )
        normalized: dict[str, np.ndarray] = {}
        for name, dimension in JOINT_ACTION_FIELDS:
            try:
                value = np.asarray(action[name], dtype=float)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"policy action field {name!r} must be numeric") from exc
            if value.shape != (dimension,):
                raise RuntimeError(
                    f"policy action field {name!r} must have shape "
                    f"({dimension},), received {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"policy action field {name!r} contains NaN or infinity")
            normalized[name] = value
        return normalized

    try:
        vector = np.asarray(action, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "policy action must be a packed 14-vector or native joint mapping"
        ) from exc
    if vector.shape != (14,):
        raise RuntimeError(f"packed policy action must have shape (14,), received {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("packed policy action contains NaN or infinity")
    normalized = {}
    offset = 0
    for name, dimension in JOINT_ACTION_FIELDS:
        normalized[name] = vector[offset : offset + dimension]
        offset += dimension
    return normalized


def _inside(path: Path, root: Path, *, allow_root: bool = False) -> Path:
    resolved = path.resolve(strict=False)
    root = root.resolve(strict=True)
    if resolved == root:
        if allow_root:
            return resolved
        raise ValueError(f"path must be below campaign root, not the root itself: {resolved}")
    if root not in resolved.parents:
        raise ValueError(f"path escapes campaign root: {resolved}")
    return resolved


def _remaining_sim_steps(budget: Any) -> int:
    """Normalize both adapter budget shapes; unknown is a hard failure."""

    if not isinstance(budget, dict):
        raise RuntimeError("adapter budget must be a dictionary")
    for key in ("remaining_steps", "remaining"):
        value = budget.get(key)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and np.isfinite(value)
            and value >= 0
        ):
            return int(value)
    used, maximum = budget.get("used"), budget.get("max")
    numeric = (
        not isinstance(used, bool)
        and not isinstance(maximum, bool)
        and isinstance(used, (int, float))
        and isinstance(maximum, (int, float))
        and np.isfinite(used)
        and np.isfinite(maximum)
        and used >= 0
        and maximum >= 0
    )
    if numeric:
        return int(max(0.0, float(maximum) - float(used)))
    raise RuntimeError("adapter budget lacks remaining_steps or numeric used/max")


def _action_batch(proposed: Any) -> list[Any]:
    """Normalize a single 14-vector or a policy chunk without flattening it."""

    if isinstance(proposed, np.ndarray):
        if proposed.ndim <= 1:
            return [proposed]
        return [proposed[index] for index in range(proposed.shape[0])]
    if isinstance(proposed, (list, tuple)):
        if not proposed:
            return []
        if np.isscalar(proposed[0]):
            return [proposed]
        return list(proposed)
    return [proposed]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_actions_sha256(actions: list[Any]) -> str:
    """Hash canonical finite action JSON for durable inline provenance."""

    payload = json.dumps(
        _jsonable(actions),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quaternion_angle(first: Any, second: Any) -> float:
    """Shortest angular distance between two wxyz quaternions."""

    lhs = np.asarray(first, dtype=float).reshape(-1)
    rhs = np.asarray(second, dtype=float).reshape(-1)
    if lhs.size != 4 or rhs.size != 4:
        return 0.0
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm < 1e-9 or rhs_norm < 1e-9:
        return 0.0
    dot = float(np.dot(lhs / lhs_norm, rhs / rhs_norm))
    return float(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0)))


def _gripper_motion(
    before: dict[str, dict[str, list[float]]],
    after: dict[str, dict[str, list[float]]],
    *,
    orientation_radius_m: float = 0.05,
) -> dict[str, float]:
    """Task-agnostic end-effector motion in equivalent metres.

    Pure wrist rotation is meaningful during approach, insertion, turning and
    pouring. Map its angular distance through a small virtual tool radius, then
    take the larger of translational and rotational motion for each arm.
    """

    result: dict[str, float] = {}
    for arm in sorted(before.keys() & after.keys()):
        first = before[arm]
        second = after[arm]
        try:
            translation = float(
                np.linalg.norm(
                    np.asarray(second["position"], dtype=float)
                    - np.asarray(first["position"], dtype=float)
                )
            )
            rotation = orientation_radius_m * _quaternion_angle(
                first["quaternion"], second["quaternion"]
            )
            jaw_motion = abs(float(second.get("width", 0.0)) - float(first.get("width", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue
        equivalent = max(translation, rotation, jaw_motion)
        if equivalent > 0.001:
            result[arm] = round(equivalent, 6)
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(_jsonable(value), sort_keys=True, indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _policy_runtime_env(vla_dir: Path) -> dict[str, str]:
    """Return writable, per-run cache locations for a policy subprocess.

    Policy environments inherit the launching shell.  On the shared host that included a
    Deployment locations are configured explicitly by the runtime.
    failed before opening its port.  Keep every mutable cache below this run just like the
    policy logs and action evidence.
    """

    cache = vla_dir / "cache"
    paths = {
        "HOME": vla_dir / "home",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache / "xdg",
        "TRITON_CACHE_DIR": cache / "triton",
        "TMPDIR": vla_dir / "tmp",
        "WANDB_DIR": vla_dir / "wandb",
        "JAX_COMPILATION_CACHE_DIR": cache / "jax",
        "UV_CACHE_DIR": cache / "uv",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return {
        **{key: str(value) for key, value in paths.items()},
        "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore::UserWarning",
    }


class VlaManager:
    """Own one lazily-started policy server and execute supervised chunks."""

    def __init__(
        self, adapter: Any, score_clock: Any, run_dir: str | Path, physical_gpu: int | None = None
    ) -> None:
        self.adapter = adapter
        self.clock = score_clock
        # every VLA artefact stays below the run directory (roboenv P3: the campaign
        # root of the codex harness is simply the runs root now)
        raw_root = os.environ.get("ROBOENV_RUNS_ROOT") or os.environ.get("CODEX_CAMPAIGN_ROOT")
        self.campaign_root = (
            Path(raw_root).resolve(strict=True) if raw_root else Path(run_dir).resolve().parent
        )
        self.run_dir = _inside(Path(run_dir), self.campaign_root)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {self.run_dir}")
        self.vla_dir = self.run_dir / "vla"
        self.vla_dir.mkdir(parents=True, exist_ok=True)
        self.vla_dir = _inside(self.vla_dir, self.run_dir)
        raw_gpu = os.environ.get("CODEX_PHYSICAL_GPU")
        if physical_gpu is None:
            physical_gpu = int(
                raw_gpu
                if raw_gpu is not None
                else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            )
        self.physical_gpu = int(physical_gpu)
        self._client: Any = None
        self._server: subprocess.Popen | None = None
        self._log_handle: Any = None
        self._needs_reset = True
        self._sequence = 0
        self._active_policy: str | None = None
        self._server_instance = uuid.uuid4().hex

    def inventory(self) -> list[dict[str, Any]]:
        if self._server is None:
            return []
        return [{
            "policy": self._active_policy,
            "instance": self._server_instance,
            "owned": True,
            "pid": self._server.pid,
            "alive": self._server.poll() is None,
        }]

    @property
    def env(self) -> Any:
        return self.adapter._env._env

    def _policy_command(self, policy: str, port: int) -> tuple[list[str], dict[str, str]]:
        policy = _validated_policy(policy)
        from roboenv_runtime.paths import robodojo_root

        rd_root = robodojo_root().resolve()
        xpl = rd_root / "XPolicyLab"
        entrypoint = xpl / "setup_policy_server.py"
        task = str(self.adapter._env.task_name)
        seed = int(self.adapter._env.seed)

        env = dict(os.environ)
        old_pythonpath = env.get("PYTHONPATH", "")
        env.update(_policy_runtime_env(self.vla_dir))
        env["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu)

        policy_dir = xpl / "policy" / policy
        config = policy_dir / "deploy.yml"
        if policy == "G05":
            g05_root = policy_dir / policy
            python = g05_root / ".venv" / "bin" / "python"
            checkpoint = (
                xpl
                / "policy"
                / "GalaxeaVLA"
                / "checkpoints"
                / "RoboDojo-sim-arx_x5-joint-0"
                / "checkpoints"
                / "checkpoint"
            )
            tokenizer = (
                xpl
                / "policy"
                / "GalaxeaVLA"
                / "checkpoints"
                / "RoboDojo-sim-arx_x5-joint-0"
                / "action_tokenizer.pt"
            )
            for required in (python, entrypoint, config, checkpoint, tokenizer):
                if not required.is_file():
                    raise FileNotFoundError(f"locked VLA dependency missing: {required}")
            cmd = [
                str(python),
                "-u",
                str(entrypoint),
                "--config_path",
                str(config),
                "--overrides",
                "host=localhost",
                f"port={port}",
                "bench_name=RoboDojo",
                f"task_name={task}",
                "ckpt_name=RoboDojo-sim-arx_x5-joint-0",
                "env_cfg_type=arx_x5",
                f"seed={seed}",
                "policy_name=G05",
                "action_type=joint",
                "action_dim=14",
                f"ckpt_path={checkpoint}",
                f"g05_root={g05_root}",
            ]
            env.update(
                {
                    "ROBODOJO_G05_ACTION_SOURCE": "fm",
                    "G05_ACTION_TOKENIZER_PATH": str(tokenizer),
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            [
                                str(g05_root / "src"),
                                str(g05_root),
                                str(rd_root),
                                str(xpl),
                                old_pythonpath,
                            ],
                        )
                    ),
                }
            )
            return cmd, env

        openpi_root = policy_dir / "openpi"
        openpi_src = openpi_root / "src"
        python = openpi_root / ".venv" / "bin" / "python"
        model = policy_dir / "model.py"
        checkpoint = policy_dir / "checkpoints" / "RoboDojo-sim-arx_x5-joint-0" / "59999"
        checkpoint_marker = checkpoint / "_CHECKPOINT_METADATA"
        norm_stats = checkpoint / "assets" / "arx_x5_sim" / "norm_stats.json"
        bootstrap = Path(__file__).with_name("pi_policy_bootstrap.py").resolve(strict=True)
        if bootstrap.parent != Path(__file__).resolve().parent:
            raise RuntimeError("Pi policy bootstrap escaped harness directory")
        for required in (
            python,
            entrypoint,
            config,
            model,
            checkpoint_marker,
            norm_stats,
            bootstrap,
        ):
            if not required.is_file():
                raise FileNotFoundError(f"locked VLA dependency missing: {required}")
        if not openpi_src.is_dir() or not checkpoint.is_dir():
            missing = openpi_src if not openpi_src.is_dir() else checkpoint
            raise FileNotFoundError(f"locked VLA dependency missing: {missing}")

        for stale_key in (
            "ROBODOJO_G05_ACTION_SOURCE",
            "G05_ACTION_TOKENIZER_PATH",
        ):
            env.pop(stale_key, None)
        cmd = [
            str(python),
            "-u",
            str(bootstrap),
            "--config_path",
            str(config),
            "--overrides",
            "host=localhost",
            f"port={port}",
            "bench_name=RoboDojo",
            f"task_name={task}",
            "ckpt_name=RoboDojo-sim-arx_x5-joint-0",
            "env_cfg_type=arx_x5",
            f"seed={seed}",
            "policy_name=Pi_05",
            "action_type=joint",
            "action_dim=14",
            "checkpoint_num=59999",
        ]
        env.update(
            {
                "BENCH_ROOT": str(rd_root),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        [
                            str(rd_root),
                            str(openpi_src),
                            str(xpl),
                            old_pythonpath,
                        ],
                    )
                ),
            }
        )
        return cmd, env

    def _ensure_client(self, policy: str = "G05", timeout_s: int = 900) -> Any:
        policy = _validated_policy(policy)
        if self._client is not None:
            if self._active_policy != policy:
                raise RuntimeError(
                    f"VLA worker is locked to {self._active_policy!r}; "
                    f"refusing policy switch to {policy!r}"
                )
            return self._client
        if self._active_policy is not None and self._active_policy != policy:
            raise RuntimeError(
                f"VLA worker was locked to {self._active_policy!r}; "
                f"refusing policy switch to {policy!r}"
            )
        if self._server is not None:
            state = (
                "running"
                if self._server.poll() is None
                else (f"exited rc={self._server.returncode}")
            )
            raise RuntimeError(
                f"VLA server exists without a client ({state}); refusing an ambiguous second launch"
            )

        port = int(os.environ.get("CODEX_VLA_PORT_BASE", "18790")) + self.physical_gpu
        cmd, env = self._policy_command(policy, port)
        self._log_handle = (self.vla_dir / "server.log").open("ab", buffering=0)
        try:
            # Inherit the worker process group.  The supervisor can therefore
            # kill the simulator+VLA tree without discovering unrelated PIDs.
            self._server = subprocess.Popen(
                cmd,
                cwd=self.vla_dir,
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self._server.poll() is not None:
                    raise RuntimeError(
                        f"VLA server exited rc={self._server.returncode}; "
                        f"see {self.vla_dir / 'server.log'}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        break
                except OSError:
                    time.sleep(5)
            else:
                raise TimeoutError(f"VLA server did not open port {port} in {timeout_s}s")

            import websockets

            original_connect = websockets.connect
            try:
                if "proxy" not in inspect.signature(original_connect).parameters:

                    def connect_without_proxy(*args: Any, proxy: Any = None, **kwargs: Any) -> Any:
                        return original_connect(*args, **kwargs)

                    websockets.connect = connect_without_proxy
            except (TypeError, ValueError):
                pass
            from client_server.ws.model_client import WsModelClient

            task = str(self.adapter._env.task_name)
            self._client = WsModelClient(
                url=f"ws://127.0.0.1:{port}",
                evaluation_id=f"roboenv-{task}",
                trial_id=f"{task}-codex",
                action_case_id=f"{task}_case",
            )
        except BaseException:
            self.close()
            raise
        self._active_policy = policy
        return self._client

    def _positions(self) -> dict[str, list[float]]:
        positions: dict[str, list[float]] = {}
        tools = self.adapter.tools()
        for label in tools["list_objects"]():
            try:
                position, _ = tools["get_object_pose"](label)
                positions[label] = [float(x) for x in position]
            except Exception:
                continue
        return positions

    def _event_poses(self, labels: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
        """Snapshot finite public poses only for explicitly allowlisted labels."""

        from roboenv_runtime.adapters.robodojo.policy_events import pose_snapshot

        return pose_snapshot(self.adapter.tools()["get_object_pose"], labels)

    def _event_grippers(self) -> dict[str, dict[str, Any]]:
        """Snapshot finite public gripper poses and widths for owner detection."""

        from roboenv_runtime.adapters.robodojo.policy_events import gripper_snapshot

        tools = self.adapter.tools()
        pose_tool = tools.get("get_gripper_pose")
        state_tool = tools.get("get_gripper_state")
        if not callable(pose_tool) or not callable(state_tool):
            return {}
        return gripper_snapshot(pose_tool, state_tool)

    def _gripper_poses(self) -> dict[str, dict[str, list[float]]]:
        poses: dict[str, dict[str, list[float]]] = {}
        tools = self.adapter.tools()
        pose_tool = tools.get("get_gripper_pose")
        state_tool = tools.get("get_gripper_state")
        if not callable(pose_tool) and not callable(state_tool):
            return poses
        for arm in ("left", "right"):
            try:
                width: float | None = None
                raw_pose: Any
                if callable(state_tool):
                    state = state_tool(arm)
                    raw_pose = state["tip_pose"]
                    width = float(state["width"])
                else:
                    raw_pose = pose_tool(arm)
                vector = np.asarray(raw_pose, dtype=float)
                if vector.shape == (7,):
                    position = vector[:3]
                    quaternion = vector[3:]
                elif isinstance(raw_pose, (list, tuple)) and len(raw_pose) == 2:
                    position = np.asarray(raw_pose[0], dtype=float).reshape(3)
                    quaternion = np.asarray(raw_pose[1], dtype=float).reshape(4)
                else:
                    continue
                pose = {
                    "position": [float(value) for value in position],
                    "quaternion": [float(value) for value in quaternion],
                }
                if width is not None and np.isfinite(width):
                    pose["width"] = width
                poses[arm] = pose
            except Exception:
                continue
        return poses

    @staticmethod
    def _moved(before: dict[str, list[float]], after: dict[str, list[float]]) -> dict[str, float]:
        result: dict[str, float] = {}
        for label in sorted(before.keys() & after.keys()):
            distance = float(np.linalg.norm(np.asarray(after[label]) - np.asarray(before[label])))
            if distance > 0.001:
                result[label] = round(distance, 6)
        return result

    def _execute_native_frame(
        self, action: Any, *, reserve_sim_steps: int
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Execute one validated native action through the single reward clock.

        Policy, replay and hold modes all use this method so raw EvalEnv steps
        are suppressed under native_action exactly once, while simulator
        budget accounting and capture advance exactly once per action frame.
        """

        normalized = _validated_joint_action(action)
        env = self.env
        interval = int(getattr(env.obs_manager, "collect_interval", 10) or 10)
        if interval <= 0:
            raise RuntimeError("collect_interval must be positive")
        remaining = _remaining_sim_steps(self.adapter.budget())
        if remaining < int(reserve_sim_steps) + interval:
            return False, "budget_reserve", _jsonable(normalized)
        native = getattr(self.clock, "native_action", None)
        context = native() if callable(native) else nullcontext()
        with context:
            env.take_action(normalized)
        self.adapter._env._sim_step_count += interval
        capture = getattr(self.adapter._env, "_maybe_capture", None)
        if callable(capture):
            capture()
        return True, None, _jsonable(normalized)

    def _direct_mode_prelude(
        self,
        *,
        allow_midstate: bool,
        reserve_sim_steps: int,
        stop_when_latched: bool,
    ) -> None:
        if not isinstance(allow_midstate, bool):
            raise ValueError("allow_midstate must be boolean")
        if (
            isinstance(reserve_sim_steps, bool)
            or not isinstance(reserve_sim_steps, int)
            or reserve_sim_steps < 0
        ):
            raise ValueError("reserve_sim_steps must be a non-negative integer")
        if not isinstance(stop_when_latched, bool):
            raise ValueError("stop_when_latched must be boolean")
        already_taken = int(np.asarray(self.env.take_action_cnt).reshape(-1)[0])
        if already_taken and not allow_midstate:
            raise RuntimeError(
                f"VLA mid-state handoff refused after {already_taken} action frames; "
                "set allow_midstate=true only after deliberate verification"
            )

    def _direct_mode_result(
        self,
        *,
        mode: str,
        executed: list[Any],
        requested_count: int,
        stop_reason: str,
        before_positions: dict[str, list[float]],
        before_grippers: dict[str, dict[str, list[float]]],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        after_positions = self._positions()
        after_grippers = self._gripper_poses()
        executed_sha = _canonical_actions_sha256(executed)
        report = {
            "mode": mode,
            "start_action": 0,
            "end_action": len(executed),
            "requested_actions": requested_count,
            "stop_reason": stop_reason,
            "objects_moved_m": self._moved(before_positions, after_positions),
            "grippers_moved_equivalent_m": _gripper_motion(before_grippers, after_grippers),
            "provenance": {
                **provenance,
                "executed_actions_sha256": executed_sha,
            },
            "score": self.clock.peek(),
        }
        artifact = self._save_actions(executed, report)
        return {
            "mode": mode,
            "policy": None,
            "actions_taken": len(executed),
            "requested_actions": requested_count,
            "stop_reason": stop_reason,
            "objects_moved_m": report["objects_moved_m"],
            "grippers_moved_equivalent_m": report["grippers_moved_equivalent_m"],
            "score": report["score"],
            "budget": self.adapter.budget(),
            "action_artifacts": [str(artifact)],
            "provenance": report["provenance"],
            "rng_replay": None,
        }

    def run_replay(
        self,
        *,
        actions: list[Any],
        allow_midstate: bool = False,
        reserve_sim_steps: int = 0,
        stop_when_latched: bool = True,
    ) -> dict[str, Any]:
        """Replay only inline finite canonical actions; paths are not accepted."""

        self._direct_mode_prelude(
            allow_midstate=allow_midstate,
            reserve_sim_steps=reserve_sim_steps,
            stop_when_latched=stop_when_latched,
        )
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_VLA_ACTIONS:
            raise ValueError(
                f"replay actions must be an inline list of 1..{MAX_VLA_ACTIONS} actions"
            )
        validated = [_validated_joint_action(action) for action in actions]
        canonical_input = [_jsonable(action) for action in validated]
        input_sha = _canonical_actions_sha256(canonical_input)
        before_positions = self._positions()
        before_grippers = self._gripper_poses()
        executed: list[Any] = []
        stop_reason = "action_limit"
        env = self.env
        for action in validated:
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break
            ok, reason, canonical = self._execute_native_frame(
                action, reserve_sim_steps=reserve_sim_steps
            )
            if not ok:
                stop_reason = reason or "budget_reserve"
                break
            executed.append(canonical)
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break
        return self._direct_mode_result(
            mode="replay",
            executed=executed,
            requested_count=len(validated),
            stop_reason=stop_reason,
            before_positions=before_positions,
            before_grippers=before_grippers,
            provenance={
                "source": "inline_actions",
                "input_actions_sha256": input_sha,
                "input_actions_count": len(canonical_input),
            },
        )

    @staticmethod
    def _current_hold_action(env: Any) -> Any:
        """Read env0's current canonical joint action from public observation."""

        observation = env.get_obs()
        if isinstance(observation, Mapping):
            if "action" not in observation:
                raise RuntimeError("hold observation lacks canonical action")
            return observation["action"]
        if (
            isinstance(observation, (list, tuple))
            and len(observation) == 1
            and isinstance(observation[0], Mapping)
            and "action" in observation[0]
        ):
            return observation[0]["action"]
        raise RuntimeError("hold observation must be env0 mapping or one-item env batch")

    def run_hold(
        self,
        *,
        n_actions: int,
        allow_midstate: bool = False,
        reserve_sim_steps: int = 0,
        stop_when_latched: bool = True,
    ) -> dict[str, Any]:
        """Advance interactive callbacks with env0's current target action."""

        self._direct_mode_prelude(
            allow_midstate=allow_midstate,
            reserve_sim_steps=reserve_sim_steps,
            stop_when_latched=stop_when_latched,
        )
        if (
            isinstance(n_actions, bool)
            or not isinstance(n_actions, int)
            or not 1 <= n_actions <= MAX_VLA_ACTIONS
        ):
            raise ValueError(f"hold n_actions must be an integer in 1..{MAX_VLA_ACTIONS}")
        before_positions = self._positions()
        before_grippers = self._gripper_poses()
        executed: list[Any] = []
        stop_reason = "action_limit"
        env = self.env
        for _ in range(n_actions):
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break
            action = _validated_joint_action(self._current_hold_action(env))
            ok, reason, canonical = self._execute_native_frame(
                action, reserve_sim_steps=reserve_sim_steps
            )
            if not ok:
                stop_reason = reason or "budget_reserve"
                break
            executed.append(canonical)
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break
        executed_sha = _canonical_actions_sha256(executed)
        return self._direct_mode_result(
            mode="hold",
            executed=executed,
            requested_count=n_actions,
            stop_reason=stop_reason,
            before_positions=before_positions,
            before_grippers=before_grippers,
            provenance={
                "source": "env0_current_action",
                "observed_actions_sha256": executed_sha,
                "observed_actions_count": len(executed),
            },
        )

    def run_until_event(
        self,
        *,
        n_actions: int = 110,
        candidate_labels: list[str] | tuple[str, ...] | None = None,
        protected_labels: list[str] | tuple[str, ...] | None = None,
        allow_midstate: bool = False,
        reserve_sim_steps: int = 0,
        stop_when_latched: bool = True,
        owner_rise_m: float = 0.035,
        owner_near_m: float = 0.17,
        owner_width_m: float = 0.125,
        owner_arm_margin_m: float = 0.015,
        owner_stable_actions: int = 2,
        owner_position_drift_m: float = 0.03,
        owner_rotation_drift_deg: float = 25.0,
        protected_displacement_m: float = 0.02,
        protected_rotation_deg: float = 15.0,
    ) -> dict[str, Any]:
        """Run bounded G05 replans while checking public events after every action."""

        from roboenv_runtime.adapters.robodojo.policy_events import (
            PolicyEventTracker,
            validate_event_config,
        )

        self._direct_mode_prelude(
            allow_midstate=allow_midstate,
            reserve_sim_steps=reserve_sim_steps,
            stop_when_latched=stop_when_latched,
        )
        if bool(getattr(self.clock, "submitted", False)):
            raise RuntimeError("policy event rollout is frozen after score submission")
        tools = self.adapter.tools()
        live_labels = set(tools["list_objects"]())
        config = validate_event_config(
            n_actions=n_actions,
            candidate_labels=candidate_labels,
            protected_labels=protected_labels,
            live_labels=live_labels,
            owner_rise_m=owner_rise_m,
            owner_near_m=owner_near_m,
            owner_width_m=owner_width_m,
            owner_arm_margin_m=owner_arm_margin_m,
            owner_stable_actions=owner_stable_actions,
            owner_position_drift_m=owner_position_drift_m,
            owner_rotation_drift_deg=owner_rotation_drift_deg,
            protected_displacement_m=protected_displacement_m,
            protected_rotation_deg=protected_rotation_deg,
        )
        labels = tuple(dict.fromkeys((*config["candidate_labels"], *config["protected_labels"])))
        initial_poses = self._event_poses(labels)
        initial_grippers = self._event_grippers()
        tracker = PolicyEventTracker(config, initial_poses, initial_grippers)

        client = self._ensure_client("G05")
        if self._needs_reset:
            client.call(func_name="reset")
            self._needs_reset = False
        env = self.env
        before_positions = self._positions()
        before_grippers = initial_grippers
        executed: list[Any] = []
        event: dict[str, Any] | None = None
        policy_inferences = 0
        stop_reason = "action_limit"
        interval = int(getattr(getattr(env, "obs_manager", None), "collect_interval", 10) or 10)
        if interval <= 0:
            raise RuntimeError("policy event collect_interval must be positive")

        def pre_action_stop() -> str | None:
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                return "environment_end"
            current_frame = int(np.asarray(env.take_action_cnt).reshape(-1)[0])
            step_lim = int(getattr(env, "step_lim", 0) or 0)
            if step_lim > 0 and current_frame >= step_lim:
                return "task_action_limit"
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                return "success_latched"
            remaining = _remaining_sim_steps(self.adapter.budget())
            if remaining < int(reserve_sim_steps) + interval:
                return "budget_reserve"
            return None

        halted = False
        while len(executed) < config["n_actions"]:
            reason = pre_action_stop()
            if reason is not None:
                stop_reason = reason
                break
            client.call(func_name="update_obs", obs=env.get_obs())
            proposed = _action_batch(client.call(func_name="get_action"))
            policy_inferences += 1
            if not proposed:
                stop_reason = "empty_policy_plan"
                break
            for action in proposed:
                reason = pre_action_stop()
                if reason is not None:
                    stop_reason = reason
                    halted = True
                    break
                ok, reason, canonical = self._execute_native_frame(
                    action, reserve_sim_steps=reserve_sim_steps
                )
                if not ok:
                    stop_reason = reason or "budget_reserve"
                    halted = True
                    break
                executed.append(canonical)
                event = tracker.observe(
                    self._event_poses(labels),
                    self._event_grippers(),
                    action_index=len(executed),
                )
                if event is not None:
                    stop_reason = str(event["kind"])
                    halted = True
                    break
                if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                    stop_reason = "environment_end"
                    halted = True
                    break
                if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                    stop_reason = "success_latched"
                    halted = True
                    break
                if len(executed) >= config["n_actions"]:
                    stop_reason = "action_limit"
                    halted = True
                    break
            if halted:
                break

        after_positions = self._positions()
        after_grippers = self._event_grippers()
        action_sha = _canonical_actions_sha256(executed)
        report = {
            "mode": "policy_event",
            "policy": "G05",
            "actions_taken": len(executed),
            "requested_actions": config["n_actions"],
            "policy_inferences": policy_inferences,
            "stop_reason": stop_reason,
            "owner": event if event and event["kind"] == "owner_witness" else None,
            "protection": event if event and event["kind"].startswith("protected_") else None,
            "objects_moved_m": self._moved(before_positions, after_positions),
            "grippers_moved_equivalent_m": _gripper_motion(before_grippers, after_grippers),
            "budget": self.adapter.budget(),
            "provenance": {
                "source": "bounded_continuous_g05_replanning_event_stop",
                "executed_actions_sha256": action_sha,
                "executed_actions_count": len(executed),
                "event_config": _jsonable(config),
            },
        }
        artifact = self._save_event_report(report)
        return {**report, "event_artifacts": [str(artifact)]}

    def _save_event_report(self, report: dict[str, Any]) -> Path:
        """Persist sanitized event evidence without policy action values."""

        self._sequence += 1
        path = self.vla_dir / f"event_{self._sequence:04d}.json"
        _atomic_json(path, {"report": report})
        return path

    def _save_actions(self, actions: list[Any], report: dict[str, Any]) -> Path:
        self._sequence += 1
        path = self.vla_dir / f"actions_{self._sequence:04d}.json"
        _atomic_json(path, {"actions": actions, "report": report})
        return path

    def run(
        self,
        n_actions: int = 20,
        policy: str = "G05",
        chunk_size: int = 20,
        allow_midstate: bool = False,
        stop_on: dict[str, Any] | None = None,
        reserve_sim_steps: int = 0,
        policy_rng_stride: int = 1,
        policy_rng_initial_advance: int = 0,
        policy_action_horizon: int | None = None,
    ) -> dict[str, Any]:
        """Run bounded policy chunks with measurable, task-agnostic stops.

        ``stop_on`` supports ``stall_motion_m`` (default 0.02),
        ``stall_chunks`` (default 2), ``target_object`` plus
        ``target_displacement_m``, and ``stop_when_latched`` (default true).
        VLA is rejected on a changed world unless ``allow_midstate`` is explicit.
        ``policy_rng_stride`` reproduces the first environment of a sequential
        batched stochastic-policy evaluation. ``policy_rng_initial_advance``
        selects a later auditable rollout without relying on process RNG luck.
        """
        if not 0 < n_actions <= MAX_VLA_ACTIONS:
            raise ValueError(f"n_actions must be in 1..{MAX_VLA_ACTIONS}")
        if not 0 < chunk_size <= min(n_actions, MAX_VLA_CHUNK):
            raise ValueError(f"chunk_size must be in 1..min(n_actions,{MAX_VLA_CHUNK})")
        if reserve_sim_steps < 0:
            raise ValueError("reserve_sim_steps must be non-negative")
        policy = _validated_policy(policy)
        if (
            isinstance(policy_rng_stride, bool)
            or not isinstance(policy_rng_stride, int)
            or not 1 <= policy_rng_stride <= 256
        ):
            raise ValueError("policy_rng_stride must be an integer in 1..256")
        if (
            isinstance(policy_rng_initial_advance, bool)
            or not isinstance(policy_rng_initial_advance, int)
            or not 0 <= policy_rng_initial_advance <= 255
        ):
            raise ValueError("policy_rng_initial_advance must be an integer in 0..255")
        if policy_action_horizon is not None and (
            isinstance(policy_action_horizon, bool)
            or not isinstance(policy_action_horizon, int)
            or not 1 <= policy_action_horizon <= MAX_VLA_CHUNK
        ):
            raise ValueError(f"policy_action_horizon must be in 1..{MAX_VLA_CHUNK}")
        rng_replay_enabled = policy_rng_stride > 1 or policy_rng_initial_advance > 0
        if rng_replay_enabled:
            if policy != "Pi_05":
                raise ValueError("policy RNG replay is currently locked to Pi_05")
            if policy_action_horizon is None:
                raise ValueError(
                    "policy_action_horizon is required when policy RNG replay is enabled"
                )
            if chunk_size != policy_action_horizon:
                raise ValueError(
                    "chunk_size must equal policy_action_horizon for RNG-stride replay"
                )
        options = dict(stop_on or {})
        env = self.env
        already_taken = int(np.asarray(env.take_action_cnt).reshape(-1)[0])
        if already_taken and not allow_midstate:
            raise RuntimeError(
                f"VLA mid-state handoff refused after {already_taken} action frames; "
                "set allow_midstate=true only after deliberate verification"
            )
        client = self._ensure_client(policy)
        if self._needs_reset:
            client.call(func_name="reset")
            self._needs_reset = False

        before_all = self._positions()
        before_chunk = before_all
        before_all_grippers = self._gripper_poses()
        before_chunk_grippers = before_all_grippers
        target = options.get("target_object")
        target_origin = before_all.get(str(target)) if target is not None else None
        stall_threshold = float(options.get("stall_motion_m", 0.02))
        stall_limit = int(options.get("stall_chunks", 2))
        stall_warmup = int(options.get("stall_warmup_actions", 40))
        if stall_threshold < 0 or stall_limit <= 0 or stall_warmup < 0:
            raise ValueError(
                "stall_motion_m and stall_warmup_actions must be non-negative "
                "and stall_chunks must be positive"
            )
        stop_when_latched = bool(options.get("stop_when_latched", True))
        stall_count = 0
        taken = 0
        inference_index = 0
        saved: list[str] = []
        stop_reason = "action_limit"

        while taken < n_actions:
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            budget = self.adapter.budget()
            remaining = _remaining_sim_steps(budget)
            if remaining <= int(reserve_sim_steps):
                stop_reason = "budget_reserve"
                break
            if stop_when_latched and bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break

            obs = env.get_obs()
            client.call(func_name="update_obs", obs=obs)
            rng_call_index: int | None = None
            if rng_replay_enabled:
                advance = (
                    policy_rng_initial_advance if inference_index == 0 else policy_rng_stride - 1
                )
                rng_call_index = (
                    1 + policy_rng_initial_advance + inference_index * policy_rng_stride
                )
                proposed_raw = client.call(
                    func_name="get_action_with_rng_advance",
                    obs={"advance": advance},
                )
            else:
                proposed_raw = client.call(func_name="get_action")
            proposed = _action_batch(proposed_raw)
            if rng_replay_enabled and len(proposed) != policy_action_horizon:
                raise RuntimeError(
                    f"policy returned horizon {len(proposed)}; expected "
                    f"{policy_action_horizon} for RNG-stride replay"
                )
            inference_index += 1
            if not proposed:
                stop_reason = "empty_policy_chunk"
                break
            executed: list[Any] = []
            budget_blocked = False
            for action in proposed:
                ok, reason, canonical = self._execute_native_frame(
                    action, reserve_sim_steps=reserve_sim_steps
                )
                if not ok:
                    stop_reason = reason or "budget_reserve"
                    budget_blocked = True
                    break
                executed.append(canonical)
                taken += 1
                if taken >= n_actions or len(executed) >= chunk_size:
                    break
                if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                    break
                if bool(getattr(self.clock, "latched", False)):
                    break

            if budget_blocked and not executed:
                break
            after_chunk = self._positions()
            after_chunk_grippers = self._gripper_poses()
            moved = self._moved(before_chunk, after_chunk)
            gripper_moved = _gripper_motion(before_chunk_grippers, after_chunk_grippers)
            max_motion = max([*moved.values(), *gripper_moved.values()], default=0.0)
            if taken < stall_warmup:
                stall_count = 0
            else:
                stall_count = stall_count + 1 if max_motion < stall_threshold else 0
            chunk_report = {
                "start_action": taken - len(executed),
                "end_action": taken,
                "objects_moved_m": moved,
                "grippers_moved_equivalent_m": gripper_moved,
                "policy_inference_index": inference_index - 1,
                "policy_rng_call_index": rng_call_index,
                "score": self.clock.peek(),
            }
            saved.append(str(self._save_actions(executed, chunk_report)))
            before_chunk = after_chunk
            before_chunk_grippers = after_chunk_grippers

            if budget_blocked:
                break
            if bool(getattr(self.clock, "latched", False)):
                stop_reason = "success_latched"
                break
            if target_origin is not None and str(target) in after_chunk:
                displacement = float(
                    np.linalg.norm(np.asarray(after_chunk[str(target)]) - np.asarray(target_origin))
                )
                if displacement >= float(options.get("target_displacement_m", float("inf"))):
                    stop_reason = "target_displacement"
                    break
            if stall_count >= stall_limit:
                stop_reason = "motion_stall"
                break

        after_all = self._positions()
        after_all_grippers = self._gripper_poses()
        return {
            "mode": "policy",
            "policy": policy,
            "actions_taken": taken,
            "stop_reason": stop_reason,
            "objects_moved_m": self._moved(before_all, after_all),
            "grippers_moved_equivalent_m": _gripper_motion(before_all_grippers, after_all_grippers),
            "score": self.clock.peek(),
            "budget": self.adapter.budget(),
            "action_artifacts": saved,
            "rng_replay": (
                {
                    "mode": "server_rng_advance",
                    "stride": policy_rng_stride,
                    "initial_advance": policy_rng_initial_advance,
                    "action_horizon": policy_action_horizon,
                    "inferences_used": inference_index,
                }
                if rng_replay_enabled
                else None
            ),
        }

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._server is not None and self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._server.kill()
                self._server.wait(timeout=5)
        self._server = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def _dispatch(self: "VlaManager", args: dict) -> Any:
    """Route the worker's allowlisted policy, replay, hold, or event mode."""
    args = dict(args)
    mode = args.pop("mode", "policy")
    if mode == "policy":
        return self.run(**args)
    if mode == "replay":
        return self.run_replay(**args)
    if mode == "hold":
        return self.run_hold(**args)
    if mode == "event":
        return self.run_until_event(**args)
    raise ValueError("vla mode must be exactly policy, replay, hold, or event")


VlaManager.dispatch = _dispatch  # type: ignore[attr-defined]
