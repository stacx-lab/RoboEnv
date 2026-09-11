"""Source-backed RoboDojo EvalEnv lifecycle and retained primitive operations.

The typed device backend subclasses this owner-side wrapper and sends absolute
joint targets through the native EvalEnv control cadence. The injected client
avoids a policy server dependency. Public bindings expose sensors and measured
robot telemetry; scene state and the native evaluator remain private.

Primitive planning methods are retained for explicit legacy callers. They are
not registered on the default device interface. Preparation uses the isolated
Isaac Sim 5.1 recipe and pinned RoboDojo sources; no cap-x package is required.
"""

# Historical origin: third_party/cap-x/capx/envs/simulators/robodojo.py, 2026-08-26.

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

# --------------------------------------------------------------------------
# RoboDojo checkout + Isaac boot (both are process-global and one-shot)
# --------------------------------------------------------------------------
from roboenv_runtime.paths import repo_root, robodojo_root, runs_root

ROBODOJO_ROOT = str(robodojo_root())  # $ROBODOJO_ROOT or <repo>/RoboDojo

_APP = None  # the SimulationApp handle; Isaac allows exactly one per process


def _boot_isaac(headless: bool = True, enable_cameras: bool = True):
    """Start Isaac Sim once. Must happen before any `env.*`/`task.*` import."""
    global _APP
    if _APP is not None:
        return _APP

    if not os.path.isdir(ROBODOJO_ROOT):
        raise RuntimeError(f"RoboDojo checkout not found at {ROBODOJO_ROOT}; set $ROBODOJO_ROOT")
    # The native EvalEnv accepts an injected client; no policy checkout is needed.
    if ROBODOJO_ROOT in sys.path:
        sys.path.remove(ROBODOJO_ROOT)
    sys.path.insert(0, ROBODOJO_ROOT)
    # EvalEnv uses this ID for resume state and temporary video cleanup. Distinct
    # workers must not reuse another world's native output directory.
    os.environ.setdefault("ROBODOJO_RUN_ID", f"roboenv-{uuid4().hex}")

    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    argv = []
    if headless:
        argv.append("--headless")
    if enable_cameras:
        argv.append("--enable_cameras")
    argv += [
        "--kit_args",
        " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
    ]
    # AppLauncher parses sys.argv too; hide the harness's own flags from it.
    saved_argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        _APP = AppLauncher(parser.parse_args(argv)).app
    finally:
        sys.argv = saved_argv
    return _APP


def list_robodojo_tasks() -> list[str]:
    """Task names = the module stems under task/RoboDojo/tasks (class == module)."""
    tasks_dir = os.path.join(ROBODOJO_ROOT, "task", "RoboDojo", "tasks")
    if not os.path.isdir(tasks_dir):
        return []
    return sorted(
        f[:-3] for f in os.listdir(tasks_dir) if f.endswith(".py") and not f.startswith("_")
    )


class _NoopModelClient:
    """Stands in for XPolicyLab's WsModelClient so no policy server is needed."""

    def call(self, *a, **k):
        return None

    def close(self, *a, **k):
        return None

    def __getattr__(self, _name):
        return lambda *a, **k: None


# --------------------------------------------------------------------------
# Env
# --------------------------------------------------------------------------


class RoboDojoEnv:
    """One RoboDojo task, driven in-process.

    Args:
        task_name: e.g. "pack_objects_into_box" (see list_robodojo_tasks()).
        seed: layout seed; RoboDojo replays saved layouts, so this selects a scene.
        env_cfg_type: env_cfg/<name>.yml stem, such as "arx_x5" or "yam".
        max_steps: sim-step budget; falls back to the task's own `step_lim`.
    """

    ARMS = ("left", "right")

    def __init__(
        self,
        task_name: str = "pack_objects_into_box",
        seed: int = 0,
        env_cfg_type: str = "arx_x5",
        privileged: bool = True,
        enable_render: bool = True,
        headless: bool = True,
        max_steps: int = 100_000,
        respect_task_step_lim: bool = True,
        subsample_rate: int = 40,
        api_exclude: tuple = (),
        **_ignored: Any,
    ) -> None:
        self.privileged = privileged
        # names the API layer should NOT expose this run (tool-ablation
        # experiments, e.g. hide get_gripper_state to test blind discovery)
        self.api_exclude = tuple(api_exclude or ())
        self.enable_render = enable_render
        self.max_steps = max_steps
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.seed = seed

        print(f"[robodojo] wrapper V16 loaded from {__file__}", flush=True)
        self._app = _boot_isaac(headless=headless)
        self._env = None
        self._sim_step_count = 0

        # Video capture for explicitly selected legacy callers.
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._subsample_rate = subsample_rate

        self._build()
        # `step_lim` counts ACTION FRAMES (take_action_cnt), each of which is
        # `collect_interval` (=10) raw sim steps — so the packing task's 1300
        # means ~13k sim steps / ~52 sim-seconds. Track the budget in sim steps
        # and never let it cut a single motion short (see _run_control).
        frames = int(getattr(self._env, "step_lim", 0) or 0)
        interval = int(
            getattr(getattr(self._env, "obs_manager", None), "collect_interval", 10) or 10
        )
        # benchmark episodes truncate at the task's own step_lim; PRACTICE
        # runs may take a longer leash (respect_task_step_lim=False) — any
        # "solved within benchmark limits" claim must re-run with the clamp
        if frames > 0 and respect_task_step_lim:
            self.max_steps = min(max_steps, frames * interval)

    # ------------------------------------------------------------------ build

    def _assemble_config(self, seed: int):
        """Replicates src/eval_client/main.py:249-341."""
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from omegaconf import OmegaConf
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self.env_cfg_type + ".yml"))
        eval_cfg["task_name"] = self.task_name
        eval_cfg["num_envs"] = 1
        eval_cfg["device_id"] = 0
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = "roboenv"
        eval_cfg["additional_info"] = ""
        eval_cfg["seed"] = seed
        eval_cfg["physx_monitor_enabled"] = False

        deploy_cfg = {
            "policy_name": "roboenv",
            "port": 0,
            "host": "localhost",
            "protocol": "ws",
            "policy_server_url": "ws://localhost:0",
            "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
            "trial_id": f"{self.task_name}-{os.environ['ROBODOJO_RUN_ID']}",
            "action_case_id": f"{self.task_name}_case",
            "repeat_index": None,
        }

        cfg = OmegaConf.create(
            {
                "sim": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")
                ),
                "scene": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")
                ),
                "camera": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")
                ),
                "robot": load_yaml(
                    os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")
                ),
                "task_env": load_yaml(
                    task_registry.task_config_path(
                        os.path.join(benchmark_path, "config"), self.task_name
                    )
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(cfg, "eval_cfg.num_envs", 1, force_add=True)
        cfg = process_randomization(cfg)
        cfg, eval_num = process_config(cfg, task_name=self.task_name)
        eval_cfg["eval_num"] = eval_num
        OmegaConf.update(
            cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        cfg.sim.seed = [0]
        return cfg

    def _build(self) -> None:
        import src.eval_client.eval_env as ee_mod

        cfg = self._assemble_config(self.seed)
        # The simulator-only path has no dependency on an optional policy package.
        self._env = ee_mod.create_eval_env(cfg, self._app, model_client=_NoopModelClient())

    # ------------------------------------------------------------- gym surface

    def _layout_id_for(self, seed: int) -> int:
        """Map a trial seed onto one of the task's saved native layout ids."""
        sm = self._env.seed_manager
        ids = list(getattr(sm, "seed_list", None) or sorted(getattr(sm, "seed_info", {}).keys()))
        if not ids:
            return int(seed)
        return int(ids[int(seed) % len(ids)])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._vla_needs_reset = True
        if seed is not None:
            self.seed = seed
        self._sim_step_count = 0
        self._frame_buffer.clear()
        self._wrist_frame_buffer.clear()

        # RoboDojo replays saved layouts. Do NOT use seed_manager.get_seeds():
        # it walks an internal cursor, so successive resets silently load
        # *different* scenes. Address the layout by id instead, so reset(seed=k)
        # is reproducible the way CaP-X's trial loop assumes.
        seeds = [self._layout_id_for(self.seed)]
        if self._soft_reset_possible(seeds):
            self._soft_reset()
        else:
            # forget the built layout BEFORE the relayout: a reset that raises half-way
            # must not leave a stale id that turns the next reset into a soft reset on a
            # half-reloaded scene
            self._loaded_layout = None
            self._env.env_seeds = seeds
            self._env.reset(seed=seeds)
            self._loaded_layout = list(seeds)

        # Arm the task's own reward/score state machines (run_eval does this).
        for hook in ("run_reward", "get_score"):
            fn = getattr(self._env, hook, None)
            if callable(fn):
                fn()

        # home joint configuration per arm, for go_home() / back-to-origin scoring
        self._home_joints = {}
        for _arm in ("left", "right"):
            try:
                _rb = self._robot(_arm)
                _rm = self._env.robot_manager
                self._home_joints[_arm] = (
                    np.asarray(_rm.get_joint(_rb, [0])[0], dtype=float).reshape(-1).copy()
                )
            except Exception:
                pass

        obs = self.get_observation()
        info = {"task_prompt": self.task_name.replace("_", " "), "layout_seeds": seeds}
        info["reset"] = getattr(self, "_last_reset_report", None) or {"mode": "full"}
        self._last_reset_report = None
        return obs, info

    def _soft_reset_possible(self, seeds: list[int]) -> bool:
        """Same layout already built → restore poses instead of reloading the layout.

        RoboDojo's full reset regenerates instance names from a counter that never
        resets (``laptop_0_6`` → ``laptop_0_15``) while articulated objects are only
        relocated, never respawned, so after a same-task reset every lookup of an
        articulated object fails ("not found in env 0", measured 2026-08-26).  Restoring
        the saved poses of the objects that already exist sidesteps that entirely.
        """
        return (
            getattr(self, "_loaded_layout", None) == list(seeds)
            and getattr(self._env, "scene_manager", None) is not None
        )

    #: bound on the settle loop of a soft reset (0.0195 rad/step → 600 steps ≈ 11.7 rad)
    SOFT_RESET_HOME_CAP = 600
    SOFT_RESET_HOME_TOL = 0.02

    def _arm_home_error(self) -> float | None:
        """Largest |q - default| over the arm joints, or None when unreadable."""
        try:
            rm = self._env.robot_manager
            worst = 0.0
            for robot, key in zip(rm.robot_list, rm.robot_key):
                if getattr(robot, "robot_type", "arm") != "arm":
                    continue
                q = np.asarray(key.data.joint_pos.cpu(), dtype=float).reshape(-1)
                q0 = np.asarray(key.data.default_joint_pos.cpu(), dtype=float).reshape(-1)
                idx = getattr(robot, "arm_joint_indices", None)
                if idx is not None:
                    idx = np.asarray(idx, dtype=int).reshape(-1)
                    q, q0 = q[idx], q0[idx]
                worst = max(worst, float(np.max(np.abs(q - q0))))
            return worst
        except Exception:
            return None

    def _teleport_arms_home(self) -> int:
        """Write every arm's default joint state (and a zero velocity) straight into the
        simulation (Isaac Lab ``Articulation.write_joint_state_to_sim``) and re-issue it
        as the position target — every DoF including the gripper, so the jaws open and
        anything held is dropped where the arm stands.  Runs on EVERY same-task reset
        (not a model tool; not tied to the privileged worker): it is what makes a soft
        reset recover an arm that the position controller can no longer move.  Robots
        whose ``robot_type`` is not ``"arm"`` are skipped.  Returns how many arms were
        written; a failure is reported per arm.
        """
        try:
            import torch
        except Exception as exc:
            print(f"[robodojo] soft reset: joint teleport unavailable ({exc})", flush=True)
            return 0
        rm = self._env.robot_manager
        written = 0
        for robot, key in zip(rm.robot_list, rm.robot_key, strict=False):
            if getattr(robot, "robot_type", "arm") != "arm":
                continue
            try:
                q0 = key.data.default_joint_pos.clone()
                key.write_joint_state_to_sim(q0, torch.zeros_like(q0))
                key.set_joint_position_target(q0)
                if callable(getattr(key, "write_data_to_sim", None)):
                    key.write_data_to_sim()
                written += 1
            except Exception as exc:
                name = getattr(robot, "robot_name", "?")
                print(
                    f"[robodojo] soft reset: joint teleport failed for {name} ({exc})", flush=True
                )
        return written

    def _settle_arms_home(self, min_steps: int, max_steps: int) -> tuple[float | None, int]:
        """Step until every arm joint is within tolerance of its default (bounded);
        returns ``(home error in rad or None, steps used)``."""
        env = self._env
        steps = 0
        for i in range(int(max_steps)):
            env.sim_step(render=False)
            steps = i + 1
            if i + 1 < min_steps:
                continue
            err = self._arm_home_error()
            if err is None or err < self.SOFT_RESET_HOME_TOL:
                return err, steps
        err = self._arm_home_error()
        if err is not None and err >= self.SOFT_RESET_HOME_TOL:
            print(
                f"[robodojo] soft reset: arms still {err:.2f} rad from home after "
                f"{max_steps} steps; the layout was restored with an arm moving",
                flush=True,
            )
        return err, steps

    def _soft_reset(self) -> None:
        env = self._env
        # Arms first, objects second: restoring the saved object poses while an arm
        # is still parked where an object spawns teleports the object into the
        # gripper — the jammed joints then never reach their targets and the
        # "home" captured afterwards is wrong (measured 2026-08-27, both arms
        # 0.3-1.0 rad off after a reset with the right hand over the pliers' spawn).
        for hook in ("reset",):
            fn = getattr(env.robot_manager, hook, None)
            if callable(fn):
                fn()
        # Write the default joint state straight into the simulation as well: a position
        # target alone cannot bring back an arm that has wound itself into the table or
        # its own base (measured 2026-08-27: the left X5 elbow at 5.5 rad below the table
        # top stayed there through two go_home calls and a soft reset; the direct write
        # put both arms home in 30 steps).
        teleported = self._teleport_arms_home()
        # The arms slew home at ~0.0195 rad/step (X5, measured 2026-08-27), so a fixed
        # 120 steps only covers a 2.35 rad excursion: keep stepping (bounded) until every
        # arm joint is within tolerance of its default, and say so when it is not.
        _err, steps_a = self._settle_arms_home(min_steps=120, max_steps=self.SOFT_RESET_HOME_CAP)
        env.scene_manager.apply_saved_poses(env_idx_list=[0])
        for _ in range(120):
            env.sim_step(render=False)
        err, steps_b = self._settle_arms_home(min_steps=0, max_steps=self.SOFT_RESET_HOME_CAP)
        # the settle loops are variable-length, so practice soft resets differ slightly
        # from each other and from a fresh-process reset: the counts are reported
        self._last_reset_report = {
            "mode": "soft",
            "teleported_arms": int(teleported),
            "home_error_rad": None if err is None else round(float(err), 4),
            "arms_home": bool(err is None or err < self.SOFT_RESET_HOME_TOL),
            "settle_steps": int(steps_a + 120 + steps_b),
        }
        for name in ("obs_manager", "camera_manager", "capture_manager"):
            mgr = getattr(env, name, None)
            if mgr is not None and callable(getattr(mgr, "reset", None)):
                mgr.reset()
        env.robot_manager.set_origin_endpose()
        env.robot_manager.set_robot_init_state()
        env.reward_manager.init_state()
        client = getattr(env, "model_client", None)
        if client is not None:
            try:
                client.call(func_name="reset")
            except Exception:  # noqa: BLE001  (no policy server attached in practice)
                pass

    def step(self, action=None):
        """Vestigial: control flows through the API layer, not through step()."""
        obs = self.get_observation()
        reward = self.compute_reward()
        return obs, reward, bool(self.task_completed()), self.budget_exhausted(), {}

    def _pump_reward(self) -> None:
        # Score lives in a state machine that ONLY advances inside
        # reward_manager.step() (RoboDojo's eval loop calls it every frame; we
        # bypass that loop). "transition" mode advances at most one tier per
        # step(), so pump several times — end-state conditions keep holding,
        # letting multiple earned tiers latch in one query.
        # Reset already called the task-level run_reward()/get_score() registrars.
        # Calling either registrar here appends a second copy of the task queues:
        # a legacy program can empty the real queue and read reward 1, then final
        # adapter.score() re-registers it and incorrectly returns 0.
        try:
            from roboenv_runtime.adapters.robodojo.score_clock import pump_registered_reward_manager

            pump_registered_reward_manager(self._env.reward_manager, ticks=5)
        except Exception:
            pass

    def compute_reward(self) -> float:
        """0..1, from RoboDojo's graded 0..100 score (10/25/50/100 tiers here)."""
        self._pump_reward()
        try:
            return float(self._env.reward_manager.get_score()[0]) / 100.0
        except Exception:
            return 0.0

    def task_completed(self) -> bool:
        self._pump_reward()
        try:
            return bool(self._env.reward_manager.get_reward(final_check=False)[0] >= 1.0)
        except Exception:
            return False

    def get_observation(self) -> dict[str, Any]:
        return self._raw_obs()

    def _raw_obs(self) -> dict[str, Any]:
        """Observation WITHOUT EvalEnv.get_obs's side effects.

        EvalEnv.get_obs_batch feeds every frame into per-camera ffmpeg
        subprocess pipes (`_stream_vision`); a stalled ffmpeg blocks the
        process forever, and even when healthy it triples the cost of a
        capture. Render + obs_manager directly instead.
        """
        self._env.render()
        return self._env.obs_manager.get_obs(env_idx_list=[0])[0]

    # ------------------------------------------------- privileged object state

    @property
    def _layout(self):
        return self._env.scene_manager.layout_manager

    #: layout groups holding manipulable task objects vs fixed scene props.
    #: "Geometry" fixtures (e.g. the camera stand bolted down behind the
    #: robots) stay queryable by name but are NOT advertised: an agent once
    #: burned its whole step budget trying to grasp the camera stand.
    OBJECT_GROUPS = ("Rigid", "Dynamic", "Articulation", "Garment")
    FIXTURE_GROUPS = ("Geometry",)

    def _records(self, groups: tuple[str, ...] | None = None) -> list[dict]:
        out = []
        for otype in groups or (self.OBJECT_GROUPS + self.FIXTURE_GROUPS):
            try:
                out.extend(self._layout.get_layout_records(0, otype))
            except Exception:
                continue
        return out

    def list_objects(self) -> list[str]:
        names = [r.get("label") for r in self._records(self.OBJECT_GROUPS) if r.get("label")]
        return sorted(dict.fromkeys(names))

    def _inst_for(self, label: str) -> str:
        inst = self._layout.get_instance_name(0, label)
        if inst:
            return inst
        for rec in self._records():
            if rec.get("label") == label or rec.get("inst_name") == label:
                return rec["inst_name"]
        raise KeyError(f"Object '{label}' not found. Available: {self.list_objects()}")

    @staticmethod
    def _quat_to_mat(q: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(q, dtype=float).reshape(4)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _tcp_offset(self, robot, quat) -> np.ndarray:
        """World-frame vector from link6 to the fingertips.

        The X5 gripper's tool axis is link6's LOCAL +X (measured in-sim: with
        identity-ish wrist the closed fingers meet ~14.5 cm from link6 along
        +X, and the fingers open along local Z). It is NOT local +Z — assuming
        +Z made every "top-down" grasp present the gripper sideways at wrist
        height, closing on air ~17 cm above the object. A top-down grasp is
        the Ry(+90 deg) family, e.g. wxyz [0.7071, 0, 0.7071, 0] (fingers
        opening along world Y) or [0.5, 0.5, 0.5, -0.5] (opening along world
        X), matching RoboTwin's GRASP_DIRECTION_DIC "top_down*" entries.
        """
        # embodiment-aware: X5's tool axis is link6 local +X at 0.1604 m
        # (measured, closed-finger table-touch); other robots declare their own
        # tcp_bias + tool_axis in robot_config.yml (YAM: gripper local -Z,
        # 0.135 m to the PAD midpoint).
        args = getattr(robot, "robot_args", {}) or {}
        bias = float(args.get("tcp_bias", 0.1604))
        axis = np.asarray(args.get("tool_axis", [1.0, 0.0, 0.0]), dtype=float).reshape(3)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        return self._quat_to_mat(quat) @ (bias * axis)

    @staticmethod
    def _np(v) -> np.ndarray:
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        return np.asarray(v, dtype=np.float64).reshape(-1)

    def get_object_pose(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        pos, rot = self._layout.get_instance_pose(env_idx=0, inst_name=self._inst_for(label))
        return self._np(pos), self._np(rot)

    def get_object_bbox(self, label: str) -> np.ndarray:
        verts = self._layout.get_instance_bbox_vertices(self._inst_for(label), 0)
        if verts is None:
            raise KeyError(f"No bounding box available for '{label}'")
        return np.asarray(verts, dtype=np.float64).reshape(-1, 3)

    # --------------------------------------------------------------- robot I/O

    def _robot(self, arm: str):
        arm = arm.lower()
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        return self._env.robot_manager.get_robot_by_arm_name(f"{arm}_arm")

    def get_ee_pose(self, arm: str) -> np.ndarray:
        """Fingertip (TCP) pose [x, y, z, qw, qx, qy, qz] in world frame."""
        robot = self._robot(arm)
        pose = self._link6_pose(arm)
        pose = pose.copy()
        pose[:3] += self._tcp_offset(robot, pose[3:7])
        return pose

    def _link6_pose(self, arm: str) -> np.ndarray:
        rm = self._env.robot_manager
        return self._np(rm.get_real_endpose(self._robot(arm), [0])[0])

    def get_finger_positions(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """World positions of the two finger bodies (link7, link8).

        This is the observability primitive that lets an agent CALIBRATE the
        gripper's (non-standard) tool frame itself: the two finger positions
        give the jaw opening axis and their midpoint vs link6 gives the tool
        axis — no documented convention needed.
        """
        rm = self._env.robot_manager
        robot = self._robot(arm)
        key = rm.robot_key[rm.robot_list.index(robot)]
        names = list(key.body_names)
        bp = key.data.body_pos_w[0].detach().cpu().numpy()
        origin = None
        try:
            origin = self._np(self._env.scene_manager.env_origins[0])[:3]
        except Exception:
            origin = np.zeros(3)
        out = []
        finger_links = (getattr(robot, "robot_args", {}) or {}).get(
            "finger_links", ("link7", "link8")
        )
        for n in finger_links:
            out.append(np.asarray(bp[names.index(n)], dtype=float).reshape(3) - origin)
        return out[0], out[1]

    def get_joints(self, arm: str) -> np.ndarray:
        rm = self._env.robot_manager
        return self._np(rm.get_joint(self._robot(arm), [0])[0])

    def get_step_budget(self) -> dict:
        """Simulation-step budget: {"used", "max"}. Motions and waits consume
        steps; the episode is truncated (and scored as-is) when it runs out."""
        return {"used": int(self._sim_step_count), "max": int(self.max_steps)}

    #: hard stop for a single motion, so a stuck queue cannot hang the trial
    MOTION_STEP_CAP = 4000

    def _run_control(self, seq: list[dict]) -> None:
        """Push a control sequence and step the sim until it drains.

        The task's `step_lim` is an *episode* budget, not a per-motion one:
        truncating individual motions with it silently leaves the arm wherever
        it happened to be. So exhausting the budget raises here instead, and the
        episode-level signal is reported through `truncated`.
        """
        if not seq:
            return
        if self.budget_exhausted():
            raise RuntimeError(
                f"episode step budget exhausted ({self._sim_step_count}/{self.max_steps} "
                "simulation steps used); no further motion is possible in this episode"
            )
        cm = self._env.robot_manager.control_manager
        cm.push([0], [seq])
        guard = 0
        while not self._env.have_empty([0]):
            self._env.step(env_idx_list=[0])
            self._sim_step_count += 1
            guard += 1
            self._maybe_capture()
            if guard >= self.MOTION_STEP_CAP:
                break

    def budget_exhausted(self) -> bool:
        return self._sim_step_count >= self.max_steps

    #: a move is only reported as successful within this position error (metres)
    POSE_TOLERANCE = 0.02
    #: sim steps for one goto_pose motion (RoboDojo's own eval uses 10-step
    #: action frames; 150 gives smooth-but-cheap motions inside a ~13k budget)
    MOTION_STEPS = 150

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @classmethod
    def _tilted(cls, quat: np.ndarray, axis: int, deg: float) -> np.ndarray:
        h = np.deg2rad(deg) / 2.0
        ax = np.zeros(3)
        ax[axis] = 1.0
        r = np.concatenate([[np.cos(h)], np.sin(h) * ax])
        return cls._quat_mul(r, np.asarray(quat, dtype=float))

    def _ik_verified(self, robot, link6_pos, quat):
        """solve_ik + FK check of the returned solution.

        RoboDojo's IK reports "Success" with solutions that do not satisfy the
        request when the pose is at/behind the workspace edge, so trusting the
        status wastes a full motion. Verify via cuRobo FK in the planner frame;
        if FK is unavailable, fall through (the post-motion check still guards).
        Returns the joint solution or None.
        """
        import copy

        rm = self._env.robot_manager
        target = np.concatenate([link6_pos, quat]).tolist()
        ik = rm.solve_ik(target_pose=target, env_idx=0, robot=robot, trans="world")
        if ik.get("status") != "Success":
            return None
        q = np.asarray(ik["joint_value"], dtype=float).reshape(-1)
        if not self._principal_ok(robot, q):
            return None
        try:
            import env.planner_manager.curobo_planner as _cp

            planner = rm.planner[robot.robot_name]
            tp = _cp.calculate_target_pose(
                copy.deepcopy(robot.entity_origin_pose),
                planner.robot_origin_pose,
                list(target),
            )
            p, _qq = planner._trans_from_world_to_base(
                np.array(planner.robot_origin_pose, dtype=np.float32),
                np.array(tp, dtype=np.float32),
            )
            p = np.asarray(p, dtype=float).reshape(3) + np.asarray(
                planner.frame_bias, dtype=float
            ).reshape(3)
            state = planner.ik_solver.compute_kinematics(planner._build_joint_state(q))
            tp = state.tool_poses  # ToolPose: position [batch, horizon, links, 3]
            pos_all = np.asarray(tp.position.detach().cpu()).reshape(-1, 3)
            frames = list(getattr(tp, "tool_frames", []) or [])
            eename = getattr(robot, "ee_link_name", "link6")
            idx = frames.index(eename) if eename in frames else -1
            fk_p = pos_all[idx]
            if np.linalg.norm(fk_p - p) > self.POSE_TOLERANCE:
                return None
        except Exception as e:
            if not getattr(self, "_fk_warned", False):
                self._fk_warned = True
                print(
                    f"[robodojo] FK verification unavailable ({type(e).__name__}: {e}); "
                    "relying on post-motion checks only",
                    flush=True,
                )
        return q

    #: cartesian servo parameters: waypoint spacing and sim steps per waypoint
    WAYPOINT_M = 0.02
    WAYPOINT_DEG = 10.0
    STEPS_PER_WAYPOINT = 10
    #: post-motion position assertion (looser than the servo's own 2 cm goal:
    #: a motion that lands 2-3 cm off is still usable for grasp/place, and the
    #: caller gets the honest residual in the error message when it's worse)
    ASSERT_TOLERANCE = 0.03

    def _stock_ik(self, robot, link6_pos, quat):
        """RoboDojo's own solve_ik (cost-best of ~32 seeds), or None on Fail.

        This is the ONLY solver observed to produce smooth, trackable joint
        sequences along a servo path; multi-seed selection (below) exists
        purely as a rescue when this one flips configuration branch.
        """
        rm = self._env.robot_manager
        ik = rm.solve_ik(
            target_pose=np.concatenate([link6_pos, quat]).tolist(),
            env_idx=0,
            robot=robot,
            trans="world",
        )
        if ik.get("status") != "Success":
            return None
        q = np.asarray(ik["joint_value"], dtype=float).reshape(-1)
        return q if self._principal_ok(robot, q) else None

    #: goto_pose treats an IK/FK orientation this far from the goal as "honoured";
    #: measured 2026-08-27 (store_tools_in_toolbox): the position-only rescue
    #: parked the wrist 35 deg off at the toolbox and the held hammer missed its slot.
    ORIENTATION_TOLERANCE_DEG = 8.0
    #: seeds asked from cuRobo's seeded solve for the branch-aware rescue
    RESCUE_SEEDS = 16
    #: max wrapped joint delta (rad) the servo follows without calling it a branch jump
    BRANCH_JUMP_RAD = 1.2

    @staticmethod
    def _quat_dev_deg(q1, q2) -> float:
        a = np.asarray(q1, dtype=float).reshape(4)
        b = np.asarray(q2, dtype=float).reshape(4)
        d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        return float(np.degrees(2.0 * np.arccos(min(1.0, d))))

    def _ik_rescue_ranked(self, robot, link6_pos, quat, q_seed) -> list[dict]:
        """Branch-aware IK candidates out of cuRobo's seeded solve, FK-checked.

        Returns ``[{"q", "branch", "pos_err", "ori_dev"}]`` sorted by branch
        distance from ``q_seed`` (max wrapped joint delta).  Every entry has an
        FK position within 1.5 cm of the request; ``ori_dev`` is the FK
        orientation's deviation from the requested quaternion in degrees, so the
        caller can prefer solutions that honour the orientation.  Empty when the
        seeded solver is unavailable.

        Why: stock ``solve_ik`` ranks ~32 seeds by cost, blind to the current
        joints — at the workspace edge its "best" solution sits on another branch
        (wrapped delta > 1.2 rad), the servo refuses the jump, and a position-only
        rescue used to leave the wrist 20-60 deg off while the position was met.
        """
        import copy as _copy

        out: list[dict] = []
        try:
            import env.planner_manager.curobo_planner as _cp

            rm = self._env.robot_manager
            planner = rm.ik_solver[robot.robot_name]
            target = np.concatenate([link6_pos, quat]).tolist()
            tp = _cp.calculate_target_pose(
                _copy.deepcopy(robot.entity_origin_pose),
                planner.robot_origin_pose,
                list(target),
            )
            p, q = planner._trans_from_world_to_base(
                np.array(planner.robot_origin_pose, dtype=np.float32),
                np.array(tp, dtype=np.float32),
            )
            p = np.asarray(p, dtype=np.float32).reshape(3) + np.asarray(
                planner.frame_bias, dtype=np.float32
            ).reshape(3)
            q_goal = np.asarray(q, dtype=float).reshape(4)
            goal = planner._build_goal_pose(p, q)
            state = planner._build_joint_state(np.asarray(q_seed, dtype=float))
            res = planner.ik_solver.solve_pose(
                goal_tool_poses=goal, current_state=state, return_seeds=self.RESCUE_SEEDS
            )
            sol = np.asarray(res.js_solution.position.detach().cpu())
            sol = sol.reshape(-1, sol.shape[-1])
            js_names = list(getattr(res.js_solution, "joint_names", None) or [])
            names = list(planner.active_joints_name)
            if len(js_names) == sol.shape[1]:
                idx = [js_names.index(n) for n in names]
                sol = sol[:, idx]
            elif sol.shape[1] == len(planner.cspace_joint_names):
                sol = sol[:, planner._active_to_cspace_idx]
            seed_active = np.asarray(q_seed, dtype=float).reshape(-1)[: sol.shape[1]]
            # FK-check each because the returned seeds include unconverged ones
            # (success-flag layout is version-dependent)
            for k in range(sol.shape[0]):
                state = planner.ik_solver.compute_kinematics(planner._build_joint_state(sol[k]))
                tps = state.tool_poses
                pos_all = np.asarray(tps.position.detach().cpu()).reshape(-1, 3)
                quat_all = np.asarray(tps.quaternion.detach().cpu()).reshape(-1, 4)
                frames = list(getattr(tps, "tool_frames", []) or [])
                ee = (
                    getattr(robot, "ee_link_name", None)
                    or getattr(planner, "ee_link", None)
                    or "link6"
                )
                fi = frames.index(ee) if ee in frames else -1
                pos_err = float(np.linalg.norm(pos_all[fi] - p))
                if pos_err >= 0.015 or not self._principal_ok(robot, sol[k]):
                    continue
                out.append(
                    {
                        "q": np.asarray(sol[k], dtype=float).reshape(-1),
                        "branch": self._branch_dist(robot, sol[k], seed_active),
                        "pos_err": pos_err,
                        "ori_dev": self._quat_dev_deg(quat_all[fi], q_goal),
                    }
                )
        except Exception as e:
            if not getattr(self, "_seeded_ik_warned", False):
                self._seeded_ik_warned = True
                print(
                    f"[robodojo] multi-seed IK rescue unavailable ({type(e).__name__}: {e})",
                    flush=True,
                )
        out.sort(key=lambda c: c["branch"])
        return out

    def _ik_rescue(self, robot, link6_pos, quat, q_seed, orientation_tol_deg=None):
        """Nearest-branch FK-verified solution out of cuRobo's seeds (or None).

        With ``orientation_tol_deg`` set, a solution that also honours the
        orientation within that tolerance is preferred; only when none exists does
        the position-only nearest branch come back (the pre-2026-08-27 behaviour).
        Used only when stock solve_ik returns a configuration-flipped solution
        (its cost ranking is blind to the current joints; near wrist +-pi the
        "best" solution alternates branches between consecutive 2 cm waypoints,
        which froze whole transports). Never used as the primary solver: ranking
        by branch distance across *every* waypoint made the servo hop between
        two valid-but-distant solution families faster than the controller
        could track, and the arm swept the scene.
        """
        ranked = self._ik_rescue_ranked(robot, link6_pos, quat, q_seed)
        if not ranked:
            return None
        if orientation_tol_deg is not None:
            for c in ranked:
                if c["branch"] > self.BRANCH_JUMP_RAD:
                    break  # only same-branch solutions may honour the orientation
                if c["ori_dev"] <= float(orientation_tol_deg):
                    return c["q"]
        return ranked[0]["q"]

    #: non-roll arm joints must stay within +-(pi + this) rad: the articulation's own
    #: limits are +-10 rad, so an elbow can be "reachable" three turns round.  Measured
    #: 2026-08-27: the jammed elbow read 5.86 rad; the store_tools_in_toolbox hammer carry
    #: to the table middle has NO solution under a 0.35 margin (17 refused, the carry that
    #: passed in round 1 needs a joint between 3.5 and ~4.3 rad), so the margin is 1.2
    PRINCIPAL_MARGIN_RAD = 1.2

    def _principal_ok(self, robot, q) -> bool:
        """True when every non-roll joint of ``q`` lies inside the principal range
        ``[-pi - margin, pi + margin]``.

        Measured 2026-08-27 (store_tools_in_toolbox): a failed approach wound the left
        X5's elbow to 5.86 rad (the USD limits are +-10 rad, cuRobo's too) and the
        wrapped-shortest-path servo then drove it into the table and its own base — the
        arm stayed jammed below the table top through go_home and a soft reset.  A
        solution with an in-limit alias a full turn closer to zero is a winding and is
        refused before it is executed; a joint whose own range cannot hold that alias
        (YAM joint 2 [0, 3.665], Franka joint 6 [-0.017, 3.75]) is left alone.  Refusals
        are counted (``_principal_refused``) and named in goto_pose's error.
        """
        from roboenv_runtime.adapters.robodojo.primitives import principal_range_ok

        q = np.asarray(q, dtype=float).reshape(-1)
        args = getattr(robot, "robot_args", {}) or {}
        roll = int(args.get("roll_joint_index", len(q) - 1))
        limits = None
        try:
            limits = self._joint_limits(robot)
        except Exception:
            limits = None
        ok = principal_range_ok(q, roll, self.PRINCIPAL_MARGIN_RAD, limits)
        if not ok:
            self._principal_refused = int(getattr(self, "_principal_refused", 0)) + 1
            if os.environ.get("CAPX_RD_DEBUG"):
                print(f"[dbg-guard] refused wound solution {np.round(q, 3).tolist()}", flush=True)
        return ok

    #: a goal whose oriented solution sits within this configuration distance of the
    #: current joints counts as reachable (the servo still guards every waypoint)
    GOAL_BRANCH_RAD = 2.0

    def _goal_solution(self, robot, link6_pos, quat, q_now=None):
        """Joint solution that honours ``quat`` at ``link6_pos``, or None.

        Stock solve_ik first (FK-verified); when it fails, the seeded solver from the
        current joints — measured 2026-08-27: at the toolbox the stock solver reported
        Fail for wrist tilts (x+16, x+20 at z 0.85) that the seeded solver reaches on
        the current branch with < 1 deg orientation error.
        """
        q = self._ik_verified(robot, link6_pos, quat)
        if q is not None:
            return q
        if q_now is None:
            rm = self._env.robot_manager
            q_now = np.asarray(rm.get_joint(robot, [0])[0], dtype=float).reshape(-1)
        for c in self._ik_rescue_ranked(robot, link6_pos, quat, q_now):
            if c["branch"] > self.GOAL_BRANCH_RAD:
                break
            if c["ori_dev"] <= self.ORIENTATION_TOLERANCE_DEG:
                return c["q"]
        return None

    @staticmethod
    def _wrapped_delta(q_to: np.ndarray, q_from: np.ndarray) -> np.ndarray:
        """Per-joint angular difference wrapped to (-pi, pi].

        A revolute joint at -3.13 vs +3.13 rad is the same orientation (2*pi
        apart); raw differences flag such solutions as huge "branch jumps" and
        froze the arm right after low grasps (wrist near +-pi). Compare
        orientations, not raw encoder values.
        """
        d = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
        return np.arctan2(np.sin(d), np.cos(d))

    def _branch_dist(self, robot, q_to: np.ndarray, q_from: np.ndarray) -> float:
        """Configuration distance (rad): the largest wrapped joint delta, ignoring the
        wrist-roll joint (``robot_args["roll_joint_index"]``, default the last joint).

        A roll about the tool axis never sweeps a link through space, but it is
        routinely large on the X5 (the servo's orientation slerp from the home pose to
        tool-down asks for 1.6 rad of roll inside one 2 cm waypoint — measured
        2026-08-27); counting it as a "branch jump" froze whole transports and left
        the orientation untracked.
        """
        d = np.abs(self._wrapped_delta(q_to, q_from))
        args = getattr(robot, "robot_args", {}) or {}
        roll = int(args.get("roll_joint_index", len(d) - 1))
        if 0 <= roll < len(d) and len(d) > 1:
            d = np.delete(d, roll)
        return float(np.max(d))

    def _joint_limits(self, robot) -> np.ndarray | None:
        """Per-joint position limits ``(dof, 2)`` of the arm, read once from the
        articulation (falls back to cuRobo's kinematics); None when unavailable.
        """
        cache = getattr(self, "_joint_limit_cache", None)
        if cache is None:
            cache = self._joint_limit_cache = {}
        name = getattr(robot, "robot_name", id(robot))
        if name in cache:
            return cache[name]
        lim = None
        try:
            rm = self._env.robot_manager
            key = rm.robot_key[rm.robot_list.index(robot)]
            data = key.data
            raw = getattr(data, "joint_pos_limits", None)
            if raw is None:
                raw = getattr(data, "joint_limits", None)
            if raw is not None:
                lim = np.asarray(raw[0].detach().cpu(), dtype=float).reshape(-1, 2)
                # the articulation lists EVERY joint in Isaac's order; the servo's q
                # vectors are joint_pos[arm_joint_indices] — index the limits the same way
                idx = getattr(robot, "arm_joint_indices", None)
                if idx is not None:
                    idx = np.asarray(idx, dtype=int).reshape(-1)
                    if idx.size and int(idx.max()) < lim.shape[0]:
                        lim = lim[idx]
        except Exception:
            lim = None
        if lim is None:
            try:
                planner = self._env.robot_manager.ik_solver[robot.robot_name]
                jl = planner.ik_solver.kinematics.get_joint_limits()
                pos = np.asarray(jl.position.detach().cpu(), dtype=float).reshape(2, -1)
                lim = pos.T  # cspace order → active order
                order = getattr(planner, "_active_to_cspace_idx", None)
                if order is not None:
                    lim = lim[np.asarray(order, dtype=int).reshape(-1)]
            except Exception:
                lim = None
        cache[name] = lim
        return lim

    def _exec_delta(self, robot, q_to: np.ndarray, q_from: np.ndarray) -> np.ndarray:
        """Joint deltas to EXECUTE from ``q_from`` toward ``q_to``: the shortest
        (wrapped) way per joint, unless that lands outside the joint's limits —
        then the long way round (2*pi the other direction) when that stays inside.

        Measured 2026-08-27 (store_tools_in_toolbox): the X5's wrist roll (joint6)
        is limited to +-3.14 while every other joint wraps; ramping the wrapped
        delta from -3.14 toward an IK solution at +2.54 (delta -0.60) drove the
        joint into its limit, the sim clamped it, and the tool stayed 35 deg off
        while the position was met.  The long way (+5.68 rad) reaches it.
        """
        q_from = np.asarray(q_from, dtype=float).reshape(-1)
        delta = self._wrapped_delta(np.asarray(q_to, dtype=float).reshape(-1), q_from)
        lim = self._joint_limits(robot)
        if lim is None:
            return delta
        if lim.shape[0] != len(delta) and not getattr(self, "_limits_dof_warned", False):
            self._limits_dof_warned = True
            print(
                f"[robodojo] joint limits cover {lim.shape[0]} joints but the servo drives "
                f"{len(delta)}: the extra joints are unguarded",
                flush=True,
            )
        n = min(len(delta), lim.shape[0])
        margin = 0.02
        for j in range(n):
            lo, hi = float(lim[j, 0]), float(lim[j, 1])
            # (no "effectively unlimited" shortcut: a +-10 rad joint wound past +-(10-pi)
            # still clamps on the wrapped way — the range check below costs nothing)
            target = q_from[j] + delta[j]
            if lo + margin <= target <= hi - margin:
                continue
            alt = delta[j] - np.sign(delta[j]) * 2 * np.pi
            alt_target = q_from[j] + alt
            if lo + margin <= alt_target <= hi - margin:
                delta[j] = alt
            elif lo <= alt_target <= hi and not (lo <= target <= hi):
                delta[j] = alt  # only the long way stays inside the hard limits at all
        return delta

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        d = float(np.dot(q0, q1))
        if d < 0:
            q1, d = -q1, -d
        if d > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q)
        th = np.arccos(np.clip(d, -1, 1))
        return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)

    def goto_pose(
        self,
        arm: str,
        position,
        quaternion_wxyz,
        max_tilt_deg: float = 0.0,
        max_yaw_deg: float = 0.0,
    ) -> None:
        """Move one arm so the *fingertips* reach a world-frame pose.

        Executes as a Cartesian servo: the straight-line path from the current
        fingertip pose to the target is walked in ~2 cm waypoints, solving IK
        at each one (seeded by the current joints, so solutions stay on the
        same branch) and tracking it with a short joint ramp. This mirrors how
        RoboDojo's own eval drives the arm (per-frame ee targets) and avoids
        the two failure modes of one-shot far IK on this platform: false
        "Success" solutions at the workspace edge, and configuration jumps
        that sweep links through the table.

        With max_tilt_deg > 0, orientations tilted up to that angle from the
        requested one are tried when the exact orientation has no reachable IK
        at the target (the benchmark's own config asks for
        "top_down_little_left/right" grasps — pure vertical is often
        infeasible for this 6-DoF arm).
        """
        rm = self._env.robot_manager
        robot = self._robot(arm)
        quat0 = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
        tip_target = np.asarray(position, dtype=float).reshape(3)
        print(
            f"[robodojo] goto_pose V16 {arm} tip={np.round(tip_target, 3).tolist()} "
            f"tilt={max_tilt_deg} dbg={bool(os.environ.get('CAPX_RD_DEBUG'))}",
            flush=True,
        )

        # -- choose a reachable goal orientation (exact first, then tilts)
        candidates = [quat0]
        if max_tilt_deg > 0:
            degs = [d for d in (10.0, 20.0, max_tilt_deg) if 0 < d <= max_tilt_deg]
            for d in sorted(set(degs)):
                for axis in (0, 1):
                    candidates.append(self._tilted(quat0, axis, +d))
                    candidates.append(self._tilted(quat0, axis, -d))
        # yaw sweep (opt-in): when the caller declares the jaw yaw free
        # (position-critical moves, symmetric objects), rotating about world Z
        # turns many workspace-edge failures into clean reaches — X5 reach is
        # strongly orientation-dependent
        if max_yaw_deg > 0:
            yaws = [y for y in (15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0) if y <= max_yaw_deg]
            base_set = list(candidates)
            for y in yaws:
                for sgn in (+1.0, -1.0):
                    for b in base_set:
                        candidates.append(self._tilted(b, 2, sgn * y))
                    if y == 180.0:
                        break
        # jaw-axis-preserving wrist flip: rotating 180 deg about world Z keeps
        # the closing line identical but lands on the other IK branch — near the
        # table or the workspace edge it is often the only reachable one.
        candidates = [c for base in candidates for c in (base, self._tilted(base, 2, 180.0))]
        dbg = bool(os.environ.get("CAPX_RD_DEBUG"))
        goal_quat = None
        refused_before = int(getattr(self, "_principal_refused", 0))
        q_now = np.asarray(rm.get_joint(robot, [0])[0], dtype=float).reshape(-1)
        for ci, quat in enumerate(candidates):
            link6 = tip_target - self._tcp_offset(robot, quat)
            if self._goal_solution(robot, link6, quat, q_now) is not None:
                goal_quat = quat
                if dbg:
                    print(
                        f"[dbg-gp] {arm} cand#{ci} quat={np.round(quat, 3).tolist()} accepted",
                        flush=True,
                    )
                break
            if dbg:
                print(f"[dbg-gp] {arm} cand#{ci} rejected", flush=True)
        if goal_quat is None:
            refused = int(getattr(self, "_principal_refused", 0)) - int(refused_before)
            raise RuntimeError(
                f"{arm} arm cannot reach fingertip position="
                f"{np.round(tip_target, 3).tolist()} with the requested orientation"
                f"{' (even tilted up to %g deg)' % max_tilt_deg if max_tilt_deg > 0 else ''}"
                f"{f' ({refused} IK solutions refused as wound joints)' if refused else ''}. "
                f"That arm's base is near x={'-0.3' if arm == 'left' else '+0.3'}, y=-0.45 — "
                "try a pose closer to it, pass max_tilt_deg=25, add max_yaw_deg=90 "
                "if the jaw yaw is free (symmetric object / pure repositioning), "
                "or use the other arm."
            )

        # -- walk the straight-line path in small waypoints
        start_pose = self.get_ee_pose(arm)
        p0, q0 = start_pose[:3], start_pose[3:7]
        dist = float(np.linalg.norm(tip_target - p0))
        # waypoints dense enough in position AND orientation: a 90 deg re-orientation
        # spread over 5 position waypoints asked for 1.3-1.7 rad of joint motion per
        # waypoint (measured 2026-08-27) and every one of them was skipped as a jump
        turn = self._quat_dev_deg(np.asarray(q0, float), goal_quat)
        n_wp = max(2, int(np.ceil(dist / self.WAYPOINT_M)), int(np.ceil(turn / self.WAYPOINT_DEG)))
        key = rm.process_name(robot.arm_name)
        q_prev = np.asarray(rm.get_joint(robot, [0])[0], dtype=float).reshape(-1)

        for i in range(1, n_wp + 1):
            t = i / n_wp
            wp_tip = p0 + (tip_target - p0) * t
            wp_quat = self._slerp(np.asarray(q0, float), goal_quat, t)
            wp_link6 = wp_tip - self._tcp_offset(robot, wp_quat)
            q_wp = self._stock_ik(robot, wp_link6, wp_quat)
            wdelta = self._branch_dist(robot, q_wp, q_prev) if q_wp is not None else None
            if q_wp is None or wdelta > self.BRANCH_JUMP_RAD:
                # stock failed or flipped configuration — try the seeded rescue,
                # preferring a branch that also honours the orientation
                q_alt = self._ik_rescue(
                    robot, wp_link6, wp_quat, q_prev, self.ORIENTATION_TOLERANCE_DEG
                )
                if q_alt is not None:
                    wd_alt = self._branch_dist(robot, q_alt, q_prev)
                    if wd_alt <= self.BRANCH_JUMP_RAD:
                        if dbg:
                            print(
                                f"[dbg-gp] wp{i}/{n_wp} rescue "
                                f"(stock={'fail' if wdelta is None else f'{wdelta:.2f}'} "
                                f"-> {wd_alt:.2f})",
                                flush=True,
                            )
                        q_wp, wdelta = q_alt, wd_alt
            if q_wp is None:
                if dbg:
                    print(f"[dbg-gp] wp{i}/{n_wp} IK FAIL", flush=True)
                continue  # skip an awkward interior waypoint; final pose is verified
            if wdelta > self.BRANCH_JUMP_RAD:
                if dbg:
                    print(f"[dbg-gp] wp{i}/{n_wp} BRANCH JUMP wrapped={wdelta:.2f}", flush=True)
                continue  # a genuine configuration flip; stay on this branch
            # Execute along the SHORTEST angular path. q_wp can be the same
            # orientation modulo 2*pi (base/wrist at +3.13 vs -3.13, wrapped
            # delta 0.06 but raw delta ~2*pi); ramping raw values there spins
            # the whole arm a full turn mid-motion — the tip sweeps half the
            # table (observed knocking objects off it). If the wrapped target
            # sits just past a +-pi joint limit the sim clamps to the limit,
            # which is the same orientation to within ~0.03 rad.
            delta = self._exec_delta(robot, q_wp, q_prev)
            amp = float(np.max(np.abs(delta)))
            steps = self.STEPS_PER_WAYPOINT if i < n_wp else self.STEPS_PER_WAYPOINT * 4
            if amp > 1.0:  # genuine large reconfiguration — give it time
                steps = max(steps, int(amp * 40))
            seq = []
            for j in range(steps):
                a = min(1.0, (j + 1) / max(1, int(steps * 0.9)))
                qq = q_prev + delta * a
                seq.append({key: {"position": qq.tolist(), "velocity": [0.0] * len(qq)}})
            self._run_control(seq)
            q_prev = q_prev + delta
            if dbg and (i % 4 == 0 or i == n_wp):
                ach = self._link6_pose(arm)[:3]
                print(
                    f"[dbg-gp] wp{i}/{n_wp} tracked_err={np.linalg.norm(ach - wp_link6) * 100:.1f}cm",
                    flush=True,
                )

        link6_target = tip_target - self._tcp_offset(robot, goal_quat)
        # micro-corrections: nudge out residual tracking error (a few cm) that
        # the fixed waypoint grid leaves behind near workspace edges, and an
        # orientation the waypoint rescues could not honour (measured 2026-08-27:
        # a position-only rescue parked the wrist 35 deg off at the workspace edge)
        for _ in range(3):
            pose_now = self._link6_pose(arm)
            err_vec = link6_target - pose_now[:3]
            ori_now = self._quat_dev_deg(pose_now[3:7], goal_quat)
            if (
                np.linalg.norm(err_vec) <= self.POSE_TOLERANCE
                and ori_now <= self.ORIENTATION_TOLERANCE_DEG
            ):
                break
            q_fix = self._stock_ik(robot, link6_target, goal_quat)
            wrapped = self._branch_dist(robot, q_fix, q_prev) if q_fix is not None else None
            if q_fix is None or wrapped > self.BRANCH_JUMP_RAD:
                ranked = self._ik_rescue_ranked(robot, link6_target, goal_quat, q_prev)
                oriented = [
                    c
                    for c in ranked
                    if c["ori_dev"] <= self.ORIENTATION_TOLERANCE_DEG
                    and c["branch"] <= self.BRANCH_JUMP_RAD
                ]
                pick = oriented[0] if oriented else (ranked[0] if ranked else None)
                if pick is not None and (wrapped is None or pick["branch"] < wrapped):
                    if not oriented and np.linalg.norm(err_vec) <= self.POSE_TOLERANCE:
                        break  # position met; no same-branch solution honours the orientation
                    q_fix, wrapped = pick["q"], pick["branch"]
            if q_fix is None:
                break
            if wrapped > 2.6:
                break  # would need a full configuration change; give up honestly
            delta = self._exec_delta(robot, q_fix, q_prev)  # shortest path inside limits
            steps = max(40, int(float(np.max(np.abs(delta))) * 60))
            seq = []
            for j in range(steps):
                a = min(1.0, (j + 1) / max(1, int(steps * 0.9)))
                qq = q_prev + delta * a
                seq.append({key: {"position": qq.tolist(), "velocity": [0.0] * len(qq)}})
            self._run_control(seq)
            q_prev = q_prev + delta
        self._assert_reached(arm, link6_target)
        return self._pose_report(arm, link6_target, goal_quat)

    def _pose_report(self, arm: str, link6_target: np.ndarray, goal_quat: np.ndarray) -> dict:
        """How well the motion tracked: position error (m) and orientation deviation (deg).

        Orientation is NOT asserted: at the workspace edge the arm can satisfy the
        position with the wrist 20-60 deg off (measured 2026-08-26), and callers that
        only care about position must not fail for it.  They can read it here.
        """
        pose = self._link6_pose(arm)
        err = float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(link6_target)))
        q1 = np.asarray(pose[3:7], dtype=float)
        q2 = np.asarray(goal_quat, dtype=float)
        d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
        dev = float(np.degrees(2 * np.arccos(min(1.0, d))))
        return {"position_error_m": round(err, 4), "orientation_dev_deg": round(dev, 1)}

    def _assert_reached(self, arm: str, link6_target: np.ndarray) -> None:
        reached = self._link6_pose(arm)[:3]
        err = float(np.linalg.norm(reached - link6_target))
        if err > self.ASSERT_TOLERANCE:
            raise RuntimeError(
                f"{arm} arm did not reach the requested pose: ended "
                f"{err * 100:.1f} cm from the target. The pose is most likely at "
                "the edge of this arm's workspace — try a nearby pose, approach "
                "higher, or use the other arm."
            )

    def go_home(self, arm: str, steps: int = 260) -> None:
        """Return one arm to its reset joint configuration (scoring requires
        all_robot_back_to_origin: pos<0.15 m, rot<20 deg)."""
        home = getattr(self, "_home_joints", {}).get(arm)
        if home is None:
            return
        rm = self._env.robot_manager
        robot = self._robot(arm)
        key = rm.process_name(robot.arm_name)
        q_now = np.asarray(rm.get_joint(robot, [0])[0], dtype=float).reshape(-1)
        delta = self._exec_delta(robot, home, q_now)
        seq = []
        for j in range(steps):
            a = min(1.0, (j + 1) / max(1, int(steps * 0.9)))
            qq = q_now + delta * a
            seq.append({key: {"position": qq.tolist(), "velocity": [0.0] * len(qq)}})
        self._run_control(seq)

    # ---------------- composed primitives (X5-native suite, V17) ----------------

    TOPDOWN_QUAT = (0.7071, 0.0, 0.7071, 0.0)  # X5: jaws close along world Y

    def _topdown_quats(self, robot) -> list:
        """Embodiment's tool-down orientation family (robot_config.yml
        top_down_quats), X5 defaults preserved."""
        args = getattr(robot, "robot_args", {}) or {}
        fam = args.get("top_down_quats")
        if fam:
            return [list(map(float, q)) for q in fam]
        return [list(self.TOPDOWN_QUAT), [0.5, 0.5, 0.5, -0.5]]

    def check_reachable(
        self, arm: str, position, quaternion_wxyz=None, max_tilt_deg: float = 25.0
    ) -> dict:
        """Plan-only reachability probe: runs the same IK candidate search
        goto_pose uses (exact quat, tilts, wrist flip) WITHOUT moving.
        Returns {"reachable": bool, "quat_wxyz": working orientation}."""
        robot = self._robot(arm)
        tip = np.asarray(position, dtype=float).reshape(3)
        if quaternion_wxyz is not None:
            bases = [np.asarray(quaternion_wxyz, dtype=float).reshape(4)]
        else:
            bases = [np.asarray(q, dtype=float).reshape(4) for q in self._topdown_quats(robot)]
        candidates = list(bases)
        if max_tilt_deg > 0:
            for d in (10.0, 20.0, max_tilt_deg):
                for axis in (0, 1):
                    candidates.append(self._tilted(bases[0], axis, +d))
                    candidates.append(self._tilted(bases[0], axis, -d))
        candidates = [c for base in candidates for c in (base, self._tilted(base, 2, 180.0))]
        for quat in candidates:
            link6 = tip - self._tcp_offset(robot, quat)
            if self._goal_solution(robot, link6, quat) is not None:
                return {"reachable": True, "quat_wxyz": [float(x) for x in quat]}
        return {"reachable": False}

    def _bbox_ext_top(self, label: str) -> tuple[np.ndarray, float]:
        """Side lengths (3,) and WORLD-frame top z of an object's bbox.
        get_object_bbox returns (N,3) VERTICES in center-relative coords
        (world-absolute handled too, discriminated by magnitude)."""
        pos, _ = self.get_object_pose(label)
        verts = np.asarray(self.get_object_bbox(label), dtype=float).reshape(-1, 3)
        ext = verts.max(axis=0) - verts.min(axis=0)
        zmax = float(verts[:, 2].max())
        top_z = zmax if abs(zmax) > 0.5 else float(pos[2]) + zmax
        return ext, top_z

    def pick(self, arm: str, label: str, lift: float = 0.10) -> dict:
        """Full pick of a rigid object: bbox-informed pinch ladder (center,
        then rim offsets along each horizontal axis), top-down orientation
        with tilt/branch fallback, descend-close-lift, and a HONEST verify:
        the object must actually rise with the gripper. Returns
        {"grasped": bool, "offset": [dx,dy], "quat_wxyz": ...}."""
        pos, _ = self.get_object_pose(label)
        ext, top_z = self._bbox_ext_top(label)
        cx, cy = float(pos[0]), float(pos[1])
        # THIN objects (tiles, coins, lids): a pinch at top_z+1.5cm closes
        # above them. Pinch at the object's mid-height instead, straddling
        # its smaller horizontal extent so the pads envelop it near the table
        robot = self._robot(arm)
        args = getattr(robot, "robot_args", {}) or {}
        hover = float(args.get("hover_clearance", 0.09))
        gtilt = float(args.get("grasp_max_tilt_deg", 25.0))
        thin = float(ext[2]) < 0.03
        # The reported fingertip is the BOTTOM of the pads (measured 2026-08-26: the finger
        # bodies sit 7 cm above it), so a pinch must put the tip BELOW the object's top for
        # the pads to overlap it; 1.5 cm above the top closes on air on every object.
        pinch_below = float(args.get("pinch_below_top_m", 0.012))
        if thin:
            top_z = float(pos[2])  # mid-height of a thin object ~ its center
        elif args.get("grasp_depth_mode") == "center":
            # tall pads (YAM): pinch at object-center height, not near the top
            top_z = float(pos[2]) - 0.015
        else:
            top_z = top_z - min(pinch_below, 0.4 * float(ext[2]))
        # pinch offsets: center first; wide objects (jaw span ~0.13) need a
        # one-rim pinch — offset ~ half-extent so one finger enters the rim
        offsets = [(0.0, 0.0)]
        for axis, e in ((0, float(ext[0])), (1, float(ext[1]))):
            if e > 0.09:
                off = e / 2.0 - 0.016
                for sgn in (-1.0, +1.0):
                    o = [0.0, 0.0]
                    o[axis] = sgn * off
                    offsets.append(tuple(o))
        last_err = None
        for dx, dy in offsets:
            target = [cx + dx, cy + dy, top_z + (0.002 if thin else 0.0)]
            probe = self.check_reachable(arm, [target[0], target[1], target[2] + hover])
            if not probe["reachable"]:
                continue
            quat = probe["quat_wxyz"]
            try:
                self.set_gripper(arm, 1.0)
                self.goto_pose(
                    arm, [target[0], target[1], target[2] + hover], quat, max_tilt_deg=gtilt
                )
                self.goto_pose(arm, target, quat, max_tilt_deg=gtilt)
                for stage in args.get("grasp_close_sequence") or [0.0]:
                    self.set_gripper(arm, float(stage))
                self.wait_steps(20)
                z_before = float(self.get_object_pose(label)[0][2])
                self.goto_pose(
                    arm, [target[0], target[1], target[2] + lift], quat, max_tilt_deg=gtilt
                )
                z_after = float(self.get_object_pose(label)[0][2])
                if z_after - z_before > 0.4 * lift:
                    return {"grasped": True, "offset": [dx, dy], "quat_wxyz": quat}
                self.set_gripper(arm, 1.0)  # honest miss: release and retry
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"pick: no pinch on {label} held (tried {len(offsets)} offsets "
            f"with {arm} arm). Object extent {np.round(ext, 3).tolist()}; "
            f"last error: {str(last_err)[:180] if last_err else 'grasp closed but object did not rise'}; "
            f"try the other arm or a manual approach."
        ) from last_err

    def place_at(self, arm: str, position, hover: float = 0.08, release_dz: float = 0.02) -> None:
        """Carry the held object above a target, descend, release, retreat.
        position is where the object's BOTTOM should land (xy) at about the
        given z; release happens release_dz above it so the object settles."""
        tgt = np.asarray(position, dtype=float).reshape(3)
        pose = self.get_ee_pose(arm)
        quat = [float(x) for x in pose[3:7]]
        self.goto_pose(
            arm, [float(tgt[0]), float(tgt[1]), float(tgt[2]) + hover], quat, max_tilt_deg=25.0
        )
        self.goto_pose(
            arm, [float(tgt[0]), float(tgt[1]), float(tgt[2]) + release_dz], quat, max_tilt_deg=25.0
        )
        self.set_gripper(arm, 1.0)
        self.wait_steps(15)
        self.goto_pose(
            arm, [float(tgt[0]), float(tgt[1]), float(tgt[2]) + hover], quat, max_tilt_deg=25.0
        )

    def press(self, arm: str, label: str, depth: float = 0.015, hover: float = 0.06) -> dict:
        """Press straight down on an object (button/key/switch) with the
        CLOSED gripper tip: hover above its top, descend to depth below the
        top surface, retreat. Returns {"pressed": bool, "top_z", "dz"}
        where dz is how far the object's center sank."""
        pos, _ = self.get_object_pose(label)
        _ext, top = self._bbox_ext_top(label)
        x, y = float(pos[0]), float(pos[1])
        self.set_gripper(arm, 0.0)
        r = self.check_reachable(arm, [x, y, top + hover])
        if not r.get("reachable"):
            return {"pressed": False, "error": "hover unreachable", "top_z": top}
        quat = r["quat_wxyz"]
        self.goto_pose(arm, [x, y, top + hover], quat, max_tilt_deg=10.0)
        descended = False
        try:
            self.goto_pose(arm, [x, y, top - float(depth)], quat, max_tilt_deg=10.0)
            descended = True
        except Exception:
            pass  # a refused short descent usually means the surface stopped us
        self.wait_steps(10)
        p2, _ = self.get_object_pose(label)
        dz = float(pos[2]) - float(p2[2])
        try:
            self.goto_pose(arm, [x, y, top + hover], quat, max_tilt_deg=10.0)
        except Exception:
            pass
        return {
            "pressed": bool(descended or dz > 0.002),
            "top_z": round(top, 4),
            "dz": round(dz, 4),
        }

    def pour(self, arm: str, angle_deg: float = 110.0, axis: int = 0, hold_steps: int = 60) -> None:
        """Rotate the held container in place to pour, hold, rotate back.
        axis 0 = world X, 1 = world Y. Do this ABOVE the receiver mouth."""
        pose = self.get_ee_pose(arm)
        pos = [float(x) for x in pose[:3]]
        quat0 = np.asarray(pose[3:7], dtype=float)
        tilted = self._tilted(quat0, axis, float(angle_deg))
        self.goto_pose(arm, pos, [float(x) for x in tilted], max_tilt_deg=0.0)
        self.wait_steps(hold_steps)
        for wiggle in (+12.0, -12.0):
            w = self._tilted(quat0, axis, float(angle_deg) + wiggle)
            try:
                self.goto_pose(arm, pos, [float(x) for x in w], max_tilt_deg=0.0)
            except Exception:
                pass
        self.goto_pose(arm, pos, [float(x) for x in quat0], max_tilt_deg=0.0)

    # -------------- VLA-as-tool (hybrid control: the code agent keeps goal
    # selection + verification; a pretrained policy supplies closed-loop
    # dexterity for contact-rich stretches) ------------------------------
    VLA_PORT_BASE = 8790

    def _ensure_vla(self, policy: str = "G05"):
        if getattr(self, "_vla_client", None) is not None:
            return self._vla_client
        import socket
        import subprocess
        import time as _t

        W = str(repo_root())
        rd_root = str(robodojo_root().resolve())
        gpu = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0").strip()
        try:
            port = self.VLA_PORT_BASE + int(gpu)
        except ValueError:
            port = self.VLA_PORT_BASE
        xpl = os.path.join(rd_root, "XPolicyLab")
        pdir = os.path.join(xpl, "policy", policy)
        env = dict(os.environ)
        from roboenv_runtime.adapters.robodojo.vla_manager import _policy_runtime_env

        policy_cache = Path(W) / ".cache" / "vla" / "robodojo" / f"{policy}_g{gpu}"
        env.update(_policy_runtime_env(policy_cache))
        env.update(
            {
                "HOME": os.path.join(W, "omni-home"),
                "HF_HOME": os.path.join(W, ".cache", "huggingface"),
                "G05_CKPT_PATH": os.path.join(
                    xpl,
                    "policy/GalaxeaVLA/checkpoints/"
                    "RoboDojo-sim-arx_x5-joint-0/checkpoints/checkpoint",
                ),
                "ROBODOJO_G05_ACTION_SOURCE": "fm",
                "G05_ACTION_TOKENIZER_PATH": os.path.join(
                    xpl,
                    "policy/GalaxeaVLA/checkpoints/RoboDojo-sim-arx_x5-joint-0/action_tokenizer.pt",
                ),
            }
        )
        vpy = os.path.join(pdir, policy, ".venv", "bin", "python")
        cmd = [
            "bash",
            os.path.join(pdir, "setup_eval_policy_server.sh"),
            "RoboDojo",
            self.task_name,
            "RoboDojo-sim-arx_x5-joint-0",
            self.env_cfg_type,
            "joint",
            str(self.seed),
            # the server script re-exports CUDA_VISIBLE_DEVICES with this
            # value, and CUDA ids are absolute - pass the PHYSICAL id
            gpu,
            vpy,
            str(port),
            "localhost",
        ]
        log_dir = runs_root() / "vla_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logf = open(log_dir / f"vla_server_{policy}_g{gpu}.log", "ab")
        self._vla_server = subprocess.Popen(
            cmd, env=env, cwd=pdir, stdout=logf, stderr=logf, start_new_session=True
        )
        deadline = _t.time() + 900
        up = False
        while _t.time() < deadline:
            if self._vla_server.poll() is not None:
                raise RuntimeError(
                    f"vla policy server exited rc={self._vla_server.returncode}"
                    f" (see solutions/logs/vla_server_{policy}_g{gpu}.log)"
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    up = True
                    break
            except OSError:
                _t.sleep(5)
        if not up:
            raise RuntimeError("vla policy server did not open its port in 900s")
        import inspect as _inspect

        import websockets as _ws

        _connect = _ws.connect
        try:
            if "proxy" not in _inspect.signature(_connect).parameters:

                def _connect_no_proxy(*a, proxy=None, **k):
                    return _connect(*a, **k)

                _ws.connect = _connect_no_proxy
        except (TypeError, ValueError):
            pass
        from client_server.ws.model_client import WsModelClient

        self._vla_client = WsModelClient(
            url=f"ws://127.0.0.1:{port}",
            evaluation_id=f"roboenv-{self.task_name}",
            trial_id=f"{self.task_name}-roboenv",
            action_case_id=f"{self.task_name}_case",
        )
        return self._vla_client

    def _observable_object_positions(self) -> dict[str, np.ndarray]:
        """Snapshot only scene labels that currently have a materialized pose.

        RoboDojo can enumerate deferred labels before their prim exists. Motion
        reporting is diagnostic and must not block policy actions for such labels.
        """
        positions: dict[str, np.ndarray] = {}
        for label in self.list_objects():
            try:
                position, _ = self.get_object_pose(label)
            except Exception:
                continue
            positions[label] = np.asarray(position, dtype=float).copy()
        return positions

    @staticmethod
    def _current_neutral_hold_action(env: Any) -> Any:
        """Read env0's current canonical action without consulting private queues."""

        observation = env.get_obs()
        if isinstance(observation, Mapping):
            if "action" not in observation:
                raise RuntimeError("neutral-hold observation lacks canonical action")
            return observation["action"]
        if (
            isinstance(observation, (list, tuple))
            and len(observation) == 1
            and isinstance(observation[0], Mapping)
            and "action" in observation[0]
        ):
            return observation[0]["action"]
        raise RuntimeError("neutral-hold observation must be an env0 mapping or one-item env batch")

    def neutral_hold(
        self,
        *,
        n_actions: int,
        allow_midstate: bool = False,
        reserve_sim_steps: int = 0,
        stop_when_latched: bool = True,
    ) -> dict[str, Any]:
        """Advance interactive task callbacks while holding both X5 arms still.

        This is the in-process counterpart of :meth:`VlaManager.run_hold`. It
        executes only the finite canonical action already present in RoboDojo's
        public observation; callers cannot supply native actions. The official
        ``EvalEnv.take_action`` path is what lets scripted support robots advance.

        Mid-episode handoff is refused by default, every frame is charged at the
        observation interval, and an installed :class:`ScoreClock` owns the native
        tick so reward state is never advanced twice. The result deliberately
        contains public motion summaries and an action hash, not support queues,
        selected labels, or other task-private state.
        """

        from roboenv_runtime.adapters.robodojo.vla_manager import (
            MAX_VLA_ACTIONS,
            _canonical_actions_sha256,
            _jsonable,
            _validated_joint_action,
        )

        if self.env_cfg_type != "arx_x5":
            raise RuntimeError(
                "neutral_hold is supported only for the ARX X5 canonical joint schema"
            )
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
        if (
            isinstance(n_actions, bool)
            or not isinstance(n_actions, int)
            or not 1 <= n_actions <= MAX_VLA_ACTIONS
        ):
            raise ValueError(f"neutral_hold n_actions must be an integer in 1..{MAX_VLA_ACTIONS}")

        env = self._env
        action_frames = int(np.asarray(env.take_action_cnt).reshape(-1)[0])
        sim_steps = int(self._sim_step_count)
        if (action_frames or sim_steps) and not allow_midstate:
            raise RuntimeError(
                "neutral-hold mid-state handoff refused after "
                f"{action_frames} action frames / {sim_steps} simulation steps; "
                "set allow_midstate=true only after deliberate verification"
            )

        interval = int(getattr(getattr(env, "obs_manager", None), "collect_interval", 10) or 10)
        if interval <= 0:
            raise RuntimeError("neutral-hold collect_interval must be positive")
        clock = getattr(env, "_codex_dojo_score_clock", None)
        if bool(getattr(clock, "submitted", False)):
            raise RuntimeError("neutral_hold is frozen after score submission")

        def success_latched() -> bool:
            if clock is not None:
                return bool(getattr(clock, "latched", False))
            try:
                rewards = env.reward_manager.get_reward(final_check=False)
                return float(np.asarray(rewards).reshape(-1)[0]) >= 1.0 - 1e-3
            except Exception:
                return False

        before_positions = self._observable_object_positions()
        executed: list[Any] = []
        stop_reason = "action_limit"
        for _ in range(n_actions):
            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and success_latched():
                stop_reason = "success_latched"
                break
            current_frame = int(np.asarray(env.take_action_cnt).reshape(-1)[0])
            step_lim = int(getattr(env, "step_lim", 0) or 0)
            if step_lim > 0 and current_frame >= step_lim:
                stop_reason = "task_action_limit"
                break
            remaining = int(self.max_steps) - int(self._sim_step_count)
            if remaining < reserve_sim_steps + interval:
                stop_reason = "budget_reserve"
                break

            action = _validated_joint_action(self._current_neutral_hold_action(env))
            native = getattr(clock, "native_action", None)
            context = native() if callable(native) else nullcontext()
            with context:
                env.take_action(action)
            self._sim_step_count += interval
            self._maybe_capture()
            executed.append(_jsonable(action))

            if bool(np.asarray(env.end_flag).reshape(-1)[0]):
                stop_reason = "environment_end"
                break
            if stop_when_latched and success_latched():
                stop_reason = "success_latched"
                break

        after_positions = self._observable_object_positions()
        moved: dict[str, float] = {}
        for label in sorted(before_positions.keys() & after_positions.keys()):
            distance = float(
                np.linalg.norm(
                    np.asarray(after_positions[label]) - np.asarray(before_positions[label])
                )
            )
            if distance > 0.001:
                moved[label] = round(distance, 6)
        action_sha = _canonical_actions_sha256(executed)
        return {
            "mode": "hold",
            "policy": None,
            "actions_taken": len(executed),
            "requested_actions": n_actions,
            "stop_reason": stop_reason,
            "objects_moved_m": moved,
            "budget": self.get_step_budget(),
            "provenance": {
                "source": "env0_current_action",
                "observed_actions_sha256": action_sha,
                "observed_actions_count": len(executed),
            },
            "rng_replay": None,
        }

    def vla_run(self, n_actions: int = 25, policy: str = "G05") -> dict:
        """Hand closed-loop control to a pretrained VLA policy for up to
        n_actions action frames, then return a before/after report. The
        policy sees the cameras + robot state exactly as in its own
        benchmark evaluation; sim-step budget is charged honestly."""
        if self.budget_exhausted():
            raise RuntimeError("episode step budget exhausted; vla_run refused")
        env = self._env
        client = self._ensure_vla(policy)
        if getattr(self, "_vla_needs_reset", True):
            client.call(func_name="reset")
            self._vla_needs_reset = False
        interval = int(getattr(getattr(env, "obs_manager", None), "collect_interval", 10) or 10)
        r0 = self.compute_reward()
        p0 = self._observable_object_positions()
        taken = 0
        note = ""
        while taken < n_actions and not env.end_flag[0]:
            if env.take_action_cnt[0] >= env.step_lim:
                note = (
                    f"policy action limit reached ({env.take_action_cnt[0]}/{env.step_lim} frames)"
                )
                break
            obs = env.get_obs()
            client.call(func_name="update_obs", obs=obs)
            actions = client.call(func_name="get_action")
            for i, a in enumerate(actions):
                env.take_action(a)
                self._sim_step_count += interval
                self._maybe_capture()
                taken += 1
                if (
                    taken >= n_actions
                    or env.end_flag[0]
                    or self.budget_exhausted()
                    or env.take_action_cnt[0] >= env.step_lim
                ):
                    break
                if i + 1 < len(actions):
                    obs = env.get_obs()
                    client.call(func_name="update_obs", obs=obs)
        r1 = self.compute_reward()
        p1 = self._observable_object_positions()
        moved = {}
        for n in p0.keys() & p1.keys():
            d = float(np.linalg.norm(p1[n] - p0[n]))
            if d > 0.01:
                moved[n] = round(d, 3)
        out = {
            "policy": policy,
            "actions_taken": taken,
            "reward_before": round(r0, 3),
            "reward_after": round(r1, 3),
            "objects_moved_m": moved,
            "budget": self.get_step_budget(),
        }
        if note:
            out["note"] = note
        return out

    def vla_run_until_event(
        self,
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
        """Run bounded G05 replans with public per-action event guards."""

        from roboenv_runtime.adapters.robodojo.policy_events import (
            PolicyEventTracker,
            gripper_snapshot,
            pose_snapshot,
            validate_event_config,
        )
        from roboenv_runtime.adapters.robodojo.vla_manager import (
            _action_batch,
            _canonical_actions_sha256,
            _gripper_motion,
            _jsonable,
            _validated_joint_action,
        )

        if self.env_cfg_type != "arx_x5":
            raise RuntimeError("policy event rollout is supported only for ARX X5")
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
        env = self._env
        action_frames = int(np.asarray(env.take_action_cnt).reshape(-1)[0])
        if (action_frames or self._sim_step_count) and not allow_midstate:
            raise RuntimeError(
                "policy event mid-state handoff refused after "
                f"{action_frames} action frames / {self._sim_step_count} simulation steps"
            )
        clock = getattr(env, "_codex_dojo_score_clock", None)
        if bool(getattr(clock, "submitted", False)):
            raise RuntimeError("policy event rollout is frozen after score submission")

        live_labels = set(self.list_objects())
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

        def public_grippers() -> dict[str, dict[str, Any]]:
            def state(arm: str) -> dict[str, Any]:
                first, second = self.get_finger_positions(arm)
                return {
                    "tip_pose": self.get_ee_pose(arm),
                    "width": float(np.linalg.norm(np.asarray(first) - np.asarray(second))),
                }

            return gripper_snapshot(self.get_ee_pose, state)

        initial_poses = pose_snapshot(self.get_object_pose, labels)
        initial_grippers = public_grippers()
        tracker = PolicyEventTracker(config, initial_poses, initial_grippers)
        before_positions = self._observable_object_positions()
        client = self._ensure_vla("G05")
        if getattr(self, "_vla_needs_reset", True):
            client.call(func_name="reset")
            self._vla_needs_reset = False
        executed: list[Any] = []
        event: dict[str, Any] | None = None
        policy_inferences = 0
        stop_reason = "action_limit"

        def success_latched() -> bool:
            # The event interface never queries the hidden reward manager. In an
            # official run ScoreClock is installed and supplies this public latch;
            # without it, final replay scoring remains the sole authority.
            return bool(getattr(clock, "latched", False)) if clock is not None else False

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
            if stop_when_latched and success_latched():
                return "success_latched"
            remaining = int(self.max_steps) - int(self._sim_step_count)
            if remaining < reserve_sim_steps + interval:
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
                normalized = _validated_joint_action(action)
                native = getattr(clock, "native_action", None)
                context = native() if callable(native) else nullcontext()
                with context:
                    env.take_action(normalized)
                self._sim_step_count += interval
                self._maybe_capture()
                executed.append(_jsonable(normalized))
                event = tracker.observe(
                    pose_snapshot(self.get_object_pose, labels),
                    public_grippers(),
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
                if stop_when_latched and success_latched():
                    stop_reason = "success_latched"
                    halted = True
                    break
                if len(executed) >= config["n_actions"]:
                    stop_reason = "action_limit"
                    halted = True
                    break
            if halted:
                break

        after_positions = self._observable_object_positions()
        moved: dict[str, float] = {}
        for label in sorted(before_positions.keys() & after_positions.keys()):
            distance = float(np.linalg.norm(after_positions[label] - before_positions[label]))
            if distance > 0.001:
                moved[label] = round(distance, 6)
        after_grippers = public_grippers()
        return {
            "mode": "policy_event",
            "policy": "G05",
            "actions_taken": len(executed),
            "requested_actions": config["n_actions"],
            "policy_inferences": policy_inferences,
            "stop_reason": stop_reason,
            "owner": event if event and event["kind"] == "owner_witness" else None,
            "protection": event if event and event["kind"].startswith("protected_") else None,
            "objects_moved_m": moved,
            "grippers_moved_equivalent_m": _gripper_motion(initial_grippers, after_grippers),
            "budget": self.get_step_budget(),
            "provenance": {
                "source": "bounded_continuous_g05_replanning_event_stop",
                "executed_actions_sha256": _canonical_actions_sha256(executed),
                "executed_actions_count": len(executed),
                "event_config": _jsonable(config),
            },
        }

    def _stop_vla(self) -> None:
        srv = getattr(self, "_vla_server", None)
        cli = getattr(self, "_vla_client", None)
        if cli is not None:
            try:
                cli.close()
            except Exception:
                pass
            self._vla_client = None
        if srv is not None and srv.poll() is None:
            import signal as _sig

            try:
                os.killpg(os.getpgid(srv.pid), _sig.SIGTERM)
            except Exception:
                try:
                    srv.terminate()
                except Exception:
                    pass
            self._vla_server = None

    def wait_steps(self, steps: int = 30) -> None:
        """Advance the simulation without moving (settling / pouring dwell)."""
        self.settle(steps)

    def set_gripper(self, arm: str, value: float, steps: int = 60) -> None:
        """value: 1.0 = fully open, 0.0 = fully closed."""
        rm = self._env.robot_manager
        robot = self._robot(arm)
        args = getattr(robot, "robot_args", {}) or {}
        steps = max(int(steps), int(args.get("gripper_min_steps", 0)))
        seq = rm.plan_endeffector_joint(
            env_idx=0,
            arm_tag=robot.arm_name,
            target_val=float(np.clip(value, 0.0, 1.0)),
            need_plan=True,
            control_step_num=steps,
        )
        self._run_control(seq or [])

    def settle(self, steps: int = 30) -> None:
        """Let physics settle without commanding anything (e.g. after a release).

        Uses the raw physics step, not EvalEnv.step(): the latter pops from the
        control queue and asserts when the queue is empty.
        """
        for _ in range(steps):
            self._env.sim_step(render=False)
            self._sim_step_count += 1
            self._maybe_capture()

    # ------------------------------------------------------------ render/video

    def _cam_rgb(self, cam: str) -> np.ndarray | None:
        """A COPY of the named camera's latest colour image (the obs manager's array is
        reused by the next render), or None with the reason printed once."""
        try:
            vision = self._raw_obs()["vision"]
        except Exception as exc:
            self._say_once("render", f"no vision observation: {exc!r}")
            return None
        if cam not in vision:
            self._say_once("render", f"no camera {cam!r}; cameras are {sorted(vision)}")
            return None
        try:
            return np.array(vision[cam]["color"])[..., :3].copy()
        except Exception as exc:
            self._say_once("render", f"camera {cam!r} unreadable: {exc!r}")
            return None

    def _say_once(self, key: str, message: str) -> None:
        seen = getattr(self, "_said_once", None)
        if seen is None:
            seen = self._said_once = set()
        if key not in seen:
            seen.add(key)
            print(f"[robodojo] {message}", flush=True)

    def _head_camera(self) -> str:
        """The head camera key: ``cam_head`` when present, else the first camera whose
        name mentions 'head', else the first camera."""
        try:
            cams = list(self._raw_obs()["vision"])
        except Exception:
            return "cam_head"
        if "cam_head" in cams:
            return "cam_head"
        return next((c for c in cams if "head" in c), cams[0] if cams else "cam_head")

    def render(self, mode: str = "rgb_array") -> np.ndarray | None:  # noqa: ARG002
        """A fresh head-camera render of the CURRENT scene (no physics step)."""
        return self._cam_rgb(self._head_camera())

    def last_video_frame(self) -> np.ndarray | None:
        """The most recent recorded frame (every ``subsample_rate`` sim steps), without
        copying the whole buffer; None when nothing has been recorded."""
        return self._frame_buffer[-1].copy() if self._frame_buffer else None

    def render_wrist(self) -> np.ndarray | None:
        return self._cam_rgb("cam_left_wrist")

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False
    ) -> None:
        self._record_frames = enabled
        self._record_wrist = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._capture()

    def _maybe_capture(self) -> None:
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._capture()

    def _capture(self) -> None:
        head = self._cam_rgb(self._head_camera())
        if head is not None:
            self._frame_buffer.append(head)
        if getattr(self, "_record_wrist", False):
            wrist = self._cam_rgb("cam_left_wrist")
            if wrist is not None:
                self._wrist_frame_buffer.append(wrist.copy())

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames


__all__ = ["RoboDojoEnv", "list_robodojo_tasks", "ROBODOJO_ROOT"]
