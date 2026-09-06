"""RoboTwin's pretrained policy as a tool (``vla_run``): hooks for InternVLA-A1.5.

What the XPolicyLab RoboTwin evaluation (``scripts/eval_policy_xpolicylab.py``)
feeds a policy is reproduced here, key for key:

* ``vision/cam_head|cam_left_wrist|cam_right_wrist/color`` — RoboTwin's own
  ``head_camera`` / ``left_camera`` / ``right_camera`` RGB frames straight out of
  ``task.get_obs()`` (uint8 HWC, 320x240 for the D435 profile), with their
  intrinsics and extrinsics; the policy server resize-with-pads them to 224;
* ``state`` — the four fields of XPolicyLab's dual-arm joint layout
  (``left_arm_joint_state`` 6, ``left_ee_joint_state`` 1, ``right_arm_joint_state`` 6,
  ``right_ee_joint_state`` 1), which is exactly ``joint_action["vector"]`` split up;
* ``instruction`` — the episode's RoboTwin sentence (see :meth:`episode_instruction`).

An action is the 14-dim joint target ``[left arm 6, left gripper, right arm 6,
right gripper]`` and is applied with RoboTwin's own
``task.take_action(action, action_type="qpos")``: the benchmark's TOPP
interpolation, gripper ramp and per-step success check, i.e. the control path the
policy's published numbers were measured on.

``take_action`` steps ``scene`` directly, so it would charge the harness nothing
and record no video.  :meth:`step` therefore wraps the task's ``_update_render``
for the duration of the call — ``take_action`` calls it exactly once per physics
step plus two book-keeping renders (one before the control loop, one after it or
inside the ``get_obs()`` on the success path) — and charges the difference to the
backend's ``_sim_step_count`` while driving its video tick.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from roboenv_runtime.paths import robotwin_root
from roboenv_runtime.vla.spec import PolicySpec

__all__ = ["RoboTwinPolicyHooks"]

#: ``take_action`` renders once before its control loop and once after it (or
#: inside the ``get_obs()`` it runs when success latches): those two are not
#: physics steps.
TAKE_ACTION_RENDER_OVERHEAD = 2

#: Upstream's fallback when a task is missing from ``_eval_step_limit.yml``.
DEFAULT_STEP_LIMIT = 1000


def _clean(text: str) -> str:
    """Strip RoboTwin's ``<...>`` emphasis markers and collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("<", "").replace(">", "")).strip()


class RoboTwinPolicyHooks:
    """:class:`roboenv_runtime.vla.runner.VlaHooks` for the roboenv RoboTwin adapter."""

    def __init__(self, adapter: Any, spec: PolicySpec, instruction: str | None = None):
        self.adapter = adapter
        self.spec = spec
        self.env = adapter._lle  # the RoboTwinEnv backend
        obs = spec.obs or {}
        cams = obs.get("cameras") or {}
        self.head_cam = str(cams.get("head", "head_camera"))
        self.left_cam = str(cams.get("left_wrist", "left_camera"))
        self.right_cam = str(cams.get("right_wrist", "right_camera"))
        self.frequency = int(obs.get("frequency", 30))
        self.action_type = str(spec.action.get("action_type", "qpos"))
        self.instruction = instruction or self.episode_instruction()
        self.step_limit = self._eval_step_limit()
        self._cwd: str | None = None

    # ------------------------------------------------------------ language

    def episode_instruction(self) -> str:
        """The episode's RoboTwin sentence, by the benchmark's own rules.

        RoboTwin builds an instruction by filling the ``{A}``/``{a}`` placeholders of
        ``description/task_instruction/<task>.json`` from ``task.info["info"]`` — but
        that dict is only written by ``play_once`` and the backend rebuilds the scene
        after its expert dry-run, so it is normally empty here.  The generator is still
        tried first (it is correct whenever the info survives); otherwise the task's own
        ``full_description``, which is a complete one-sentence goal, is used.
        """
        task = self.env.task
        with contextlib.suppress(Exception):
            existing = task.get_instruction()
            if existing:
                return str(existing)
        name = str(self.env.task_name)
        info = {}
        with contextlib.suppress(Exception):
            info = dict(task.info.get("info") or {})
        generated = self._generated_instruction(name, info)
        if generated:
            return generated
        path = Path(robotwin_root()) / "description" / "task_instruction" / f"{name}.json"
        with contextlib.suppress(OSError, ValueError):
            described = json.loads(path.read_text(encoding="utf-8")).get("full_description")
            if described:
                return _clean(str(described))
        return name.replace("_", " ")

    @staticmethod
    def _generated_instruction(task_name: str, info: dict[str, Any]) -> str:
        """One ``seen`` sentence from RoboTwin's own episode-instruction generator.

        The generator shuffles and samples from the process-wide ``random``, prints
        when no template fits, and may ``exit()`` on a missing description file — so
        the RNG state is restored around the call, its chatter is swallowed, and
        ``SystemExit`` is treated as "no instruction".
        """
        utils = str(Path(robotwin_root()) / "description" / "utils")
        if utils not in sys.path:
            sys.path.append(utils)
        state = random.getstate()
        random.seed(hash(task_name) & 0xFFFFFFFF)
        try:
            from generate_episode_instructions import generate_episode_descriptions

            with contextlib.redirect_stdout(io.StringIO()):
                results = generate_episode_descriptions(task_name, [info], 1)
            candidates = results[0].get("seen") or results[0].get("unseen") or []
            return _clean(str(candidates[0])) if candidates else ""
        except (Exception, SystemExit):
            return ""
        finally:
            random.setstate(state)

    # ------------------------------------------------------------ lifecycle

    def begin(self) -> None:
        """Pin the simulator's working directory (RoboTwin resolves assets from it).

        Nothing else has to change hands: ``take_action`` drives the same PD targets
        the planner route drives, and it never touches the backend's sticky
        ``plan_success`` flag or its tracked planning world.
        """
        self._cwd = os.getcwd()
        env_cwd = getattr(self.adapter, "_env_cwd", None)
        if env_cwd and env_cwd != self._cwd:
            os.chdir(env_cwd)

    def end(self) -> None:
        """Restore the caller's working directory."""
        if self._cwd and os.getcwd() != self._cwd:
            with contextlib.suppress(OSError):
                os.chdir(self._cwd)
        self._cwd = None

    # ------------------------------------------------------------ observation

    def _camera(self, raw: dict[str, Any], name: str) -> dict[str, Any]:
        camera = raw.get(name) or {}
        if "rgb" not in camera:
            raise KeyError(
                f"RoboTwin observation has no RGB for camera {name!r}; the episode renders "
                f"{sorted(raw)} — the policy needs head plus both wrist cameras"
            )
        color = np.ascontiguousarray(np.asarray(camera["rgb"], dtype=np.uint8))
        entry: dict[str, Any] = {"color": color, "shape": tuple(color.shape[:2])}
        if "intrinsic_cv" in camera:
            entry["intrinsic_matrix"] = np.asarray(camera["intrinsic_cv"], dtype=np.float32)
        if "extrinsic_cv" in camera:
            entry["extrinsics_matrix"] = np.asarray(camera["extrinsic_cv"], dtype=np.float32)
        return entry

    def state(self, raw: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        """XPolicyLab's dual-arm joint state (4 fields, 6 + 1 + 6 + 1 = 14)."""
        joint = (raw if raw is not None else self.env.task.get_obs())["joint_action"]
        return {
            "left_arm_joint_state": np.asarray(joint["left_arm"], dtype=np.float32),
            "left_ee_joint_state": np.asarray([joint["left_gripper"]], dtype=np.float32),
            "right_arm_joint_state": np.asarray(joint["right_arm"], dtype=np.float32),
            "right_ee_joint_state": np.asarray([joint["right_gripper"]], dtype=np.float32),
        }

    def element(self) -> dict[str, Any]:
        """The XPolicyLab observation dictionary the policy server expects.

        Depth is deliberately left out: ``task.get_obs()`` renders it (the backend
        turns it on for the harness's own tools) but no RoboTwin policy in the zoo
        reads it, and it would triple the size of every websocket frame.
        """
        raw = self.env.task.get_obs()
        cameras = raw["observation"]
        return {
            "data_format_version": "v1.0",
            "instruction": self.instruction,
            "instructions": [self.instruction],
            "env_idx": 0,
            "vision": {
                "cam_head": self._camera(cameras, self.head_cam),
                "cam_left_wrist": self._camera(cameras, self.left_cam),
                "cam_right_wrist": self._camera(cameras, self.right_cam),
            },
            "state": self.state(raw),
            "additional_info": {"frequency": self.frequency},
        }

    # ------------------------------------------------------------ control

    def step(self, action: np.ndarray) -> None:
        """Apply ONE 14-dim joint target through RoboTwin's own ``take_action``.

        The backend's ``_sim_step_count`` and video buffer are charged for exactly the
        physics steps the interpolated motion took.
        """
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self.spec.action_dim:
            raise ValueError(
                f"RoboTwin policy actions are {self.spec.action_dim}-dim, got {vector.shape[0]}"
            )
        env = self.env
        task = env.task
        original = task._update_render
        renders = 0

        def counted() -> None:
            nonlocal renders
            original()
            renders += 1
            if renders > 1 and env._record_frames and renders % env._subsample_rate == 0:
                env._video_tick()

        task._update_render = counted
        try:
            task.take_action(vector, action_type=self.action_type)
        finally:
            task._update_render = original
            env._sim_step_count += max(0, renders - TAKE_ACTION_RENDER_OVERHEAD)

    # ------------------------------------------------------------ budget

    def _eval_step_limit(self) -> int:
        """RoboTwin's own per-task policy-frame budget (``_eval_step_limit.yml``)."""
        path = Path(robotwin_root()) / "env_cfg" / "task_config" / "_eval_step_limit.yml"
        try:
            import yaml

            limits = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return int(limits[str(self.env.task_name)])
        except (OSError, KeyError, TypeError, ValueError):
            return DEFAULT_STEP_LIMIT

    def solved(self) -> bool:
        """Whether the episode is already over (RoboTwin latches its own success)."""
        task = self.env.task
        if getattr(task, "eval_success", False) or getattr(self.env, "_success_latched", False):
            return True
        with contextlib.suppress(Exception):
            return bool(self.env.task_completed())
        return False

    def steps_left(self) -> int:
        """Policy frames still allowed: the benchmark's limit, the sim budget, success.

        ``run_policy`` counts one call per policy frame, so this is in frames — one
        frame is a joint-target waypoint the simulator interpolates over ~30 physics
        steps.  The physics budget (``max_steps``) is honoured as a hard gate rather
        than converted, because the steps a waypoint costs vary with its length.
        """
        env = self.env
        if self.solved():
            return 0
        used_sim = int(getattr(env, "_sim_step_count", 0))
        max_sim = int(getattr(env, "max_steps", 0) or 0)
        if max_sim and used_sim >= max_sim:
            return 0
        frames = int(getattr(env.task, "take_action_cnt", 0))
        return max(0, self.step_limit - frames)

    def poses(self) -> dict[str, Any]:
        """Object positions from the adapter's normalised snapshot."""
        out: dict[str, Any] = {}
        for name, item in (self.adapter.observe().get("objects") or {}).items():
            pos = item.get("position") if isinstance(item, dict) else None
            if pos is not None:
                out[name] = pos
        return out
