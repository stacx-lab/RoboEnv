"""LIBERO's pretrained policy as a tool (``vla_run``): the openpi ``pi05_libero`` hooks.

What the policy saw in training (and in openpi's own ``examples/libero``
evaluation) is reproduced here exactly:

* ``agentview`` and ``robot0_eye_in_hand`` rendered at 256 x 256, passed through
  robosuite's observation convention (a vertical flip), rotated 180 degrees
  (LIBERO's stored images are upside down) and resize-with-padded to 224;
* state = end-effector position (3) + orientation as axis-angle (3) + gripper
  finger positions (2);
* the benchmark's language instruction as the prompt;
* 7-dim actions for robosuite's ``OSC_POSE`` controller at 20 Hz — the roboenv
  LIBERO env is built with ``JOINT_POSITION`` for the IK primitives, so the
  arm's controller is swapped to ``OSC_POSE`` for the duration of a
  ``vla_run`` and restored afterwards (robosuite reloads the controller from
  ``robot.controller_config`` on demand).

Every policy action is one ``env.step`` — one of LIBERO's 4000 horizon steps —
so the budget is charged exactly as for any primitive.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from roboenv_runtime.vla.client import quat_xyzw_to_axisangle, resize_with_pad, rotate180
from roboenv_runtime.vla.spec import PolicySpec

__all__ = ["LiberoPolicyHooks"]


def _image_convention() -> int:
    """The row order robosuite applies to rendered observation images (``img[::c]``).

    ``IMAGE_CONVENTION_MAPPING`` is ``{"opengl": 1, "opencv": -1}``: under the default
    ``opengl`` macro observations are the raw render, untouched.
    """
    try:
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        return int(IMAGE_CONVENTION_MAPPING[str(macros.IMAGE_CONVENTION)])
    except Exception:  # robosuite absent (tests) — the default macro is opengl
        return 1


class LiberoPolicyHooks:
    """:class:`roboenv_runtime.vla.runner.VlaHooks` for the roboenv LIBERO adapter."""

    def __init__(self, adapter: Any, spec: PolicySpec, instruction: str | None = None):
        self.adapter = adapter
        self.spec = spec
        self.lle = adapter._lle
        self.handle = self.lle.handle
        self.prims = adapter._primitives()
        obs = spec.obs or {}
        cams = obs.get("cameras") or {}
        self.base_cam = str(cams.get("base", "agentview"))
        self.wrist_cam = str(cams.get("wrist", "robot0_eye_in_hand"))
        self.render_size = int(obs.get("render_size", 256))
        self.resize = int(obs.get("resize", 224))
        self.rotate = bool(obs.get("rotate180", True))
        self.controller = str((spec.action or {}).get("controller", "OSC_POSE"))
        self.instruction = instruction or str(
            getattr(self.handle, "task_language", "") or adapter.goal()
        )
        self._saved_config: dict[str, Any] | None = None
        self._convention = _image_convention()

    # -- robosuite plumbing -------------------------------------------------------------
    @property
    def _robot(self) -> Any:
        env = self.handle.env
        inner = getattr(env, "env", env)  # LIBERO ControlEnv -> robosuite env
        return inner.robots[0]

    @property
    def _sim(self) -> Any:
        return self.handle.env.sim

    def begin(self) -> None:
        """Switch the arm to the policy's controller (``OSC_POSE``)."""
        import robosuite as suite

        robot = self._robot
        current = dict(robot.controller_config or {})
        if str(current.get("type", "")).upper() == self.controller:
            return
        self._saved_config = current
        robot.controller_config = suite.load_controller_config(default_controller=self.controller)
        robot._load_controller()
        self._refresh_action_dim()

    def _refresh_action_dim(self) -> None:
        """Robosuite caches the env's action width at setup; recompute it after a swap."""
        env = self.handle.env
        inner = getattr(env, "env", env)
        if hasattr(inner, "_action_dim"):
            inner._action_dim = sum(int(r.action_dim) for r in inner.robots)

    def end(self) -> None:
        """Restore the IK primitives' controller and resync their joint/gripper state."""
        robot = self._robot
        if self._saved_config is not None:
            robot.controller_config = self._saved_config
            robot._load_controller()
            self._saved_config = None
            self._refresh_action_dim()
        # the IK primitives track the arm through these; resync them with reality
        lle = self.lle
        lle._current_joints = self._sim.data.qpos[:7].copy()
        with contextlib.suppress(Exception):
            opening = float(self.prims.get_gripper_opening())
            lle._gripper_fraction = float(np.clip(opening / 0.08, 0.0, 1.0))
        with contextlib.suppress(Exception):
            self.prims.api.cfg = None

    # -- observation --------------------------------------------------------------------
    def _camera(self, name: str) -> np.ndarray:
        img = self._sim.render(camera_name=name, width=self.render_size, height=self.render_size)
        img = np.asarray(img)[:: self._convention]  # what robosuite's observation would hold
        if self.rotate:
            img = rotate180(img)
        img = resize_with_pad(np.ascontiguousarray(img), self.resize, self.resize)
        return np.ascontiguousarray(img)

    def _robot_obs(self) -> dict[str, Any]:
        obs = getattr(self.lle, "_current_obs", None)
        if not isinstance(obs, dict) or "robot0_eef_pos" not in obs:
            inner = getattr(self.handle.env, "env", self.handle.env)
            obs = inner._get_observations(force_update=True)
        return obs

    def element(self) -> dict[str, Any]:
        """The policy's observation: two 224x224 images, the 8-dim state, the prompt."""
        obs = self._robot_obs()
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3),
                quat_xyzw_to_axisangle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:2],
            ]
        )
        return {
            "observation/image": self._camera(self.base_cam),
            "observation/wrist_image": self._camera(self.wrist_cam),
            "observation/state": state,
            "prompt": self.instruction,
        }

    # -- control ------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> None:
        """One OSC_POSE control step with the cap-x env's bookkeeping (budget, video)."""
        lle = self.lle
        act = np.asarray(action, dtype=np.float64).reshape(-1)[: self.spec.action_dim]
        stepped = self.handle.step(act)
        lle._current_obs, lle._current_reward, lle._current_done, lle._current_info = stepped
        lle._sim_step_count += 1
        with contextlib.suppress(Exception):
            data = self._sim.data
            lle.gripper_link_wxyz_xyz = np.concatenate(
                [data.xquat[lle.gripper_link_idx], data.xpos[lle.gripper_link_idx]]
            )
        every = getattr(lle, "_subsample_rate", 1) or 1
        if getattr(lle, "_record_frames", False) and lle._sim_step_count % every == 0:
            with contextlib.suppress(Exception):
                lle._record_frame()

    def steps_left(self) -> int:
        """Horizon steps still allowed (never the 4001st)."""
        return int(self.prims._steps_left()) - 1  # never take the 4001st step

    def poses(self) -> dict[str, Any]:
        """Body positions (base frame) — LIBERO has no ``list_objects`` tool."""
        fn = self.adapter.tools().get("get_body_positions")
        return dict(fn()) if callable(fn) else {}

