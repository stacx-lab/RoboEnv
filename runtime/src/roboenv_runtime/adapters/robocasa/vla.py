"""RoboCasa's pretrained policy as a tool (``vla_run``): hooks for ``pi05_robocasa``.

What the RoboCasa365 openpi baseline saw in training (and in the fork's
``examples/robocasa/main.py`` evaluation) is reproduced here:

* ``robot0_agentview_left`` / ``robot0_agentview_right`` / ``robot0_eye_in_hand`` at
  256 x 256: robosuite's observation image flipped upright, exactly as RoboCasa's gym
  wrapper hands it to policies (``gym_wrapper.py``: "flip it upside down"), then
  resize-with-padded to 224 — measured 2026-08-27: without the flip π0.5 emits
  near-zero actions;
* state (16) = end-effector position and quaternion **relative to the base** (3 + 4),
  base position and quaternion in the world (3 + 4), gripper finger positions (2);
* the episode's language instruction as the prompt;
* 12-dim actions: end-effector delta pose (6), gripper close (1), base motion
  (vx, vy, yaw rate, torso: 4), control mode (1) — dispatched to the composite
  controller's ``right`` / ``right_gripper`` / ``base`` / ``torso`` / ``base_mode``
  parts exactly like the gym wrapper's ``unmap_action``.

Every policy action is one environment step through the backend's own ``_step``,
so the step budget and video capture are charged as for any primitive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from roboenv_runtime.vla.client import resize_with_pad
from roboenv_runtime.vla.spec import PolicySpec

__all__ = ["RoboCasaPolicyHooks"]


def _image_convention() -> int:
    """Robosuite's ``IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]`` (1 = untouched)."""
    try:
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        return int(IMAGE_CONVENTION_MAPPING[str(macros.IMAGE_CONVENTION)])
    except Exception:  # robosuite absent (tests)
        return 1


class RoboCasaPolicyHooks:
    """:class:`roboenv_runtime.vla.runner.VlaHooks` for the roboenv RoboCasa adapter."""

    def __init__(self, adapter: Any, spec: PolicySpec, instruction: str | None = None):
        self.adapter = adapter
        self.spec = spec
        self.env = adapter._env  # the RoboCasaEnv backend
        obs = spec.obs or {}
        cams = obs.get("cameras") or {}
        self.base_cam = str(cams.get("base", "robot0_agentview_left"))
        self.wrist_cam = str(cams.get("wrist", "robot0_eye_in_hand"))
        self.right_cam = str(cams.get("right", "robot0_agentview_right"))
        self.render_size = int(obs.get("render_size", 256))
        self.resize = int(obs.get("resize", 224))
        self.flip_vertical = bool(obs.get("flip_vertical", True))
        self.instruction = instruction or str(self.env.instruction() or adapter.goal())
        self._convention = _image_convention()
        self._parts: list[tuple[str, int]] | None = None

    def begin(self) -> None:
        """Nothing to switch: RoboCasa's controllers are the ones the policy was trained on.

        Drops the scored-joint cache so a run after a reset looks at the new episode's model.
        """
        self._parts = None

    def end(self) -> None:
        """Nothing to restore."""

    def _camera(self, name: str) -> np.ndarray:
        img = self.env.sim.render(camera_name=name, width=self.render_size, height=self.render_size)
        img = np.asarray(img)[:: self._convention]  # robosuite's observation image ...
        if self.flip_vertical:
            img = img[::-1]  # ... which RoboCasa's gym wrapper flips upright for its policies
        img = resize_with_pad(np.ascontiguousarray(img), self.resize, self.resize)
        return np.ascontiguousarray(img)

    def element(self) -> dict[str, Any]:
        """Three 224x224 images, the 16-dim state and the prompt."""
        obs = self.env._obs
        state = np.concatenate(
            [
                np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float64).reshape(3),
                np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float64).reshape(4),
                np.asarray(obs["robot0_base_pos"], dtype=np.float64).reshape(3),
                np.asarray(obs["robot0_base_quat"], dtype=np.float64).reshape(4),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:2],
            ]
        )
        return {
            "observation/image": self._camera(self.base_cam),
            "observation/wrist_image": self._camera(self.wrist_cam),
            "observation/right_image": self._camera(self.right_cam),
            "observation/state": state,
            "prompt": self.instruction,
        }

    @staticmethod
    def action_dict(action: np.ndarray) -> dict[str, Any]:
        """The composite-controller parts for one 12-dim policy action (gym ``unmap_action``)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < 12:
            raise ValueError(f"RoboCasa policy actions are 12-dim, got {a.shape[0]}")
        return {
            "right": a[0:6],
            "right_gripper": np.array([-1.0 if a[6] < 0.5 else 1.0]),
            "base": a[7:10],
            "torso": a[10:11],
            "base_mode": -1.0 if a[11] < 0.5 else 1.0,
        }

    def step(self, action: np.ndarray) -> None:
        """One environment step through the backend (budget + video accounted)."""
        self.env._step(self.action_dict(action))

    def steps_left(self) -> int:
        """Steps before the task horizon (advisory in RoboCasa, honoured here)."""
        budget = self.env.get_step_budget()
        return int(budget["max"]) - int(budget["used"])

    def _moving_parts(self) -> list[tuple[str, int]]:
        """``(joint name, body id)`` for every hinge/slide joint of the task's fixtures.

        Half of RoboCasa's atomic suite is scored on a fixture joint, not on an object, and
        the door/drawer/knob that the joint moves is not in ``observe()["objects"]``.
        Cached per ``vla_run`` (the model does not change inside an episode).
        """
        try:
            model = self.env.sim.model
            raw_m = getattr(model, "_model", model)
            fixtures = sorted(set((self.env.task_fixtures() or {}).values()), key=len, reverse=True)
        except Exception:  # a task with no fixture refs, or a backend without a live model
            return []
        parts: list[tuple[str, int]] = []
        for jn in model.joint_names:
            if not any(jn.startswith(f"{fx}_") for fx in fixtures):
                continue
            try:
                jid = model.joint_name2id(jn)
                if int(raw_m.jnt_type[jid]) not in (2, 3):  # slide, hinge
                    continue
                parts.append((jn, int(raw_m.jnt_bodyid[jid])))
            except Exception:
                continue
        return parts

    def poses(self) -> dict[str, Any]:
        """Object positions, plus the moving part of every fixture joint this task scores.

        Without the second half ``vla_run``'s ``objects_moved_m`` is ``{}`` for every
        articulated-fixture task, and :func:`roboenv.control.patterns.stalled` -- the campaign's
        stopping rule -- reads a swinging door as "nothing moved".  Measured 2026-08-28 on
        the certified ``robocasa.OpenCabinet`` run: the door goes 0 -> 1.0 by frame 600
        while the first three 150-frame chunks reported ``{}``, so the default ``give_up``
        fired at frame 300.  A part is reported by the world centre of mass of the body its
        joint moves (``xipos``): that point travels for a hinge as well as for a slide,
        which the body frame origin (on the hinge axis) does not.
        """
        out: dict[str, Any] = {}
        for name, item in (self.adapter.observe().get("objects") or {}).items():
            pos = item.get("position") if isinstance(item, dict) else None
            if pos is not None:
                out[name] = pos
        if self._parts is None:
            self._parts = self._moving_parts()
        if self._parts:
            try:
                data = self.env.sim.data
                raw_d = getattr(data, "_data", data)
                for jn, bid in self._parts:
                    out[f"joint:{jn}"] = np.array(raw_d.xipos[bid], dtype=float)
            except Exception:  # never let a report break the run
                pass
        return out
