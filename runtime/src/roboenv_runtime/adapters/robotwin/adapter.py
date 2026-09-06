"""RoboTwin 2.0 (dual Agilex Piper, SAPIEN) — bridges the cap-x integration.

Runs inside ``.venv-robotwin`` (SAPIEN needs no RTX GPU).  The cap-x
low-level worker chdirs into the RoboTwin checkout, so any ``output_dir`` in
a config must be absolute.

Any task from cap-x's RoboTwin registry works: pass ``task_name`` and the
adapter synthesises the env node around the generic ``RoboTwinCodeEnv`` with
``robotwin_<task_name>_<embodiment>_low_level``.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from roboenv_runtime.adapters.base import CapxBridge, ensure_capx_importable
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import Capability
from roboenv_runtime.embodiments import load_embodiment

__all__ = ["RoboTwinAdapter"]


@register_adapter
class RoboTwinAdapter(CapxBridge):
    """RoboTwin task ``task_name`` with the ``PiperPlannerPrivilegedApi`` tool set."""

    name: ClassVar[str] = "robotwin"
    embodiment: ClassVar[str] = "piper"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {Capability.PRIVILEGED, Capability.VIDEO, Capability.CHECKPOINT}
    )

    def __init__(self, task_name: str = "handover_block", embodiment: str = "piper", **_: Any):
        ensure_capx_importable()
        import yaml

        # importing the backend registers robotwin_<task>_*_low_level with cap-x,
        # and the API modules register their names — both idempotent
        import roboenv_runtime.adapters.robotwin.backend  # noqa: F401
        from roboenv_runtime.adapters.robotwin import api_joint, api_planner
        from roboenv_runtime.adapters.robotwin.embodiments import register_extra_envs

        register_extra_envs()  # aloha_agilex and friends, for the pretrained policies

        api_planner.register()
        api_joint.register()
        self.task_name = task_name
        self.embodiment = embodiment  # instance override of the class-level skill domain
        node = {
            "env": {
                "_target_": "capx.envs.tasks.piper.robotwin_tasks.RoboTwinCodeEnv",
                "cfg": {
                    "_target_": "capx.envs.tasks.base.CodeExecEnvConfig",
                    "low_level": f"robotwin_{task_name}_{embodiment}_low_level",
                    "privileged": True,
                    "apis": ["PiperPlannerPrivilegedApi"],
                },
            }
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, prefix=f"robotwin_{task_name}_"
        ) as tmp:
            yaml.safe_dump(node, tmp)
            path = tmp.name
        try:
            super().__init__(path)
        finally:
            os.unlink(path)

    def situation(self) -> str:
        """From the embodiment profile (``piper``)."""
        return load_embodiment(self.embodiment).render_situation()

    def facts(self) -> str:
        """From the embodiment profile."""
        return load_embodiment(self.embodiment).render_facts()

    def extra_tools(self) -> dict[str, Callable[..., Any]]:
        """Contact, TCP and articulation observations the planner API lacks."""

        def get_contact_positions(object_name: str, arm: str | None = None) -> list:
            """World positions where a gripper is CURRENTLY in physical
            contact with the named object (physics-engine contact points with
            a non-zero impulse or a separation of at most 1 mm — pairs that
            PhysX reports merely inside its 1 cm contact offset are dropped).
            Empty list = not touching. Use this to verify touch/press
            interactions that do not move the object.

            arm: "left" | "right" restricts the answer to THAT hand's finger
            links; None (default) accepts either hand. Contacts are matched by
            entity identity, so the side is honoured even though both Piper
            arms name their fingers link7/link8.

            Articulated objects report contacts on ANY of their links here —
            touching the cabinet's frame counts as touching "cabinet". Use
            check_grasp() when you need to know WHICH part you are holding.
            """
            return self._lle.gripper_contacts(object_name, arm)["positions"]

        def check_grasp(
            object_name: str,
            arm: str | None = None,
            part: str | list | None = None,
            contact_point_id: int | list | None = None,
        ) -> dict:
            """Is this hand really holding that thing? Ground truth from the
            physics engine, matched by ENTITY IDENTITY rather than by name.

            This is the honest answer to "did my grasp close on the object or
            on air?", and for articulated objects to "am I holding the DRAWER
            or resting on the cabinet's frame?" — a distinction the raw
            simulator query cannot make, because both Piper arms name their
            finger links link7/link8 and a cabinet names its links
            link_0..link_3.

            Args:
                object_name: see list_objects().
                arm: "left" | "right" | None (either hand).
                part: link name of an articulated object to require. Link
                    names: 'object_links' of a check_grasp() that is touching,
                    or the RuntimeError from a wrong name lists them all
                    ("... has no link named [...]; links are [...]"). A rigid
                    object has no links: passing part= for it raises.
                contact_point_id: instead of `part`, require the link that
                    this annotated contact point rides on — the same id you
                    pass to grasp_object.

            Returns {'touching': bool, 'arms': [...], 'object_links': [...],
            'finger_links': [...], 'positions': [[x,y,z], ...],
            'required_link': [str]|None, 'min_separation_m', 'max_impulse'}.
            'touching' counts only REAL contact (impulse > 0 or separation
            <= 1 mm): an open gripper hovering 1 cm over a box reads pair
            points with separation +12 mm and zero impulse, and those are
            NOT touching. 'object_links' names the parts your fingers are
            actually on; a lift that shows 'touching' False is holding
            nothing regardless of what the grasp call returned. `arm` is
            case-sensitive ('left' / 'right'); anything else raises.
            """
            return self._lle.gripper_contacts(
                object_name, arm, link_name=part, contact_point_id=contact_point_id
            )

        def get_tcp_pose(arm: str = "left") -> list:
            """Gripper-center (TCP) pose — the point BETWEEN the fingertips —
            in the world frame: [x, y, z, qw, qx, qy, qz]. This is where the
            hand actually touches things. It sits exactly 0.12 m from the
            end-effector frame that get_gripper_pose_left/right report, along
            that frame's local +x (the approach direction) — verified on both
            arms. So a top-down tool has its end-effector 12 cm ABOVE the
            fingertips, while a tool pitched 30 deg has it ~10 cm above and
            ~6 cm back toward the robot, which is why pitch buys reach.
            Use THIS pose when judging heights and offsets for grasping,
            pushing or pressing. arm: "left" | "right".
            """
            robot = self._lle.task.robot
            fn = robot.get_left_tcp_pose if arm == "left" else robot.get_right_tcp_pose
            return [float(v) for v in fn()]

        def get_joint_states() -> dict:
            """Joint positions AND limits of ARTICULATED scene objects
            (laptop lid, microwave door, cabinet drawer, switch...):
            {name: {"qpos": [...], "qlimits": [[lo, hi], ...]}}. Progress
            on open/close/turn goals shows up HERE — the object's base pose
            does not move when a lid or door rotates. Robot excluded.

            Note that a FUNCTIONAL POINT anchored to a moving part does move
            with it: get_object_functional_pose() on a cabinet returns a spot
            inside the drawer, which travels toward the robot as the drawer
            opens. Re-read it after every change of joint state.
            """
            out: dict[str, dict[str, list]] = {}
            for attr, value in self._lle.task.__dict__.items():
                if attr == "robot":
                    continue
                if callable(getattr(value, "get_qpos", None)) and callable(
                    getattr(value, "get_qlimits", None)
                ):
                    try:
                        out[attr] = {
                            "qpos": np.asarray(value.get_qpos()).flatten().tolist(),
                            "qlimits": np.asarray(value.get_qlimits()).reshape(-1, 2).tolist(),
                        }
                    except Exception:
                        continue
            return out

        def get_gripper_state(arm: str = "left") -> dict:
            """Is this gripper OPEN or CLOSED — by the benchmark's own
            definition, the one a task's success check reads.

            Several RoboTwin tasks score the gripper directly ("the receiving
            gripper is closed and the original one is open").  Measured
            2026-08-27 (round 3): those predicates are thresholds on the LAST
            COMMANDED gripper fraction (> 0.8 open, < 0.2 closed;
            robot.py is_<arm>_gripper_open/close), NOT on the fingers — a jaw
            clamped on a 5 cm object reads closed=True, and open flips the
            moment open_gripper is issued whether or not the fingers moved.
            They disagree with you only after a PARTIAL command
            (close_gripper(0.3), open_gripper(0.7)) or a gripper call that
            never ran.  Whether the fingers actually hold something is a
            separate fact: use check_grasp.

            Args:
                arm: "left" | "right".

            Returns {'commanded': float|None (the last commanded 0..1 fraction
            the predicates read; also under the legacy key 'opening'),
            'open': bool, 'closed': bool (the benchmark's own
            is_<arm>_gripper_open / is_<arm>_gripper_close), 'finger_qpos':
            [joint positions of the gripper joints] or None, 'measured_fraction':
            the base finger joint's position normalised over the gripper scale
            (None when unreadable), 'has_gripper': bool}.  Raises ValueError
            for an unknown arm or an embodiment without a gripper.
            """
            if arm not in ("left", "right"):
                raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
            task = self._lle.task
            robot = task.robot
            joints = getattr(robot, f"{arm}_gripper", None) or []
            if not joints:
                raise ValueError(f"the {arm} arm of this embodiment has no gripper")
            value = getattr(robot, f"{arm}_gripper_val", None)
            finger_qpos = None
            measured = None
            try:
                entity = getattr(robot, f"{arm}_entity")
                active = list(entity.get_active_joints())
                qpos = np.asarray(entity.get_qpos(), dtype=float).reshape(-1)
                idx = [active.index(j[0]) for j in joints if j[0] in active]
                finger_qpos = [float(qpos[i]) for i in idx]
                scale = getattr(robot, f"{arm}_gripper_scale", None)
                if finger_qpos and scale is not None and float(scale[1]) != float(scale[0]):
                    measured = float(
                        (finger_qpos[0] - float(scale[0])) / (float(scale[1]) - float(scale[0]))
                    )
            except Exception:  # a read-only extra, never a failure
                pass
            return {
                "commanded": None if value is None else float(value),
                "opening": None if value is None else float(value),
                "open": bool(getattr(task, f"is_{arm}_gripper_open")()),
                "closed": bool(getattr(task, f"is_{arm}_gripper_close")()),
                "finger_qpos": finger_qpos,
                "measured_fraction": None if measured is None else round(measured, 4),
                "has_gripper": True,
            }

        return {
            "get_contact_positions": get_contact_positions,
            "check_grasp": check_grasp,
            "get_tcp_pose": get_tcp_pose,
            "get_joint_states": get_joint_states,
            "get_gripper_state": get_gripper_state,
        }
