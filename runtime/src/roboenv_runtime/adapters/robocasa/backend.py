"""RoboCasa365 backend: a robosuite ``Kitchen`` environment driven by pose deltas.

RoboCasa (https://robocasa.ai) builds on robosuite ≥ 1.5 with composite
controllers.  The default robot ``PandaOmron`` exposes four parts:
``right`` (an operational-space controller taking pose *deltas in the base
frame*), ``right_gripper`` (-1 open … +1 close), ``base`` (velocities in the
base's current heading frame: forward, left, yaw rate) and ``torso``.  There is
no motion planner, so every high-level primitive here is a closed-loop servo
built from small deltas and checked against ground truth from MuJoCo.

Measured on robosuite master / robocasa a07e365 (2026-08-26), 20 Hz control:
a unit arm delta moves the grip site ≈ 5.7 mm per step; a unit base velocity
≈ 2.1 cm/step and ≈ 0.065 rad/step.

Only imported inside the ``robocasa`` interpreter (conda env ``robocasa``:
Python 3.11, robosuite master, mujoco 3.3, the 23 GB kitchen assets).
"""

from __future__ import annotations

import ast
import contextlib
import heapq
import inspect
import itertools
import logging
import os
import re
import textwrap
from typing import Any

import numpy as np

__all__ = ["RoboCasaEnv", "list_robocasa_tasks"]

log = logging.getLogger(__name__)
TRACE = bool(os.environ.get("ROBOENV_RC_TRACE"))


_SNAP_DIR = os.environ.get("ROBOENV_RC_SNAP")
_SNAP_COUNT = [0]
_SNAP_ENV: list[Any] = []


def _trace(msg: str) -> None:
    """Diagnostic prints for primitive development (``ROBOENV_RC_TRACE=1``); with
    ``ROBOENV_RC_SNAP=<dir>`` every trace line also saves the agent-view frame there.
    """
    if TRACE:
        print(f"[rc] {msg}", flush=True)
    if _SNAP_DIR and _SNAP_ENV:
        with contextlib.suppress(Exception):
            import imageio

            frame = _SNAP_ENV[-1].render()
            if frame is not None:
                _SNAP_COUNT[0] += 1
                tag = "".join(c if c.isalnum() else "_" for c in msg[:40])
                imageio.imwrite(f"{_SNAP_DIR}/{_SNAP_COUNT[0]:03d}_{tag}.png", frame)


DEFAULT_ROBOT = "PandaOmron"
DEFAULT_TASK = "PickPlaceCounterToCabinet"
ARM = "right"
GRIPPER = "right_gripper"
BASE = "base"
GRIP_SITE = "gripper0_right_grip_site"
BASE_BODY = "mobilebase0_base"
BASE_MIN_CMD = 0.5  # smallest base velocity command that actually moves it (deadband below)
BASE_MIN_TURN = 0.3

# Finger-pad geometry along the tool axis, measured 2026-08-28: an open gripper reads
# 0.0851 m between the pads and a gripper closed on nothing reads 0.0080-0.0100.
GRIPPER_MAX_OPENING = 0.085
GRIPPER_EMPTY_OPENING = 0.016  # above this, the jaws are on material


def _ensure_offscreen_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def list_robocasa_tasks() -> list[str]:
    """Names of every registered kitchen environment (374 in RoboCasa365)."""
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    return sorted(ALL_KITCHEN_ENVIRONMENTS)


def _wxyz_from_mat(mat: np.ndarray) -> np.ndarray:
    import robosuite.utils.transform_utils as tu

    x, y, z, w = tu.mat2quat(np.asarray(mat, dtype=float).reshape(3, 3))
    return np.array([w, x, y, z], dtype=float)


def _xyzw(q_wxyz: Any) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _plain(v: Any) -> Any:
    """Numpy scalars/arrays and nested containers as plain JSON-able Python values."""
    if isinstance(v, np.ndarray):
        return v.round(5).tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


class RoboCasaEnv:
    """One kitchen task instance with a small, verified control surface."""

    def __init__(
        self,
        task_name: str = DEFAULT_TASK,
        *,
        robot: str = DEFAULT_ROBOT,
        layout_ids: Any = None,
        style_ids: Any = None,
        seed: int | None = None,
        max_steps: int | None = None,
        enable_render: bool = True,
        camera_names: tuple[str, ...] = ("robot0_agentview_center",),
        camera_size: tuple[int, int] = (256, 256),
        obj_instance_split: str | None = None,
    ) -> None:
        _ensure_offscreen_rendering()
        import robocasa  # noqa: F401  (importing it registers the kitchen environments)
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        self.task_name = task_name
        self.robot = robot
        self.seed = seed
        self._render = bool(enable_render)
        self._camera_names = tuple(camera_names)
        self._camera_size = tuple(camera_size)
        self._max_steps = max_steps
        self._step_count = 0
        self._frames: list[np.ndarray] = []
        self._capture = False
        controller_config = load_composite_controller_config(robot=robot)
        env_kwargs: dict[str, Any] = {
            "robots": robot,
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": bool(enable_render),
            "use_camera_obs": bool(enable_render),
            "camera_names": list(camera_names) if enable_render else None,
            "camera_heights": camera_size[0],
            "camera_widths": camera_size[1],
            "ignore_done": True,
            "translucent_robot": False,
            "seed": seed,
        }
        if layout_ids is not None:
            env_kwargs["layout_ids"] = layout_ids
        if style_ids is not None:
            env_kwargs["style_ids"] = style_ids
        if obj_instance_split is not None:
            env_kwargs["obj_instance_split"] = obj_instance_split
        self.env = robosuite.make(task_name, **env_kwargs)
        self.horizon = int(getattr(self.env, "horizon", 1000) or 1000)
        self._obs: dict[str, Any] | None = None
        if _SNAP_DIR:
            _SNAP_ENV.append(self)

    def _ensure_built(self) -> None:
        """Robosuite builds the kitchen (fixtures, objects, language) on the first reset;
        prompts read those before the trial's own reset, so build once on demand.
        """
        if not hasattr(self.env, "fixtures") or not hasattr(self.env, "obj_body_id"):
            self.reset(seed=self.seed)

    @property
    def _robot(self) -> Any:
        # robosuite rebuilds its Robot objects on every reset, so never cache one.
        return self.env.robots[0]

    @property
    def sim(self) -> Any:
        """The MuJoCo simulation wrapper."""
        return self.env.sim

    # ------------------------------------------------------------ lifecycle

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Reset the episode (a new layout/style/object draw for a new seed)."""
        if seed is not None:
            # robosuite draws the layout/style and every object placement from ``env.rng``;
            # reseed on EVERY reset so reset(seed=k) always rebuilds the same episode
            # (measured 2026-08-27: without this, two reset(0) calls gave layouts 52 and 27).
            self.seed = seed
            self.env.rng = np.random.default_rng(seed)
        self._obs = self.env.reset()
        self._step_count = 0
        self._frames.clear()
        self._home_rel = self._eef_in_base_frame()
        self._home_torso = self.torso_height() if self._has_torso() else 0.0
        return self._obs

    def _has_torso(self) -> bool:
        """Whether the robot model carries the Omron torso-lift joint."""
        return self.TORSO_JOINT in self.sim.model.joint_names

    def _eef_in_base_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Grip-site pose expressed in the base's (x, y, yaw) frame."""
        pose = self.get_eef_pose()
        bx, by, byaw = self.get_base_pose()
        rel_pos = _rot_z(-byaw) @ (pose[:3] - np.array([bx, by, 0.0]))
        rel_quat = self._rotate_quat(pose[3:7], np.array([0.0, 0.0, 1.0]), -byaw)
        return rel_pos, rel_quat

    def tuck_arm(self, max_steps: int = 120, release: bool = False) -> dict[str, Any]:
        """Bring the arm back to its reset pose relative to the base (raised, close in)
        before driving, so an extended arm does not ram the furniture the base approaches.

        It also releases the OSC null-space posture back to the reset one (a
        ``goto_pose_posture`` leaves the spring pointed at its IK posture, which biases
        every later Cartesian servo of the episode).
        The gripper is left as it is (a carried object stays carried) unless ``release``
        is True.  The arm first backs off 6 cm along its tool axis AT ITS CURRENT HEIGHT
        (rising first jams the wrist against a counter overhang when the hand is at a
        drawer handle — measured 2026-08-27, CloseDrawer) so the straight-line servo home
        does not start inside a cavity.  Returns the ``goto_pose`` report plus ``reason``
        when it could not.
        """
        with contextlib.suppress(Exception):
            self.set_nullspace_posture(None)
        rel_pos, rel_quat = getattr(self, "_home_rel", (None, None))
        if rel_pos is None:
            return {"reached": False, "reason": "no reset pose recorded"}
        bx, by, byaw = self.get_base_pose()
        pos = np.array([bx, by, 0.0]) + _rot_z(byaw) @ rel_pos
        pos[2] += self.torso_height() - getattr(self, "_home_torso", 0.0)
        quat = self._rotate_quat(rel_quat, np.array([0.0, 0.0, 1.0]), byaw)
        if release:
            self.set_gripper(closed=False)
        try:
            import robosuite.utils.transform_utils as tu

            tip = self.get_eef_pose()
            tool = np.asarray(tu.quat2mat(_xyzw(tip[3:7])), dtype=float)[:, 2]
            back = tip[:3] - 0.06 * tool
            with contextlib.suppress(RuntimeError):
                self.goto_pose(back, tip[3:7], tolerance_m=0.03, tolerance_rad=0.5, max_steps=40)
            return self.goto_pose(
                pos, quat, tolerance_m=0.04, tolerance_rad=0.3, max_steps=max_steps
            )
        except RuntimeError as exc:
            return {"reached": False, "reason": str(exc)[:200]}

    def close(self) -> None:
        """Close the simulation."""
        self.env.close()

    # ------------------------------------------------------------ language / score

    def episode_meta(self) -> dict[str, Any]:
        """RoboCasa's episode metadata (layout, style, language, fixture refs ...)."""
        self._ensure_built()
        try:
            return dict(self.env.get_ep_meta())
        except Exception:
            return {}

    def goal_text(self) -> str:
        """The task's language instruction for this episode."""
        return str(self.episode_meta().get("lang", ""))

    def task_fixtures(self) -> dict[str, str]:
        """Role → fixture name for the fixtures this task is about.

        The benchmark's registered ``fixture_refs`` come first (``cab``, ``fxtr``,
        ``counter`` ...).  Then every Fixture the task class keeps as a plain attribute of
        the environment is ADDED under its attribute name (``blender``, ``toaster_oven``,
        ``cab2`` ...) — tasks that register some refs and keep the appliance as an
        attribute (PickPlaceCounterToBlender: ``counter`` registered, ``blender`` not)
        used to hide it.  An auxiliary fixture hanging off one of those (a blender's lid,
        a jar's lid, the vinegar bottle of an oil bottle) is listed as
        ``<attribute>_<part>`` when the fixture keeps it as an attribute, or
        ``<attribute>_auxiliary`` when RoboCasa only placed it as ``<name>_auxiliary_*``.
        """
        try:
            refs = self.episode_meta().get("fixture_refs") or {}
        except Exception:
            refs = {}
        out = {str(k): str(v) for k, v in refs.items()}
        self._ensure_built()
        from robocasa.models.fixtures import Fixture

        listed = set(out.values())
        for attr, value in vars(self.env).items():
            if attr.startswith(("_", "init_")) or not isinstance(value, Fixture):
                continue
            name = getattr(value, "name", None)
            if name not in self.env.fixtures:
                continue
            if name not in listed:
                out[attr] = str(name)
                listed.add(name)
            # auxiliary parts: Fixture attributes of the fixture itself ...
            for sub_attr, part in vars(value).items():
                pname = getattr(part, "name", None)
                if not isinstance(part, Fixture) or pname not in self.env.fixtures:
                    continue
                if pname in listed:
                    continue
                key = sub_attr[len(attr) + 1 :] if sub_attr.startswith(f"{attr}_") else sub_attr
                out[f"{attr}_{key}"] = str(pname)
                listed.add(pname)
            # ... or fixtures RoboCasa's scene builder named after it
            for other in self.env.fixtures:
                if other.startswith(f"{name}_auxiliary") and other not in listed:
                    out[f"{attr}_auxiliary"] = str(other)
                    listed.add(other)
        return out

    def fixture_parts(self, name: str) -> dict[str, Any]:
        """Every collision geom and every joint that belongs to fixture ``name``, from the
        compiled MuJoCo model (live poses).

        ``geoms``: name, ``type`` (box/cylinder/capsule/sphere/mesh), world ``center``,
        ``half_size`` along the geom's own axes and those ``axes`` as world rows, ``top_z``
        / ``bottom_z`` (the geom's world z extent) and ``body`` - buttons, knobs, levers,
        lids and handles are all here (a microwave's ``*start_button`` / ``*stop_button``
        geoms, a blender lid's ``*handle_main`` knob).  ``joints``: every HINGE or SLIDE
        joint whose name carries the fixture prefix, whether or not the fixture class
        lists it (a stove's ``knob_*`` joints, a blender's ``power_button``), with type,
        world axis/anchor, range, ``limited``, qpos and ``fraction`` — the SAME scorer's
        fraction ``get_fixture_joints`` reports (a drawer read 0.34 here against 1.04
        there before).  A button joint resting past its soft lower limit reads a
        NEGATIVE fraction (the blender's buttons: qpos -0.0006 on a [0, 0.0015] range ->
        -0.40); an unlimited joint reports ``range``/``fraction`` None.  A free-jointed
        part (a blender lid) lists NO joint — read its pose with ``get_object_pose``; a
        visual-only fixture lists no geoms.  Geoms of another fixture whose name merely
        extends this prefix (a blender's auxiliary lid) are left out - ask for that
        fixture by its own name.
        """
        fixture = self._fixture(name)
        prefix = f"{fixture.name}_"
        others = [
            f"{other}_"
            for other in self.env.fixtures
            if other != fixture.name and other.startswith(prefix)
        ]
        raw_m, raw_d = self._raw()
        gtypes = {
            0: "plane",
            2: "sphere",
            3: "capsule",
            4: "ellipsoid",
            5: "cylinder",
            6: "box",
            7: "mesh",
        }
        geoms = []
        for gid in range(int(raw_m.ngeom)):
            gname = self.sim.model.geom_id2name(gid) or ""
            if not gname.startswith(prefix) or any(gname.startswith(o) for o in others):
                continue
            if not (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid]):
                continue
            xmat = np.array(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
            gtype = int(raw_m.geom_type[gid])
            size = np.array(raw_m.geom_size[gid], dtype=float)
            aabb = np.array(raw_m.geom_aabb[gid], dtype=float).reshape(2, 3)
            if gtype == 7:  # mesh: the AABB in the geom frame
                half = aabb[1]
                centre = np.array(raw_d.geom_xpos[gid], dtype=float) + xmat @ aabb[0]
            else:
                if gtype == 3:  # capsule: the hemispherical caps add the radius
                    half = np.array([size[0], size[0], size[1] + size[0]])
                elif gtype == 5:
                    half = np.array([size[0], size[0], size[1]])
                elif gtype == 2:
                    half = np.array([size[0]] * 3)
                else:
                    half = size.copy()
                centre = np.array(raw_d.geom_xpos[gid], dtype=float)
            z_ext = float(np.sum(np.abs(xmat[2, :]) * half))
            geoms.append(
                {
                    "name": gname,
                    "type": gtypes.get(gtype, "other"),
                    "center": centre.round(4).tolist(),
                    "half_size": np.asarray(half).round(4).tolist(),
                    "axes": xmat.T.round(4).tolist(),
                    "top_z": round(float(centre[2] + z_ext), 4),
                    "bottom_z": round(float(centre[2] - z_ext), 4),
                    "body": self.sim.model.body_id2name(int(raw_m.geom_bodyid[gid])),
                }
            )
        joints = {}
        for jn in self.sim.model.joint_names:
            if not jn.startswith(prefix) or any(jn.startswith(o) for o in others):
                continue
            if int(raw_m.jnt_type[self.sim.model.joint_name2id(jn)]) not in (2, 3):
                continue  # a free joint (an auxiliary lid) has no scalar qpos
            geo = self._joint_geometry(jn)
            jid = self.sim.model.joint_name2id(jn)
            limited = bool(raw_m.jnt_limited[jid])
            frac: float | None = None
            if limited:
                try:
                    frac = float(self._joint_fraction(fixture, jn))
                except Exception:
                    lo, hi = geo["range"]
                    frac = (geo["qpos"] - lo) / max(hi - lo, 1e-9)
                    frac = float(1.0 - frac if lo < 0 else frac)
            joints[jn] = {
                "type": geo["type"],
                "axis": geo["axis"].round(4).tolist(),
                "anchor": geo["anchor"].round(4).tolist(),
                "range": [round(v, 4) for v in geo["range"]] if limited else None,
                "limited": limited,
                "qpos": round(geo["qpos"], 4),
                "fraction": round(frac, 4) if frac is not None else None,
                "body": self.sim.model.body_id2name(geo["body"]),
            }
        return {"geoms": geoms, "joints": joints}

    def task_completed(self) -> bool:
        """The benchmark's own success check."""
        return bool(self.env._check_success())

    # ------------------------------------------------------------ observation

    def list_objects(self) -> list[str]:
        """Names of the task's movable objects."""
        self._ensure_built()
        return sorted(self.env.objects)

    def list_fixtures(self) -> list[str]:
        """Names of the kitchen fixtures (counters, cabinets, appliances ...)."""
        self._ensure_built()
        return sorted(self.env.fixtures)

    def get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose ``(position, quaternion_wxyz)`` of an object or fixture."""
        body_id = self.env.obj_body_id.get(name)
        if body_id is None:
            fixture = self.env.fixtures.get(name)
            if fixture is None:
                raise KeyError(f"unknown object or fixture {name!r}")
            root = getattr(fixture, "root_body", None)
            if root and root in self.sim.model.body_names:
                body_id = self.sim.model.body_name2id(root)
            else:  # static fixture: placement pose from the model
                pos = np.asarray(getattr(fixture, "pos", (0.0, 0.0, 0.0)), dtype=float)
                quat = np.asarray(getattr(fixture, "quat", (1.0, 0.0, 0.0, 0.0)), dtype=float)
                return pos.copy(), quat.copy()
        pos = np.array(self.sim.data.body_xpos[body_id], dtype=float)
        quat = np.array(self.sim.data.body_xquat[body_id], dtype=float)  # MuJoCo: wxyz
        return pos, quat

    def fixture_extent(self, name: str) -> dict[str, Any]:
        """Footprint of a fixture: world ``pos``, ``yaw``, ``width/depth/height`` and its
        8 bounding-box corners (world), so an agent can pick a standing spot beside it.
        """
        fixture = self.env.fixtures.get(name)
        if fixture is None:
            raise KeyError(f"unknown fixture {name!r}")
        try:
            bbox = [np.asarray(pt, dtype=float).tolist() for pt in fixture.get_bbox_points()]
        except Exception:
            bbox = []
        return {
            "pos": np.asarray(fixture.pos, dtype=float).tolist(),
            "yaw": float(getattr(fixture, "rot", 0.0) or 0.0),
            "width": fixture.width,
            "depth": fixture.depth,
            "height": fixture.height,
            "bbox": bbox,
        }

    def base_placement_for(self, fixture: str, ref_object: str | None = None) -> np.ndarray:
        """RoboCasa's own standing pose ``[x, y, yaw]`` in front of ``fixture`` (the pose
        it would spawn the robot at for a task about that fixture / object).
        """
        from robocasa.utils import env_utils

        fx = self.env.fixtures.get(fixture)
        if fx is None:
            raise KeyError(f"unknown fixture {fixture!r}")
        pos, ori = env_utils.compute_robot_base_placement_pose(
            self.env, ref_fixture=fx, ref_object=ref_object
        )
        ori = np.atleast_1d(np.asarray(ori, dtype=float))
        yaw = float(ori[-1])
        return np.array([float(pos[0]), float(pos[1]), yaw])

    def drive_to_fixture(self, fixture: str, ref_object: str | None = None) -> dict[str, Any]:
        """Drive to RoboCasa's standing pose for ``fixture`` (see :meth:`drive_to`)."""
        x, y, yaw = self.base_placement_for(fixture, ref_object)
        tuck = self.tuck_arm()
        result = self.drive_to(x, y, yaw)
        result["target"] = [x, y, yaw]
        result["tuck"] = {k: tuck.get(k) for k in ("reached", "reason", "error_m") if k in tuck}
        return result

    def object_extent(self, name: str) -> dict[str, Any]:
        """Size of an object along its own axes and where those axes point in the world."""
        import robosuite.utils.transform_utils as tu

        obj = self.env.objects.get(name)
        if obj is None:
            raise KeyError(f"unknown object {name!r}")
        pos, quat = self.get_object_pose(name)
        axes = tu.quat2mat(_xyzw(quat)).T  # rows: local x, y, z in world coordinates
        return {
            "size": np.asarray(obj.size, dtype=float).tolist(),
            "axes_world": np.round(axes, 4).tolist(),
            "bottom_z": float(pos[2] + obj.bottom_offset[2]),
            "top_z": float(pos[2] + obj.top_offset[2]),
        }

    def _nearest_finger_axis(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The parallel gripper is symmetric under a half-turn about its tool axis, so the
        finger axis ``x`` and ``-x`` describe the same grasp.  Return whichever is closer to
        the grip site's current x axis.

        This is a heuristic, not a guarantee: it removes the common stall (measured
        2026-08-27: from the reset pose the far half-turn stalled the OSC servo at 0.81 rad
        of orientation error, the near one reached), but "closest to the current finger
        axis" is not "inside joint 7's ±2.9 rad range" — after a wrist already parked near
        2.3 rad the near side can still hit the limit (measured 2026-08-27, review probe).
        Compute the quaternion right before the ``goto_pose`` that uses it (a base turn
        changes the answer) and read the orientation residual afterwards.
        """
        sid = self.sim.model.site_name2id(GRIP_SITE)
        cur_x = np.asarray(self.sim.data.site_xmat[sid], dtype=float).reshape(3, 3)[:, 0]
        return -x if cur_x @ x < 0.0 else x

    def horizontal_grasp_quat(
        self, point_dir: Any, close_dir: Any = (0.0, 0.0, 1.0), nearest: bool = True
    ) -> np.ndarray:
        """Grip-site orientation (wxyz) with the tool axis along the horizontal ``point_dir``
        (e.g. at a door) and the fingers closing along ``close_dir`` (default vertical).
        ``nearest`` flips the (symmetric) finger axis to the side the wrist reaches faster.
        """
        z = np.asarray(point_dir, dtype=float)
        z = z / max(np.linalg.norm(z), 1e-9)
        x = np.asarray(close_dir, dtype=float)
        x = x - z * (x @ z)
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("close_dir must not be parallel to point_dir")
        x = x / np.linalg.norm(x)
        if nearest:
            x = self._nearest_finger_axis(x, z)
        y = np.cross(z, x)
        return _wxyz_from_mat(np.column_stack([x, y, z]))

    def place_gently(
        self,
        name: str,
        max_descent: float = 0.3,
        step: float = 0.015,
        target_xy: Any = None,
        correction_per_step_m: float = 0.008,
    ) -> dict[str, Any]:
        """Lower the HELD object until it stops following the hand (it rests on something)
        or the hand cannot descend, then open the gripper and retreat 6 cm upward.

        ``held`` is checked FIRST, and it is the report that matters: with empty jaws the
        object never follows the hand, so "it stopped following" is true on the very first
        step and the old report read ``rested: True, descent_m: 0.015`` for a placement
        that never happened (measured across 20 shape-D runs: ``descent_m`` 0.001-0.019 in
        every failure).  Empty jaws now return at once with ``held: False``.

        ``target_xy`` keeps the descent honest: the object hangs 1-2 cm off the grip site,
        so each step also slides the hand by up to ``correction_per_step_m`` toward putting
        the OBJECT on that spot, while the hand is still descending freely.
        """
        if self.get_gripper_opening() <= GRIPPER_EMPTY_OPENING:
            # same ending as a real placement (release, then 6 cm up) minus the descent
            # that was never going to move anything
            self.set_gripper(closed=False, steps=12)
            with contextlib.suppress(RuntimeError):
                tip = self.get_eef_pose()
                self.goto_pose(
                    tip[:3] + np.array([0, 0, 0.06]), tip[3:7], tolerance_m=0.02, max_steps=80
                )
            return {
                "rested": False,
                "held": False,
                "descent_m": 0.0,
                "object_z": float(self.get_object_pose(name)[0][2]),
                "reason": "the jaws are empty (opening <= 0.016 m): nothing is being placed",
            }
        goal_xy = None if target_xy is None else np.asarray(target_xy, dtype=float)[:2]
        descended = 0.0
        rested = False
        blocked = 0
        while descended < max_descent:
            tip0 = self.get_eef_pose()
            obj0 = self.get_object_pose(name)[0]
            slide = np.zeros(2)
            if goal_xy is not None and blocked == 0:
                residual = goal_xy - obj0[:2]
                norm = float(np.linalg.norm(residual))
                if norm > 1e-4:
                    slide = residual * min(1.0, float(correction_per_step_m) / norm)
            try:
                self.goto_pose(
                    tip0[:3] + np.array([slide[0], slide[1], -step]),
                    tip0[3:7],
                    tolerance_m=0.006,
                    max_steps=60,
                )
            except RuntimeError:
                blocked += 1
                if blocked >= 2:
                    rested = True
                    break
                continue
            tip1 = self.get_eef_pose()
            obj1 = self.get_object_pose(name)[0]
            hand_dz = float(tip0[2] - tip1[2])
            obj_dz = float(obj0[2] - obj1[2])
            descended += max(hand_dz, 0.0)
            if hand_dz < 0.3 * step:
                blocked += 1
                if blocked >= 2:
                    rested = True
                    break
                continue
            blocked = 0
            if obj_dz < 0.4 * hand_dz:
                rested = True
                break
        self.set_gripper(closed=False, steps=12)
        with contextlib.suppress(RuntimeError):
            tip = self.get_eef_pose()
            self.goto_pose(
                tip[:3] + np.array([0, 0, 0.06]), tip[3:7], tolerance_m=0.02, max_steps=80
            )
        obj = self.get_object_pose(name)[0]
        return {
            "rested": rested,
            "held": True,
            "descent_m": round(descended, 3),
            "object_z": float(obj[2]),
            "xy_error_m": None
            if goal_xy is None
            else round(float(np.linalg.norm(obj[:2] - goal_xy)), 4),
        }

    # ------------------------------------------------------------ pick and place

    JAW_SPAN_M: float = 0.082  # widest object the pads will still close on material

    def grasp_axis(self, name: str, tilt_tol: float = 0.7) -> dict[str, Any]:
        """Which of an object's own axes the fingers should close across, and how wide it is.

        The parallel jaws span 0.085 m, so the pinch has to be across the object's NARROWEST
        roughly-horizontal axis.  Returns ``{"yaw", "width_m", "height_m", "bottom_z",
        "top_z", "fits_jaw", "axis_index"}``; ``yaw`` is the world heading of that axis and
        is what :meth:`top_down_grasp_quat` wants.
        """
        ext = self.object_extent(name)
        cands = []
        for i in range(3):
            ax = np.asarray(ext["axes_world"][i], dtype=float)
            if float(np.linalg.norm(ax[:2])) > tilt_tol:
                cands.append((float(ext["size"][i]), float(np.arctan2(ax[1], ax[0])), i))
        if not cands:
            cands = [(float(min(ext["size"])), 0.0, int(np.argmin(ext["size"])))]
        width, yaw, idx = min(cands)
        bottom, top = float(ext["bottom_z"]), float(ext["top_z"])
        return {
            "yaw": yaw,
            "width_m": round(width, 4),
            "height_m": round(top - bottom, 4),
            "bottom_z": bottom,
            "top_z": top,
            "fits_jaw": bool(width <= self.JAW_SPAN_M),
            "axis_index": idx,
        }

    def _stage_over(
        self, xy: np.ndarray, z: float, quat: np.ndarray | None, tolerance_m: float
    ) -> dict[str, Any]:
        """Position-only hover over ``xy`` at ``z``, then (optionally) rotate in place.

        Measured 2026-08-28: an ORIENTED goto into a cavity or at the edge of the envelope
        stalls at 0.14-0.30 m and 0.44-0.65 rad and poisons the next call, while the
        position-only request to the same point reaches it.  So the position and the
        rotation are two separate requests, and the rotation is allowed to fail.
        """
        rep: dict[str, Any] = {"positioned": False, "oriented": False}
        with contextlib.suppress(RuntimeError):
            r = self.goto_pose([xy[0], xy[1], z], None, tolerance_m=tolerance_m, max_steps=200)
            rep["positioned"] = bool(r.get("reached"))
        rep["hover_error_m"] = round(
            float(np.linalg.norm(self.get_eef_pose()[:3] - np.array([xy[0], xy[1], z]))), 4
        )
        if quat is None or not rep["positioned"]:
            # a hover the arm could not make means a cavity roof or the edge of the
            # envelope; an oriented request from there stalls AND poisons the next call
            return rep
        for q in (quat, self._rotate_quat(quat, np.array([0.0, 0.0, 1.0]), np.pi)):
            with contextlib.suppress(RuntimeError):
                r = self.goto_pose([xy[0], xy[1], z], q, tolerance_m=tolerance_m, max_steps=140)
                if r.get("reached") or float(r.get("error_rad", 1.0)) < 0.12:
                    rep["oriented"] = True
                    rep["quat"] = np.asarray(q, dtype=float).tolist()
                    return rep
            # a failed oriented goto leaves the arm in a posture that misses the next
            # request by 3-13 cm: re-establish the position-only pose before trying again
            with contextlib.suppress(RuntimeError):
                self.goto_pose([xy[0], xy[1], z], None, tolerance_m=tolerance_m, max_steps=120)
        return rep

    def pick_object(
        self,
        name: str,
        *,
        grip_frac: float = 0.0,
        below_m: float = 0.02,
        approach_m: float = 0.12,
        lift_m: float = 0.15,
        attempts: int = 2,
        yaw: float | None = None,
        tolerance_m: float = 0.02,
        oriented: bool = True,
    ) -> dict[str, Any]:
        """Take a top-down pinch on ``name`` and lift it, verified against MuJoCo.

        **The descent is a floor-seeking request, not a position request.**  Measured over
        six shape-D tasks (2026-08-28): a `goto_pose` descent onto a counter object stalls
        1.4-1.6 cm above whatever it was asked for and bottoms out with the grip site
        2.1-3.1 cm above the SUPPORT, whatever it was aimed at.  So aiming at the object's
        upper third (``top_z - 0.3 h``, which the family program used) lands the pads
        entirely above a 5.6-7.2 cm object and the jaws close on air -- 0.008-0.010 m in
        five of six tasks.  Asking for ``bottom_z - below_m`` instead lets the contact
        decide the height, the site lands in the object's lower half, and the same six
        grasps read 0.062-0.075 m and lift the object 0.124-0.129 m.

        Each attempt is checked -- the pads must stay open on material AND the object must
        rise with the hand -- and a failed one restages and descends ``below_m`` deeper.
        Returns ``{"grasped", "opening_m", "rise_m", "attempts", "width_m", "fits_jaw",
        "site_above_bottom_m", "yaw", "oriented", "budget_left"}``.
        """
        plan = self.grasp_axis(name)
        use_yaw = plan["yaw"] if yaw is None else float(yaw)
        rep: dict[str, Any] = {
            "grasped": False,
            "width_m": plan["width_m"],
            "height_m": plan["height_m"],
            "fits_jaw": plan["fits_jaw"],
            "yaw": round(use_yaw, 4),
            "attempts": 0,
        }
        need = max(0.018, 0.30 * float(plan["width_m"]))
        for attempt in range(int(attempts)):
            if attempt and self._budget_left() < 220:
                rep["reason"] = "not enough frames left for another attempt"
                break
            rep["attempts"] = attempt + 1
            ext = self.object_extent(name)
            bottom, top = float(ext["bottom_z"]), float(ext["top_z"])
            before = self.get_object_pose(name)[0]
            self.set_gripper(closed=False, steps=10)
            quat = self.top_down_grasp_quat(use_yaw) if oriented else None
            # The staging height is adaptive: over an open counter 12 cm of headroom is
            # free, but inside a microwave or under a shelf the same request is ABOVE the
            # cavity roof and the arm never makes it (measured: hover_error 0.13 m, and
            # everything after it missed).  Drop to whatever headroom there is.
            for head in (float(approach_m), 0.06, 0.03):
                stage = self._stage_over(before[:2], top + head, quat, tolerance_m)
                stage["approach_m"] = head
                if stage["positioned"]:
                    break
                if self._budget_left() < 200:
                    break
            rep["stage"] = stage
            if not stage["positioned"] and attempt:
                # a second attempt from a hover the arm cannot make costs 300-400 frames
                # and repeats the same miss: the limit is the envelope, not the depth
                rep["reason"] = "no staging pose over the object: the hover is out of reach"
                rep["budget_left"] = self._budget_left()
                return rep
            here = self.get_object_pose(name)[0]  # the hover may have nudged it
            drop = float(below_m) * (1.0 + attempt)
            z_req = bottom + float(grip_frac) * (top - bottom) - drop
            with contextlib.suppress(RuntimeError):
                self.goto_pose(
                    [here[0], here[1], z_req],
                    stage.get("quat") if stage["oriented"] else None,
                    tolerance_m=max(tolerance_m, 0.015),
                    max_steps=160,
                )
            site = self.get_eef_pose()
            rep["descent_request_z"] = round(float(z_req), 4)
            rep["site_above_bottom_m"] = round(float(site[2] - bottom), 4)
            rep["site_xy_error_m"] = round(float(np.linalg.norm(site[:2] - here[:2])), 4)
            opening = self.set_gripper(closed=True, steps=18)
            rep["opening_m"] = round(float(opening), 4)
            tip0 = self.get_eef_pose()
            with contextlib.suppress(RuntimeError):
                self.goto_pose(
                    tip0[:3] + np.array([0.0, 0.0, float(lift_m)]),
                    None,
                    tolerance_m=0.03,
                    max_steps=110,
                )
            rise = float(self.get_object_pose(name)[0][2] - before[2])
            # The test is that the object rose WITH THE HAND, and the hand does not always
            # get to rise: inside a microwave the roof stops it after 3-4 cm.  Judging the
            # object's rise against the 15 cm ASKED for called a real grasp a failure
            # (measured 2026-08-28: PickPlaceMicrowaveToCounter, opening 0.065, object up
            # 0.027 with the hand up 0.034 -- and the second attempt it then spent could
            # not restage inside the cavity, which is 340 frames for nothing).
            hand_rise = float(self.get_eef_pose()[2] - tip0[2])
            rep["hand_rise_m"] = round(hand_rise, 4)
            rep["rise_m"] = round(rise, 4)
            rep["oriented"] = bool(stage["oriented"])
            rep["budget_left"] = self._budget_left()
            if opening > need and rise > max(0.012, 0.55 * hand_rise):
                rep["grasped"] = True
                return rep
            if attempt + 1 < int(attempts):
                self.set_gripper(closed=False, steps=10)
                self.hold(5)
        return rep

    def place_object(
        self,
        name: str,
        x: float | None = None,
        y: float | None = None,
        *,
        surface_z: float | None = None,
        clearance_m: float = 0.05,
        travel_dz: float = 0.10,
        gentle: bool = True,
        settle: int = 20,
        max_carry_error: float = 0.06,
        max_descent: float = 0.25,
        force: bool = False,
    ) -> dict[str, Any]:
        """Put the HELD object down at ``(x, y)`` (its own centre, not the tool's) and let go.

        Two corrections the family program did not make.  **(a) Aim the object.**  It hangs
        1-2 cm off the grip site, wherever the pads caught it, so the tool goes to
        ``target - (object_xy - tip_xy)``.  **(b) Know whether anything is held**: a pinch
        that closed on air makes ``place_gently``'s "the object stopped following the hand"
        test true on its very first step, which is exactly the ``descent_m 0.001-0.019``
        the campaign recorded -- a report of a placement that never happened.

        With ``surface_z`` the release is the cheap one the campaign priced at ~27 frames
        (open from ``surface_z + height + clearance_m`` and wait); without it the descent is
        :meth:`place_gently`'s contact search.  Returns ``{"placed", "held", "xy_error_m",
        "object_z", "descent_m", "rested"}``.
        """
        held = self.get_gripper_opening() > GRIPPER_EMPTY_OPENING
        obj = self.get_object_pose(name)[0]
        rep: dict[str, Any] = {"placed": False, "held": bool(held)}
        if not held:
            self.set_gripper(closed=False, steps=10)
            rep["reason"] = "the jaws are empty: nothing was carried here"
            rep["object_z"] = float(obj[2])
            return rep
        tip = self.get_eef_pose()
        offset = obj[:2] - tip[:3][:2]  # where the object hangs relative to the grip site
        goal_xy = obj[:2] if x is None or y is None else np.array([float(x), float(y)])
        rep["carry_offset_m"] = [round(float(v), 4) for v in offset]
        ext = self.object_extent(name)
        height = float(ext["top_z"]) - float(ext["bottom_z"])
        tool_xy = goal_xy - offset
        # Rise only as far as the destination needs, and NEVER descend during the carry: a
        # travel height clipped down toward the destination drags the object through
        # whatever it is carried over (measured 2026-08-28: carrying a tomato at 0.94 m
        # toward a sink basin scraped it out of the jaws on the counter rim, and the
        # "placement" then arrived with empty jaws).
        travel_z = float(tip[2]) + float(travel_dz)
        if surface_z is not None:
            travel_z = max(float(tip[2]), float(surface_z) + height + float(clearance_m) + 0.04)
        with contextlib.suppress(RuntimeError):
            self.goto_pose([tip[0], tip[1], travel_z], None, tolerance_m=0.03, max_steps=120)
        with contextlib.suppress(RuntimeError):
            self.goto_pose([tool_xy[0], tool_xy[1], travel_z], None, tolerance_m=0.04,
                           max_steps=180)
        hold_z = None
        if surface_z is not None:
            hold_z = float(surface_z) + height + float(clearance_m)
            with contextlib.suppress(RuntimeError):
                self.goto_pose([tool_xy[0], tool_xy[1], hold_z], None, tolerance_m=0.025,
                               max_steps=140)
        now = self.get_eef_pose()
        rep["carry_error_m"] = round(float(np.linalg.norm(now[:2] - tool_xy)), 4)
        rep["release_z"] = round(float(now[2]), 4)
        if hold_z is not None:
            rep["descend_error_m"] = round(float(now[2]) - hold_z, 4)
        # A release the carry never reached is a DROP, and it is how an object ends up on
        # the floor beside the destination.  Say so and keep holding: the caller can lift
        # the torso, drive closer, or re-aim and call again.
        too_high = hold_z is not None and float(now[2]) - hold_z > 0.12
        if (rep["carry_error_m"] > float(max_carry_error) or too_high) and not force:
            rep["reason"] = (
                f"the carry stopped {rep['carry_error_m']:.3f} m short in xy and "
                f"{rep.get('descend_error_m', 0.0):.3f} m high (out of the envelope): "
                "still holding, nothing released"
            )
            rep["budget_left"] = self._budget_left()
            return rep
        if gentle:
            # bound the contact search: place_gently walks 1.5 cm at a time and each step
            # is its own goto, so an unbounded descent from a carry height costs 300-800
            # frames.  The hold height is already one clearance above the surface.
            span = float(clearance_m) + 0.05 if surface_z is not None else float(max_descent)
            rep.update(
                self.place_gently(name, max_descent=span, step=0.02, target_xy=goal_xy)
            )
        else:
            self.set_gripper(closed=False, steps=12)
            self.hold(int(settle))
            rep["descent_m"] = 0.0
            rep["rested"] = True
        obj = self.get_object_pose(name)[0]
        rep["object_z"] = float(obj[2])
        rep["xy_error_m"] = round(float(np.linalg.norm(obj[:2] - goal_xy)), 4)
        rep["placed"] = True
        rep["held"] = True
        rep["budget_left"] = self._budget_left()
        return rep

    def retreat_from(self, name: str, distance_m: float = 0.28, hops: int = 3) -> dict[str, Any]:
        """Back the hand away from ``name`` until the grip site clears ``distance_m``.

        Every shape-D scorer ends in ``gripper_obj_far`` (0.25 m; 0.15 m for the fridge
        tasks).  Retreat HORIZONTALLY first, toward the base -- a vertical goto the arm
        cannot make oscillates the hand and sweeps the object off what it was just put on.
        """
        for hop in range(int(hops)):
            obj = self.get_object_pose(name)[0]
            tip = self.get_eef_pose()
            gap = float(np.linalg.norm(tip[:3] - obj))
            if gap > float(distance_m):
                return {"clear": True, "gap_m": round(gap, 4), "hops": hop}
            base = self.get_base_pose()
            back = base[:2] - tip[:2]
            norm = float(np.linalg.norm(back))
            back = back / norm * 0.14 if norm > 1e-6 else np.array([0.0, -0.14])
            with contextlib.suppress(RuntimeError):
                self.goto_pose(
                    tip[:3] + np.array([back[0], back[1], 0.06]),
                    None,
                    tolerance_m=0.04,
                    max_steps=90,
                )
        obj = self.get_object_pose(name)[0]
        gap = float(np.linalg.norm(self.get_eef_pose()[:3] - obj))
        return {"clear": gap > float(distance_m), "gap_m": round(gap, 4), "hops": int(hops)}

    # ------------------------------------------------------------ base placement

    # Collision footprint of the Omron base relative to ``mobilebase0_base`` in its heading
    # frame: ``mobilebase0_pedestal_feet_col`` is a box of half-size 0.35 x 0.25 whose body
    # (``wheeled_base``) sits 0.20 m BEHIND the base body (robosuite
    # omron_mobile_base.xml: ``pos="-0.20 0 0.192"``; re-measured 2026-08-27 in sim: pedestal
    # AABB centre at [-0.20, 0.00] of the base yaw frame while a forward command moves the
    # base along +x).  So the base extends 0.15 m ahead of its origin and 0.55 m behind it.
    # (Round 1 had the sign mirrored, which flagged obstacle-free spawn poses as blocked and
    # returned no standing spot at all in the microwave and wall-cabinet scenes.)
    BASE_FOOTPRINT: tuple[tuple[float, float], tuple[float, float]] = ((-0.55, 0.15), (-0.25, 0.25))
    SHOULDER_Z: float = 1.033
    ARM_REACH_M: float = 0.86  # radius from the shoulder the servo can still reach (Panda 0.855)
    # Finger geometry along the tool axis, from the grip site (panda_omron.yaml ``gripper``):
    # the pads span -1.2 .. +0.4 cm, the fingertips end 1.1 cm beyond the site.
    PAD_BEHIND_SITE_M: float = 0.012
    FINGERTIP_BEYOND_SITE_M: float = 0.011

    def _raw(self) -> tuple[Any, Any]:
        return (
            getattr(self.sim.model, "_model", self.sim.model),
            getattr(self.sim.data, "_data", self.sim.data),
        )

    def _fixture(self, name: str) -> Any:
        self._ensure_built()
        fixture = self.env.fixtures.get(name)
        if fixture is None:
            raise KeyError(f"unknown fixture {name!r}")
        return fixture

    def _floor_obstacles(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame 2-D AABBs ``(lo_xy, hi_xy)`` of every collision geom the base can bump
        into at its height: fixtures, walls, open doors and drawers as they stand now —
        everything but the robot, the floor and the movable objects.
        """
        raw_m, raw_d = self._raw()
        n = int(raw_m.ngeom)
        names = [self.sim.model.geom_id2name(g) or "" for g in range(n)]
        robot = np.array([nm.startswith(("robot0", "mobilebase0", "gripper0")) for nm in names])
        obj_bodies = set(self.env.obj_body_id.values())
        is_obj = np.array([int(b) in obj_bodies for b in raw_m.geom_bodyid])
        coll = (np.asarray(raw_m.geom_contype) | np.asarray(raw_m.geom_conaffinity)) != 0
        aabb = np.asarray(raw_m.geom_aabb, dtype=float).reshape(-1, 2, 3)
        xmat = np.asarray(raw_d.geom_xmat, dtype=float).reshape(-1, 3, 3)
        centres = np.asarray(raw_d.geom_xpos, dtype=float) + np.einsum(
            "nij,nj->ni", xmat, aabb[:, 0]
        )
        halves = np.einsum("nij,nj->ni", np.abs(xmat), aabb[:, 1])
        lo, hi = centres - halves, centres + halves
        keep = coll & ~robot & ~is_obj & (lo[:, 2] < 0.45) & (hi[:, 2] > 0.05)
        return lo[keep, :2], hi[keep, :2]

    def _clearance(self, margin: float = 0.03) -> Any:
        """A ``clear(x, y, yaw)`` closure over ONE reading of the floor obstacles.

        :meth:`base_pose_clear` re-reads and re-names every geom in the model on each call
        (that is how it drops the robot's own), which a route plan asks for thousands of
        times.  Nothing moves while a plan is computed, so read the obstacles once.
        """
        (x0, x1), (y0, y1) = self.BASE_FOOTPRINT
        ox, oy = (x0 + x1) / 2, (y0 + y1) / 2
        hu, hv = (x1 - x0) / 2 + margin, (y1 - y0) / 2 + margin
        lo, hi = self._floor_obstacles()
        if lo.shape[0] == 0:
            return lambda x, y, yaw: True
        box_c = (lo + hi) / 2
        box_h = (hi - lo) / 2

        def clear(x: float, y: float, yaw: float) -> bool:
            c, s = np.cos(yaw), np.sin(yaw)
            u, v = np.array([c, s]), np.array([-s, c])
            centre = np.array([x, y]) + u * ox + v * oy
            d = box_c - centre
            # world axes
            base_ext = np.array(
                [abs(u[0]) * hu + abs(v[0]) * hv, abs(u[1]) * hu + abs(v[1]) * hv]
            )
            sep = np.abs(d) > base_ext + box_h
            # footprint axes
            for axis, half in ((u, hu), (v, hv)):
                proj = np.abs(d @ axis)
                box_ext = box_h @ np.abs(axis)
                sep |= (proj > half + box_ext)[:, None]
            return bool(np.all(sep.any(axis=1)))

        return clear

    def base_pose_clear(self, x: float, y: float, yaw: float, margin: float = 0.03) -> bool:
        """True when the base footprint at ``(x, y, yaw)`` overlaps no obstacle (2-D
        separating-axis test of the oriented footprint against every obstacle box).
        """
        return bool(self._clearance(margin)(float(x), float(y), float(yaw)))

    def find_standing_pose(
        self,
        target: Any,
        approach_dir: Any | None = None,
        *,
        standoffs: tuple[float, ...] = (0.7, 0.6, 0.8, 0.5),
        yaw_offsets: tuple[float, ...] = (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2),
        max_candidates: int = 6,
        prefer_m: float = 0.72,
    ) -> list[dict[str, Any]]:
        """Base poses from which the arm can reach the world point ``target``, best first.

        Candidates stand ``standoff`` metres from the target's xy along directions fanned
        around ``approach_dir`` (the direction from the target toward free floor, e.g. a
        door's outward normal; default: toward the current base position), facing the
        target.  Poses whose footprint overlaps a fixture footprint are dropped; the rest are
        ranked by the target's distance from the shoulder against ``prefer_m`` (0.72 m: a
        drawer handle 0.6 m ahead stalled, 0.73 m was grasped; a door pushed along an arc
        wants the shoulder CLOSER, ~0.55 m, so that consecutive strokes stay in reach —
        ``push_door_shut`` passes that), then by driving distance.  The shoulder height is
        read LIVE (a torso lift raises it up to 0.34 m).  Returns ``[{"pose": [x, y, yaw],
        "standoff", "shoulder_dist", "drive_m"}, ...]``.
        """
        tgt = np.asarray(target, dtype=float)
        base = self.get_base_pose()
        if approach_dir is None:
            approach_dir = np.array([base[0] - tgt[0], base[1] - tgt[1], 0.0])
        d = np.asarray(approach_dir, dtype=float)[:2]
        if np.linalg.norm(d) < 1e-6:
            # a vertical approach (a downward press) says nothing about the floor: fan the
            # candidates around the direction toward the base, as with no approach_dir
            d = np.array([base[0] - tgt[0], base[1] - tgt[1]])
            if np.linalg.norm(d) < 1e-6:
                d = np.array([1.0, 0.0])
        phi0 = float(np.arctan2(d[1], d[0]))
        shoulder_z = float(self._shoulder()[2])  # LIVE: the torso lift moves it up to 0.34 m
        found: list[dict[str, Any]] = []
        for standoff in standoffs:
            for off in yaw_offsets:
                phi = phi0 + off
                x = float(tgt[0] + standoff * np.cos(phi))
                y = float(tgt[1] + standoff * np.sin(phi))
                yaw = float(np.arctan2(np.sin(phi + np.pi), np.cos(phi + np.pi)))
                if not self.base_pose_clear(x, y, yaw):
                    continue
                dz = float(tgt[2] - shoulder_z) if tgt.shape[0] > 2 else 0.0
                shoulder_dist = float(np.hypot(standoff, dz))
                if shoulder_dist > self.ARM_REACH_M:
                    continue
                drive = float(np.hypot(x - base[0], y - base[1]))
                # measured 2026-08-27: a drawer handle 0.6 m ahead of the shoulder stalled
                # 2.4 cm short (the wrist folds under the counter edge), 0.73 m ahead was
                # grasped — so prefer ~0.7 m standoffs
                score = abs(shoulder_dist - float(prefer_m)) + 0.25 * abs(off) + 0.15 * drive
                found.append(
                    {
                        "pose": [round(x, 3), round(y, 3), round(yaw, 3)],
                        "standoff": standoff,
                        "shoulder_dist": round(shoulder_dist, 3),
                        "drive_m": round(drive, 3),
                        "score": round(score, 3),
                    }
                )
        found.sort(key=lambda c: c["score"])
        return found[:max_candidates]

    # --------------------------------------------- pricing a standing pose before driving

    BASE_JOINTS: tuple[str, ...] = (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    )

    def _base_frame(self) -> dict[str, Any]:
        """The map between a world base pose ``[x, y, yaw]`` and the base's own three
        joints, calibrated by finite differences on the scratch data (zero steps).

        Measured 2026-08-28: the two slides are unit world-axis translations (``forward``
        is world +y, ``side`` is world -x on this model) and they do **not** turn with the
        yaw hinge, because they are its parents in the chain.  The yaw hinge does move the
        base body's origin (it sits 6 cm off the hinge axis), so the yaw is set first and
        the slides are solved against where the body lands afterwards.
        """
        import mujoco

        raw_m, raw_d = self._raw()
        m = self.sim.model
        adr = [int(raw_m.jnt_qposadr[m.joint_name2id(j)]) for j in self.BASE_JOINTS]
        bid = m.body_name2id(BASE_BODY)
        sc = self._scratch()

        def world(q: Any) -> np.ndarray:
            sc.qpos[:] = raw_d.qpos
            for a, v in zip(adr, np.asarray(q, dtype=float), strict=False):
                sc.qpos[a] = float(v)
            mujoco.mj_kinematics(raw_m, sc)
            pos = np.asarray(sc.xpos[bid], dtype=float)
            mat = np.asarray(sc.xmat[bid], dtype=float).reshape(3, 3)
            return np.array([pos[0], pos[1], float(np.arctan2(mat[1, 0], mat[0, 0]))])

        q_ref = np.array([float(raw_d.qpos[a]) for a in adr])
        p_ref = world(q_ref)
        cache = getattr(self, "_base_axes_cache", None)
        if cache is None or cache[0] is not raw_m:
            a_f = (world(q_ref + np.array([1.0, 0.0, 0.0])) - p_ref)[:2]
            a_s = (world(q_ref + np.array([0.0, 1.0, 0.0])) - p_ref)[:2]
            cache = (raw_m, np.column_stack([a_f, a_s]))
            self._base_axes_cache = cache
        return {"adr": adr, "q_ref": q_ref, "p_ref": p_ref, "axes": cache[1], "world": world}

    def hypothetical_qpos(
        self, base_pose: Any | None = None, torso: float | None = None
    ) -> np.ndarray:
        """A full model ``qpos`` for a world where the base stands at ``base_pose``
        ``[x, y, yaw]`` and the torso lift is at ``torso`` metres — everything else exactly
        as it is now.  **Nothing is stepped and nothing moves**: this is the input to a free
        reachability question (``solve_arm_ik(..., qpos=...)``), which is how a standing
        pose is priced before the base is driven to it.
        """
        raw_m, raw_d = self._raw()
        q = np.array(raw_d.qpos, dtype=float)
        if base_pose is not None:
            fr = self._base_frame()
            x, y, yaw = (float(v) for v in np.asarray(base_pose, dtype=float).reshape(-1)[:3])
            q_yaw = float(fr["q_ref"][2] + (yaw - fr["p_ref"][2]))
            after = fr["world"](np.array([fr["q_ref"][0], fr["q_ref"][1], q_yaw]))
            dq = np.linalg.solve(fr["axes"], np.array([x, y]) - after[:2])
            q[fr["adr"][0]] = float(fr["q_ref"][0] + dq[0])
            q[fr["adr"][1]] = float(fr["q_ref"][1] + dq[1])
            q[fr["adr"][2]] = q_yaw
        if torso is not None and self._has_torso():
            m = self.sim.model
            adr = int(raw_m.jnt_qposadr[m.joint_name2id(self.TORSO_JOINT)])
            q[adr] = float(np.clip(float(torso), *self.TORSO_RANGE_M))
        return q

    def posture_collision(
        self, q_arm: Any, *, qpos: Any | None = None, ignore: tuple[str, ...] = ()
    ) -> str:
        """Name of the first robot/world geom pair the arm posture ``q_arm`` puts in contact
        (kinematics + MuJoCo collision on the scratch data, zero steps), or ``""``.

        ``qpos`` supplies a hypothetical world (see :meth:`hypothetical_qpos`).  Self
        contacts, the floor and any geom whose name contains one of ``ignore`` (the object
        being grasped: the pads are MEANT to touch it) do not count.  This is the check that
        makes a free reachability answer honest -- measured 2026-08-28, IK alone reported
        the grip site reachable from a base 0.36 m from the object while the only posture
        that reached it folded the forearm through the counter, and the real descent then
        stalled 0.27 m away.
        """
        import mujoco

        raw_m, raw_d = self._raw()
        sc = self._scratch()
        sc.qpos[:] = np.asarray(raw_d.qpos if qpos is None else qpos, dtype=float)
        for jn, v in zip(self.ARM_JOINTS, np.asarray(q_arm, dtype=float), strict=False):
            sc.qpos[int(raw_m.jnt_qposadr[self.sim.model.joint_name2id(jn)])] = float(v)
        mujoco.mj_kinematics(raw_m, sc)
        mujoco.mj_collision(raw_m, sc)
        robot_prefix = ("robot0", "gripper0", "mobilebase0")
        for i in range(int(sc.ncon)):
            c = sc.contact[i]
            if c.dist > -0.002:
                continue
            n1 = self.sim.model.geom_id2name(int(c.geom1)) or ""
            n2 = self.sim.model.geom_id2name(int(c.geom2)) or ""
            r1, r2 = n1.startswith(robot_prefix), n2.startswith(robot_prefix)
            if r1 == r2:
                continue
            other = n2 if r1 else n1
            if other.startswith("floor") or other == "":
                continue
            if any(w and w in other for w in ignore):
                continue
            return f"{n1 if r1 else n2} x {other}"
        return ""

    #: the descent ladder ``grasp_reach`` walks upward from ``bottom_z - below_m``
    REACH_LADDER_M: tuple[float, ...] = (0.0, 0.03, 0.06, 0.10, 0.16, 0.24, 0.34)

    def grasp_reach(
        self,
        name: str,
        *,
        base_pose: Any | None = None,
        torso: float | None = None,
        below_m: float = 0.02,
        approach_m: float = 0.06,
        quick: bool = True,
    ) -> dict[str, Any]:
        """**Predicted** ``site_above_bottom_m`` for a top-down pinch on ``name`` from a
        hypothetical standing pose — kinematics only, ZERO frames.

        ``pick_object`` reports ``site_above_bottom_m`` *after* it has spent the descent,
        and it is the number that separates a grasp from a miss (measured over the 22
        shape-D atomic tasks: ≤ 0.039 grasped 12 of 12, ≥ 0.07 failed 9 of 9).  This asks
        the same question of a base pose the robot has not driven to yet: solve IK for the
        grip site at ``bottom_z - below_m`` over the object, and when that is outside the
        envelope walk up :attr:`REACH_LADDER_M` until it is not.  The answer is the lowest
        the grip site can be put over the object, minus the object's bottom — so
        ``< 0.04`` predicts a grasp and ``>= 0.07`` predicts a miss.

        Returns ``{"site_above_bottom_m", "reachable", "hover_ok", "base_pose", "torso",
        "width_m", "fits_jaw", "solves"}``.
        """
        ext = self.object_extent(name)
        plan = self.grasp_axis(name)
        pos = self.get_object_pose(name)[0]
        bottom, top = float(ext["bottom_z"]), float(ext["top_z"])
        quat = self.top_down_grasp_quat(float(plan["yaw"]))
        qpos = self.hypothetical_qpos(base_pose, torso)
        rep: dict[str, Any] = {
            "base_pose": None if base_pose is None else [round(float(v), 3) for v in base_pose],
            "torso": None if torso is None else round(float(torso), 3),
            "width_m": plan["width_m"],
            "fits_jaw": plan["fits_jaw"],
            "reachable": False,
            "solves": 0,
        }
        ignore = (name, *[g for g in (self._object_geom_words(name),) if g])
        best = None
        for dz in self.REACH_LADDER_M:
            z = bottom - float(below_m) + float(dz)
            rep["solves"] += 1
            sol = self.solve_arm_ik(
                [pos[0], pos[1], z], quat, qpos=qpos, quick=quick, either_finger_side=True
            )
            for cand in [sol, *sol.get("candidates", [])]:
                if not cand.get("ok"):
                    continue
                hit = self.posture_collision(cand["q"], qpos=qpos, ignore=ignore)
                if not hit:
                    best = z
                    break
                rep["collision"] = hit
            if best is not None:
                break
        rep["site_above_bottom_m"] = (
            round(float(best - bottom), 4) if best is not None else round(
                float(self.REACH_LADDER_M[-1] - below_m + 0.10), 4
            )
        )
        rep["reachable"] = best is not None and float(best - bottom) < 0.04
        # pick_object drops its staging headroom 0.12 -> 0.06 -> 0.03 when the cavity roof
        # is in the way, so the prediction has to walk the same ladder down
        rep["hover_ok"] = False
        for head in (float(approach_m), 0.03, 0.015):
            rep["solves"] += 1
            hov = self.solve_arm_ik([pos[0], pos[1], top + head], quat, qpos=qpos, quick=quick)
            if hov.get("ok") and not self.posture_collision(hov["q"], qpos=qpos, ignore=ignore):
                rep["hover_ok"] = True
                rep["hover_m"] = head
                break
        return rep

    def _object_geom_words(self, name: str) -> str:
        """The string an object's own collision geoms carry in their names (so a grasp
        posture is not called a collision for touching what it is grasping).
        """
        obj = self.env.objects.get(name)
        return str(getattr(obj, "name", name) or name) if obj is not None else name

    def find_pick_pose(
        self,
        name: str,
        *,
        standoffs: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        yaw_offsets: tuple[float, ...] = (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.9, -1.9, 2.8),
        torsos: tuple[float, ...] = (0.0, 0.17, 0.34),
        max_probe: int = 20,
        below_m: float = 0.02,
        target: Any | None = None,
    ) -> dict[str, Any]:
        """Search standing poses for one the arm can pick ``name`` from — **zero frames**.

        Candidates stand ``standoff`` metres around the object on a fan of headings, facing
        it, footprint clear (:meth:`base_pose_clear`); the torso lift is a free variable and
        is searched with them, because on a shelf or in a cabinet it is worth more than any
        base position.  They are prefiltered by shoulder geometry (cheap), then the best
        ``max_probe`` are priced with :meth:`grasp_reach` (IK, still zero frames) and ranked
        by the predicted ``site_above_bottom_m``, then by how far the base has to drive.

        Returns ``{"best", "here", "candidates", "probed"}``; ``best`` is None when nothing
        is predicted to reach.  ``here`` is the same prediction for standing still, so the
        caller can decline to drive.
        """
        pos = self.get_object_pose(name)[0]
        ext = self.object_extent(name)
        tgt = np.array([pos[0], pos[1], float(ext["bottom_z"])], dtype=float)
        goal = tgt if target is None else np.asarray(target, dtype=float)
        base = self.get_base_pose()
        torso_now = self.torso_height() if self._has_torso() else 0.0
        here = self.grasp_reach(name, base_pose=None, torso=None, below_m=below_m)
        here["drive_m"] = 0.0
        here["torso"] = round(float(torso_now), 3)
        here["pose"] = [round(float(v), 3) for v in base]
        # The shoulder is what reaches, not the base origin: it sits ~0.3 m ahead of the
        # base and rides up to 0.34 m with the torso.  Both are read LIVE and carried onto
        # every candidate, because a standing pose is chosen for where the SHOULDER lands.
        d0 = np.array([base[0] - goal[0], base[1] - goal[1]])
        phi0 = float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 1e-6 else 0.0
        sh = self._shoulder()
        cb, sb = np.cos(base[2]), np.sin(base[2])
        sh_off = np.array([[cb, sb], [-sb, cb]]) @ (sh[:2] - base[:2])  # in the base frame
        sh_z0 = float(sh[2]) - float(torso_now)
        pool: list[dict[str, Any]] = []
        seen: set[tuple[float, float, float]] = set()
        for standoff in standoffs:
            for off in yaw_offsets:
                phi = phi0 + off
                x = float(goal[0] + standoff * np.cos(phi))
                y = float(goal[1] + standoff * np.sin(phi))
                yaw = float(np.arctan2(np.sin(phi + np.pi), np.cos(phi + np.pi)))
                key = (round(x, 2), round(y, 2), round(yaw, 2))
                if key in seen or not self.base_pose_clear(x, y, yaw):
                    continue
                seen.add(key)
                drive = float(np.hypot(x - base[0], y - base[1]))
                cy, sy = np.cos(yaw), np.sin(yaw)
                sxy = np.array([x, y]) + np.array([[cy, -sy], [sy, cy]]) @ sh_off
                flat = float(np.linalg.norm(goal[:2] - sxy))
                for t in torsos:
                    dz = float(goal[2]) - (sh_z0 + float(t))
                    dist = float(np.hypot(flat, dz))
                    # too CLOSE is a failure as surely as too far: measured 2026-08-28, a
                    # base 0.36 m from the object left no posture that was not folded
                    # through the counter, and the descent stalled 0.27 m away
                    if dist > self.ARM_REACH_M or dist < 0.34:
                        continue
                    pool.append(
                        {
                            "pose": [round(x, 3), round(y, 3), round(yaw, 3)],
                            "torso": round(float(t), 3),
                            "standoff": standoff,
                            "shoulder_dist": round(dist, 3),
                            "drive_m": round(drive, 3),
                            "rank": abs(dist - 0.58) + 0.10 * drive + 0.15 * abs(off) + 0.3 * t,
                        }
                    )
        pool.sort(key=lambda c: c["rank"])
        probed: list[dict[str, Any]] = []
        for cand in pool[: int(max_probe)]:
            got = self.grasp_reach(
                name, base_pose=cand["pose"], torso=cand["torso"], below_m=below_m
            )
            cand.update(
                {
                    "site_above_bottom_m": got["site_above_bottom_m"],
                    "reachable": got["reachable"],
                    "hover_ok": got["hover_ok"],
                }
            )
            probed.append(cand)
            if got["reachable"] and got["hover_ok"] and cand["drive_m"] < 1.2:
                break
        probed.sort(
            key=lambda c: (
                0 if (c["reachable"] and c["hover_ok"]) else 1 if c["reachable"] else 2,
                round(c["site_above_bottom_m"], 3),
                c["drive_m"] + 0.4 * c["torso"],
            )
        )
        good = [c for c in probed if c["reachable"] and c["hover_ok"]]
        best = good[0] if good else (probed[0] if probed and probed[0]["reachable"] else None)
        return {
            "best": best,
            "here": here,
            "candidates": probed[:6],
            "probed": len(probed),
            "pool": len(pool),
        }

    def stand_to_pick(
        self,
        name: str,
        *,
        below_m: float = 0.02,
        margin_m: float = 0.02,
        max_drive_m: float = 2.6,
        max_steps: int = 300,
        **search: Any,
    ) -> dict[str, Any]:
        """Stand where :meth:`find_pick_pose` predicts the pinch will reach, and lift the
        torso to match.  Costs nothing when standing still already predicts a grasp.

        The order matters: the torso is set BEFORE the drive when it has to come down (a
        raised torso with the arm out is what the drive would fight) and after it when it
        goes up.  Returns ``{"moved", "predicted_site_above_bottom_m", "before", "after",
        "drive", "torso", "steps"}``.
        """
        s0 = self._step_count
        search_out = self.find_pick_pose(name, below_m=below_m, **search)
        here = search_out["here"]
        rep: dict[str, Any] = {
            "moved": False,
            "here_site_above_bottom_m": here["site_above_bottom_m"],
            "predicted_site_above_bottom_m": here["site_above_bottom_m"],
            "candidates": search_out["candidates"][:3],
            "steps": 0,
        }
        best = search_out["best"]
        if here["reachable"] and here["hover_ok"]:
            rep["reason"] = "standing still already reaches: no drive"
            return rep
        if best is None:
            rep["reason"] = (
                "no standing pose in the fan reaches the object's lower half: the limit is "
                "the arm's vertical envelope, not the base position"
            )
            return rep
        if float(best["drive_m"]) > float(max_drive_m):
            rep["reason"] = f"the best pose is {best['drive_m']} m away (over max_drive_m)"
            return rep
        if self._budget_left() < 380:
            rep["reason"] = (
                f"only {self._budget_left()} frames left: a tuck and a drive is 150-400 of "
                "them and the pick and the placement still have to be paid for"
            )
            return rep
        better = (best["reachable"], best["hover_ok"]) > (here["reachable"], here["hover_ok"])
        if not better and best["site_above_bottom_m"] > here["site_above_bottom_m"] - float(
            margin_m
        ):
            rep["reason"] = "no candidate is materially better than standing still"
            return rep
        torso_now = self.torso_height() if self._has_torso() else 0.0
        want = float(best["torso"])
        if self._has_torso() and want < torso_now - 0.02:
            rep["torso_down"] = self.set_torso_height(want)
        self.tuck_arm()
        x, y, yaw = best["pose"]
        rep["drive"] = self.drive_to(x, y, yaw, max_steps=int(max_steps))
        if not rep["drive"].get("reached") and rep["drive"].get("blocked"):
            # the straight line runs through an island: route around it (measured
            # 2026-08-28, PickPlaceCounterToCabinet's best pose is 2.5 m away across one)
            with contextlib.suppress(Exception):
                rep["route"] = self.drive_route(x, y, yaw)
        if self._has_torso() and want > self.torso_height() + 0.02:
            rep["torso"] = self.set_torso_height(want)
        rep["moved"] = True
        rep["pose"] = self.get_base_pose().round(3).tolist()
        after = self.grasp_reach(name, below_m=below_m)
        rep["predicted_site_above_bottom_m"] = after["site_above_bottom_m"]
        rep["reachable"] = bool(after["reachable"] and after["hover_ok"])
        rep["steps"] = self._step_count - s0
        return rep

    def _approach(
        self,
        point: np.ndarray,
        quat: np.ndarray,
        normal: np.ndarray,
        *,
        tolerance_m: float = 0.03,
        max_steps: int = 250,
        force_move: bool = False,
    ) -> bool:
        """Reach ``point`` with the arm; when it cannot (or ``force_move``), drive to the
        standing poses that :meth:`find_standing_pose` proposes along ``normal`` and try
        again from each.  The arm attempt is skipped when the point is plainly beyond the
        shoulder's reach.
        """
        if not force_move and self._shoulder_distance(point) <= self.ARM_REACH_M:
            try:
                rep = self.goto_pose(point, quat, tolerance_m=tolerance_m, max_steps=max_steps)
                _trace(f"approach {np.round(point, 3).tolist()} direct: {rep}")
                return True
            except RuntimeError as exc:
                _trace(f"approach {np.round(point, 3).tolist()} direct failed: {exc}")
        else:
            _trace(
                f"approach {np.round(point, 3).tolist()}: shoulder distance "
                f"{self._shoulder_distance(point):.2f} > reach, repositioning"
            )
        n = np.asarray(normal, dtype=float).copy()
        n[2] = 0.0
        candidates = self.find_standing_pose(point, n)
        if candidates:
            self.tuck_arm()
        for cand in candidates:
            x, y, yaw = cand["pose"]
            drive = self.drive_to(x, y, yaw, max_steps=300)
            _trace(f"  candidate {cand} drive: {drive}")
            try:
                rep = self.goto_pose(point, quat, tolerance_m=tolerance_m, max_steps=max_steps)
                _trace(f"  reached from candidate: {rep}")
                return True
            except RuntimeError as exc:
                _trace(f"  goto failed from candidate: {exc}")
                continue
        return False

    def _shoulder_distance(self, point: Any) -> float:
        """Distance from the arm's shoulder (``robot0_link1``) to a world point."""
        bid = self.sim.model.body_name2id("robot0_link1")
        shoulder = np.asarray(self.sim.data.body_xpos[bid], dtype=float)
        return float(np.linalg.norm(np.asarray(point, dtype=float) - shoulder))

    # ------------------------------------------------------------ articulated fixtures

    def _joint_geometry(self, jn: str) -> dict[str, Any]:
        raw_m, raw_d = self._raw()
        jid = self.sim.model.joint_name2id(jn)
        jtype = int(raw_m.jnt_type[jid])
        lo, hi = (float(v) for v in raw_m.jnt_range[jid])
        # np.array, not np.asarray: MjData arrays are already float64, so asarray is a LIVE
        # VIEW that moves with the part (and the in-place normalisation used to write into
        # mjData.xaxis) — round-3 review
        axis = np.array(raw_d.xaxis[jid], dtype=float)
        axis /= max(np.linalg.norm(axis), 1e-9)
        return {
            "name": jn,
            "type": {3: "hinge", 2: "slide"}.get(jtype, "other"),
            "axis": axis,
            "anchor": np.array(raw_d.xanchor[jid], dtype=float),
            "range": (lo, hi),
            "qpos": float(self.sim.data.get_joint_qpos(jn)),
            # qpos direction in which the benchmark's normalised fraction grows:
            # Fixture.get_joint_state flips whenever the range starts below zero
            # (fraction 1 sits at ``lo``), so a bidirectional joint such as a sink's
            # temperature lever (-0.785..0.785) "opens" toward its negative end
            "open_sign": -1.0 if lo < 0 else 1.0,
            "body": int(raw_m.jnt_bodyid[jid]),
        }

    def _subtree_bodies(self, root: int) -> set[int]:
        raw_m, _ = self._raw()
        parent = raw_m.body_parentid
        out: set[int] = set()
        for body in range(int(raw_m.nbody)):
            k = body
            while k > 0:
                if k == root:
                    out.add(body)
                    break
                k = int(parent[k])
        return out

    def _joint_fraction(self, fixture: Any, jn: str) -> float:
        """The open fraction the benchmark compares with its thresholds for ``jn``.

        Most fixtures normalise over the joint range (``Fixture.get_joint_state``); drawers
        score ``get_door_state`` (travel over 55 % of the depth), so a drawer reads 0.95
        there when the range fraction is only 0.52.
        """
        door_joints = list(getattr(fixture, "door_joint_names", []) or [])
        # only Drawer tasks score get_door_state (travel over 0.55 * depth); every door
        # task scores is_open/is_closed -> get_joint_state (range fraction), and on a
        # HingeCabinet the two differ by ~0.05 % at the 0.9 threshold — round-3 review
        if (
            "Drawer" in type(fixture).__name__
            and hasattr(fixture, "get_door_state")
            and jn in door_joints
        ):
            try:
                state = fixture.get_door_state(self.env)
            except Exception:
                state = {}
            if len(state) == 1 and len(door_joints) == 1:
                return float(next(iter(state.values())))
            local = jn[len(fixture.name) + 1 :]
            for key, val in state.items():
                tag = key.replace("_door", "").replace("door", "")
                if tag and tag in local:
                    return float(val)
        try:
            return float(fixture.get_joint_state(self.env, [jn])[jn])
        except Exception:
            geo = self._joint_geometry(jn)
            lo, hi = geo["range"]
            frac = (geo["qpos"] - lo) / max(hi - lo, 1e-9)
            return float(1.0 - frac if lo < 0 else frac)

    def _geom_bar(self, gid: int, kind: str) -> dict[str, Any]:
        raw_m, raw_d = self._raw()
        # snapshots, not views: a handle centre kept across strokes used to travel with the
        # part (CloseDrawer's report carried the FINAL handle position as "center")
        pos = np.array(raw_d.geom_xpos[gid], dtype=float)
        xmat = np.array(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
        size = np.array(raw_m.geom_size[gid], dtype=float)
        gtype = int(raw_m.geom_type[gid])
        if gtype == 6:  # box: the bar runs along the longest half-size
            i = int(np.argmax(size))
            axis = xmat[:, i]
            half_length = float(size[i])
            thickness = float(2.0 * np.min(size))
        elif gtype in (3, 5):  # capsule / cylinder: along local z
            axis = xmat[:, 2]
            half_length = float(size[1])
            thickness = float(2.0 * size[0])
        else:
            axis = xmat[:, 2]
            half_length = 0.0
            thickness = 0.02
        return {
            "geom": self.sim.model.geom_id2name(gid),
            "kind": kind,
            "center": pos,
            "axis": axis,
            "xmat": xmat,
            "size": size,
            "half_length": half_length,
            "thickness": thickness,
            "geom_type": gtype,
        }

    def _handle_for_joint(self, geo: dict[str, Any]) -> dict[str, Any] | None:
        """The graspable feature on the part moved by joint ``geo``: the largest collision
        geom named ``*handle*`` in the joint body's subtree (a bar), else — for slide
        joints — the collision geom protruding farthest along the opening direction (the
        front lip of a handle-less drawer).
        """
        raw_m, raw_d = self._raw()
        bodies = self._subtree_bodies(geo["body"])
        members = [
            gid
            for gid in range(int(raw_m.ngeom))
            if int(raw_m.geom_bodyid[gid]) in bodies
            and (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid])
        ]
        handles = [g for g in members if "handle" in (self.sim.model.geom_id2name(g) or "")]
        if handles:
            gid = max(handles, key=lambda g: float(np.max(raw_m.geom_size[g])))
            return self._geom_bar(gid, "handle")
        if geo["type"] != "slide" or not members:
            return None
        direction = geo["axis"] * geo["open_sign"]
        best = None
        for gid in members:
            xmat = np.asarray(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
            size = np.asarray(raw_m.geom_size[gid], dtype=float)
            extent = float(np.sum(np.abs(xmat.T @ direction) * size))
            reach = float(np.asarray(raw_d.geom_xpos[gid]) @ direction) + extent
            if best is None or reach > best[0]:
                best = (reach, gid)
        bar = self._geom_bar(best[1], "lip")
        # the lip's top edge is what a top-down pinch closes on
        up = np.array([0.0, 0.0, 1.0])
        z_extent = float(np.sum(np.abs(bar["xmat"].T @ up) * bar["size"]))
        bar["center"] = bar["center"] + up * z_extent
        bar["axis"] = bar["xmat"][:, int(np.argmax(np.abs(np.cross(bar["xmat"].T, up)).sum(1)))]
        return bar

    def _door_panel(self, geo: dict[str, Any]) -> dict[str, Any] | None:
        """The largest non-handle collision geom moved by the joint (the door panel)."""
        raw_m, _ = self._raw()
        bodies = self._subtree_bodies(geo["body"])
        best = None
        for gid in range(int(raw_m.ngeom)):
            if int(raw_m.geom_bodyid[gid]) not in bodies:
                continue
            if not (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid]):
                continue
            if "handle" in (self.sim.model.geom_id2name(gid) or ""):
                continue
            area = float(np.prod(np.sort(np.asarray(raw_m.geom_size[gid], dtype=float))[1:]))
            if best is None or area > best[0]:
                best = (area, gid)
        return self._geom_bar(best[1], "panel") if best else None

    @staticmethod
    def _rodrigues(v: np.ndarray, axis: np.ndarray, th: float) -> np.ndarray:
        return (
            v * np.cos(th) + np.cross(axis, v) * np.sin(th) + axis * (axis @ v) * (1 - np.cos(th))
        )

    def _outward(self, geo: dict[str, Any], point: np.ndarray) -> np.ndarray:
        """Unit direction in which ``point`` (on the moving part) travels when the joint
        opens — for a door this is the outer face's normal, for a drawer the pull axis.
        """
        if geo["type"] == "slide":
            return geo["axis"] * geo["open_sign"]
        r = point - geo["anchor"]
        r = r - geo["axis"] * (r @ geo["axis"])
        r /= max(np.linalg.norm(r), 1e-9)
        return np.cross(geo["axis"] * geo["open_sign"], r)

    def fixture_joints(self, name: str) -> dict[str, Any]:
        """Joints of a fixture with everything a program needs to actuate them.

        Per joint: ``type`` (hinge/slide), world ``axis`` and ``anchor``, ``range``, ``qpos``,
        ``fraction`` (the benchmark's own normalised value: 0 closed … 1 open for doors and
        drawers; sub-millimetre buttons resting past a limit can read slightly outside
        0..1), ``open`` / ``closed`` (that fraction against the thresholds the door and
        drawer scorers use: > 0.9 / < 0.005 for hinges, > 0.95 / < 0.05 when the fixture
        scores drawer travel), ``open_direction`` (where the handle travels when opening =
        the outer face normal; ``None`` when no handle/lip was found) and ``handle``
        (``None``, or bar centre/axis/half_length/thickness with ``kind`` "handle" for a
        *handle* geom, "lip" for the collision geom protruding farthest along the opening
        direction of a handle-less SLIDING part — a drawer front, a rack edge, or a
        button/switch cap when ``half_length`` is under a centimetre).  Hinged parts
        without a handle geom (knobs, lids, a mixer head) get ``handle=None``: use the
        joint axis/anchor.  ``door_joints`` lists the joints the fixture calls its door(s);
        ``is_open`` / ``is_closed`` are ``fixture.is_open()/is_closed()`` with default
        arguments (a fridge's answer for its fridge-door compartment only), kept for
        reference — prefer the per-joint ``open``/``closed``.
        """
        fixture = self._fixture(name)
        door_joints = list(getattr(fixture, "door_joint_names", []) or [])
        names = list(getattr(fixture, "_joint_infos", {}).keys()) or door_joints
        out: dict[str, Any] = {}
        skipped: list[str] = []
        # Drawer tasks score get_door_state (travel / 0.55 depth) at 0.95/0.05; door tasks
        # score the range fraction at 0.9/0.005 (measured 2026-08-27, see _joint_fraction)
        drawer_scored = "Drawer" in type(fixture).__name__
        for jn in names:
            if jn not in self.sim.model.joint_names:
                skipped.append(jn)
                log.warning("%s: joint %s is not in the compiled model", name, jn)
                continue
            geo = self._joint_geometry(jn)
            handle = self._handle_for_joint(geo)
            frac = self._joint_fraction(fixture, jn)
            hi_th, lo_th = (0.95, 0.05) if drawer_scored else (0.9, 0.005)
            entry: dict[str, Any] = {
                "type": geo["type"],
                "axis": geo["axis"].round(4).tolist(),
                "anchor": geo["anchor"].round(4).tolist(),
                "range": [round(v, 4) for v in geo["range"]],
                "qpos": round(geo["qpos"], 4),
                "fraction": round(frac, 4),
                "open": bool(frac > hi_th),
                "closed": bool(frac < lo_th),
                "is_door": jn in door_joints,
                "handle": None,
                "open_direction": None,
            }
            if handle is not None:
                entry["handle"] = {
                    "geom": handle["geom"],
                    "kind": handle["kind"],
                    "center": np.asarray(handle["center"]).round(4).tolist(),
                    "axis": np.asarray(handle["axis"]).round(4).tolist(),
                    "half_length": round(handle["half_length"], 4),
                    "thickness": round(handle["thickness"], 4),
                }
                entry["open_direction"] = self._outward(geo, handle["center"]).round(4).tolist()
            out[jn] = entry
        is_open = is_closed = None
        if door_joints:
            try:
                is_open = bool(fixture.is_open(self.env))
            except Exception as exc:
                log.warning("%s.is_open raised %s", name, exc)
            try:
                is_closed = bool(fixture.is_closed(self.env))
            except Exception as exc:
                log.warning("%s.is_closed raised %s", name, exc)
        handle_geom = getattr(fixture, "handle_name", None)
        if handle_geom is not None and handle_geom not in self.sim.model.geom_names:
            handle_geom = None  # a naming convention (often a body), not a geom
        return {
            "joints": out,
            "door_joints": door_joints,
            "handle_geom": handle_geom,
            "is_open": is_open,
            "is_closed": is_closed,
            "skipped_joints": skipped,
        }

    def fixture_regions(self, name: str) -> dict[str, Any]:
        """Interior placement regions of a fixture (tray, shelf, rack ...) as world boxes:
        the exact volumes RoboCasa samples and checks objects in.
        """
        import robosuite.utils.transform_utils as tu

        fixture = self._fixture(name)
        regions = (
            fixture.get_reset_regions(self.env) if hasattr(fixture, "get_reset_regions") else {}
        )
        pos = np.asarray(fixture.pos, dtype=float)
        rot = tu.euler2mat(np.array([0.0, 0.0, float(getattr(fixture, "rot", 0.0) or 0.0)]))
        out: dict[str, Any] = {}
        for tag, spec in (regions or {}).items():
            offset = np.asarray(spec.get("offset", (0, 0, 0)), dtype=float)
            centre = pos + rot @ offset
            out[tag] = {
                "center": centre.round(4).tolist(),
                "size_xy": [float(v) for v in np.asarray(spec.get("size", (0, 0)), dtype=float)],
                "height": float(spec.get("height", 0.0)),
            }
        anchor = getattr(fixture, "anchor_offset", None)
        if anchor is not None:
            with contextlib.suppress(Exception):
                from robocasa.utils import object_utils

                out["anchor"] = {
                    "center": np.asarray(
                        object_utils.get_pos_after_rel_offset(fixture, anchor), dtype=float
                    )
                    .round(4)
                    .tolist(),
                    "size_xy": [0.0, 0.0],
                    "height": 0.0,
                    "note": "where an auxiliary part (e.g. a lid) sits when in place",
                }
        return out

    @staticmethod
    def _default_door_joints(fixture: Any) -> list[str]:
        """The joints the fixture's own ``is_open`` checks by default (a fridge lists its
        freezer door under ``door_joint_names`` but scores only the fridge compartment).
        """
        for attr in ("_fridge_door_joint_names",):
            names = getattr(fixture, attr, None)
            if names:
                return list(names)
        return list(getattr(fixture, "door_joint_names", []) or [])

    def actuate_door(
        self,
        name: str,
        target_fraction: float,
        *,
        joints: list[str] | None = None,
        stroke_rad: float = 0.3,
        stroke_m: float = 0.08,
        max_strokes: int = 14,
        max_regrasps: int = 4,
    ) -> dict[str, Any]:
        """Bring every door joint of fixture ``name`` (or the given ``joints``) to
        ``target_fraction`` of open — pull or push by the handle, verified against the
        benchmark's own fraction, re-grasping and re-positioning the base when the arm
        stalls.  Returns ``{"done", "fraction", "strokes", "regrasps", "mode", "joints"}``.
        """
        fixture = self._fixture(name)
        names = list(joints or self._default_door_joints(fixture))
        if not names:
            raise RuntimeError(f"{name}: no door joints (pass joints=[...])")
        reports = {}
        for jn in names:
            reports[jn] = self._actuate_joint(
                fixture,
                jn,
                float(target_fraction),
                stroke_rad=stroke_rad,
                stroke_m=stroke_m,
                max_strokes=max_strokes,
                max_regrasps=max_regrasps,
            )

        def short_now(jn: str) -> bool:
            frac = self._joint_fraction(fixture, jn)
            opening = bool(reports[jn].get("opening", target_fraction > 0.5))
            return (frac < target_fraction) if opening else (frac > max(target_fraction, 0.004))

        if len(names) > 1:
            # working the second door of a pair can disturb the first (the base towing the
            # right cabinet door brushed the open left one): re-check and redo what moved,
            # for as many passes as there are doors, until a pass disturbs nothing
            for _pass in range(len(names)):
                redone = False
                for jn in names:
                    if short_now(jn) and reports[jn]["done"]:
                        frac = self._joint_fraction(fixture, jn)
                        _trace(f"{jn} moved to {frac:.3f} while the other door was worked: redo")
                        redo = self._actuate_joint(
                            fixture,
                            jn,
                            float(target_fraction),
                            stroke_rad=stroke_rad,
                            stroke_m=stroke_m,
                            max_strokes=max_strokes,
                            max_regrasps=max_regrasps,
                        )
                        redo["strokes"] += reports[jn]["strokes"]
                        redo["regrasps"] += reports[jn]["regrasps"]
                        redo["redone"] = True
                        reports[jn] = redo
                        redone = True
                if not redone:
                    break
        # the report is a FRESH measurement of every joint, not the last thing each
        # per-joint pass recorded: a door knocked shut by the other door's tow used to
        # be reported "done" (measured 2026-08-27: left 0.034 reported as 0.999)
        live = {jn: self._joint_fraction(fixture, jn) for jn in names}
        for jn in names:
            reports[jn]["fraction_final"] = round(live[jn], 3)
            reports[jn]["done"] = bool(reports[jn]["done"] and not short_now(jn))
        done = all(r["done"] for r in reports.values())
        out = {
            "done": done,
            "fraction": min(live.values()) if target_fraction > 0.5 else max(live.values()),
            "strokes": sum(r["strokes"] for r in reports.values()),
            "regrasps": sum(r["regrasps"] for r in reports.values()),
            "mode": ",".join(sorted({r["mode"] for r in reports.values()})),
            "budget_exhausted": any(r.get("budget_exhausted") for r in reports.values()),
            "joints": reports,
        }
        if len(names) > 1 and not done:
            out["reason"] = (
                "the doors disturb each other: working one door (base tow / arm retreat) "
                "moved the other back; a base path that respects the opened panel is needed"
            )
        return out

    def _actuate_joint(
        self,
        fixture: Any,
        jn: str,
        target: float,
        *,
        stroke_rad: float,
        stroke_m: float,
        max_strokes: int,
        max_regrasps: int,
    ) -> dict[str, Any]:
        geo = self._joint_geometry(jn)
        if geo["type"] not in ("hinge", "slide"):
            raise RuntimeError(f"{jn}: joint type {geo['type']} is not actuatable")

        def fraction() -> float:
            return self._joint_fraction(fixture, jn)

        f0 = fraction()
        opening = target > f0
        sense = geo["open_sign"] if opening else -geo["open_sign"]

        def reached(f: float) -> bool:
            return f >= target if opening else f <= max(target, 0.004)

        handle = self._handle_for_joint(geo)
        strokes = regrasps = rebounds = 0
        frac = f0
        mode = "pull-handle" if opening else "push-handle"
        history: list[str] = []
        last_seat: dict[str, Any] | None = None
        horizontal_hinge = geo["type"] == "hinge" and abs(geo["axis"][2]) < 0.5
        vertical_hinge = geo["type"] == "hinge" and not horizontal_hinge
        if reached(frac):
            return {
                "done": True,
                "fraction": round(frac, 3),
                "strokes": 0,
                "regrasps": 0,
                "mode": "noop",
            }
        if handle is not None and not opening and (geo["type"] == "slide" or vertical_hinge):
            # push the drawer / door shut: closed fingertips against the handle bar (or the
            # lip's outer face), driven along the slide axis - or along the hinge arc, the
            # wrist turning with the panel - in verified strokes.  A drop-down door keeps
            # the edge-lift path below: its bar ends up under the panel.
            stalls = 0
            while not reached(frac) and strokes < max_strokes and regrasps <= max_regrasps:
                handle = self._handle_for_joint(geo)
                point, quat, n = self._grasp_frame(geo, handle)
                contact = point + n * 0.02
                # after a stalled stroke the closed fingertips sit past the bar: open them
                # before retreating, or the hooked hand pins the arm and every later goto
                # stalls (measured 2026-08-27 once tuck_arm stopped opening the gripper)
                self.set_gripper(closed=False, steps=8)
                if not self._approach(point + n * 0.10, quat, n):
                    break
                self.set_gripper(closed=True, steps=8)
                try:
                    self.goto_pose(contact, quat, tolerance_m=0.015, max_steps=150)
                except RuntimeError:
                    regrasps += 1
                    continue
                while not reached(frac) and strokes < max_strokes:
                    before = frac
                    tip = self.get_eef_pose()
                    if vertical_hinge:
                        th = sense * self._remaining_stroke(geo, target, stroke_rad)
                        target_pos = geo["anchor"] + self._rodrigues(
                            tip[:3] - geo["anchor"], geo["axis"], th
                        )
                        quat2 = self._rotate_quat(tip[3:7], geo["axis"], th)
                    else:
                        target_pos, quat2 = tip[:3] - n * stroke_m, quat
                    with contextlib.suppress(RuntimeError):
                        self.goto_pose(
                            target_pos, quat2, tolerance_m=0.02, tolerance_rad=0.2, max_steps=120
                        )
                    strokes += 1
                    self.hold(6)  # let a drawer that hit its stop settle (it can rebound)
                    frac = fraction()
                    _trace(f"push stroke {strokes}: fraction {before:.3f} -> {frac:.3f}")
                    stalls = stalls + 1 if abs(frac - before) < 0.015 else 0
                    if stalls >= 2:
                        break
                if reached(frac):
                    self.hold(10)
                    frac = fraction()
                    if reached(frac):
                        break
                    rebounds += 1
                    continue  # it rebounded: push again from the handle
                regrasps += 1
                stalls = 0
                with contextlib.suppress(RuntimeError):
                    tip = self.get_eef_pose()
                    n = self._outward_now(geo, n)
                    self.goto_pose(tip[:3] + n * 0.10, tip[3:7], tolerance_m=0.03, max_steps=100)
            with contextlib.suppress(RuntimeError):
                tip = self.get_eef_pose()
                n = self._outward_now(geo, n)
                self.goto_pose(tip[:3] + n * 0.10, tip[3:7], tolerance_m=0.03, max_steps=100)
            self.set_gripper(closed=False, steps=6)
            history.append(f"push-handle: {strokes} strokes -> {fraction():.3f}")
        elif handle is not None and opening:
            # pinch the handle (or lip) and move it along the joint's path in verified
            # strokes.  Vertical-hinge doors are pulled with the BASE (the arm holds the
            # pinch while the base circles the hinge): the arm stays out of the panel's
            # sweep and the pull force is the base's, not the wrist's.
            styles = self._grasp_styles(handle)
            style_idx = 0
            base_pull = False
            stalled_strokes = 0
            need_room = False
            while not reached(frac) and strokes < max_strokes and regrasps <= max_regrasps:
                handle = self._handle_for_joint(geo)
                self.set_gripper(closed=False, steps=8)
                grasped = None
                for k in range(len(styles)):
                    style = styles[(style_idx + k) % len(styles)]
                    point, quat, n = self._grasp_frame(geo, handle, style)
                    pre = point + self._pre_dir(style, n) * 0.10
                    if not self._reach(pre, quat):
                        continue
                    try:
                        rep = self.goto_pose(point, quat, tolerance_m=0.012, max_steps=200)
                        _trace(f"grasp {style} {np.round(point, 3).tolist()}: {rep}")
                    except RuntimeError as exc:
                        _trace(f"grasp {style} {np.round(point, 3).tolist()} failed: {exc}")
                        continue
                    if not rep.get("reached"):
                        continue  # a near miss is not a pinch on the bar
                    grasped = style
                    style_idx = (style_idx + k) % len(styles)
                    break
                if grasped is None or need_room:
                    if grasped is not None:
                        self.set_gripper(closed=False, steps=6)
                    point, quat, n = self._grasp_frame(geo, handle, styles[style_idx])
                    pre = point + self._pre_dir(styles[style_idx], n) * 0.10
                    stand_dir = n.copy()
                    if geo["type"] == "hinge":
                        # stand toward the free edge so the arm reaches the handle from
                        # outside the panel's sweep
                        r = point - geo["anchor"]
                        r -= geo["axis"] * (r @ geo["axis"])
                        stand_dir = n + 0.5 * r / max(np.linalg.norm(r), 1e-9)
                    # after the arm ran out of reach mid-stroke, move the base even when the
                    # handle is still reachable: the next strokes need room
                    if not self._approach(pre, quat, stand_dir, force_move=need_room):
                        break
                    need_room = False
                    regrasps += 1
                    continue
                seat = last_seat = self._seat_pinch(geo, handle, styles[style_idx])
                width = self.set_gripper(closed=True)
                _trace(
                    f"pinch opening {width:.3f} (bar thickness {handle['thickness']:.3f}), "
                    f"seat {seat}"
                )
                if width < 0.006:
                    regrasps += 1
                    continue
                slipped = False
                while not reached(frac) and strokes < max_strokes:
                    before = frac
                    tip = self.get_eef_pose()
                    if geo["type"] == "hinge":
                        th = sense * self._remaining_stroke(geo, target, stroke_rad)
                        target_pos = geo["anchor"] + self._rodrigues(
                            tip[:3] - geo["anchor"], geo["axis"], th
                        )
                        quat2 = self._rotate_quat(tip[3:7], geo["axis"], th)
                    else:
                        th = sense * stroke_rad
                        target_pos = tip[:3] + geo["axis"] * sense * stroke_m
                        quat2 = tip[3:7]
                    if base_pull:
                        ok = self._base_pull(geo["anchor"], geo["axis"], th)
                        after_tip = self.get_eef_pose()[:3]
                        err = float(np.linalg.norm(after_tip - target_pos))
                        _trace(f"base stroke {strokes}: ok={ok} tip err {err:.3f}")
                    else:
                        try:
                            rep = self.goto_pose(
                                target_pos,
                                quat2,
                                tolerance_m=0.03,
                                tolerance_rad=0.2,
                                max_steps=150,
                            )
                            _trace(f"stroke {strokes} -> {np.round(target_pos, 3).tolist()}: {rep}")
                        except RuntimeError as exc:
                            _trace(f"stroke {strokes} -> {np.round(target_pos, 3).tolist()}: {exc}")
                            if vertical_hinge and fraction() > 0.3:
                                # the arm ran out of reach with the pinch still on: finish by
                                # towing with the base instead of releasing (a re-approach
                                # sweeps the arm through the half-open door)
                                base_pull = True
                                _trace("arm out of reach: towing with the base")
                            else:
                                slipped = True  # re-grasp from a new standing spot
                                need_room = True
                    strokes += 1
                    frac = fraction()
                    _trace(f"  fraction {before:.3f} -> {frac:.3f}")
                    if abs(frac - before) < 0.015:
                        slipped = True
                        stalled_strokes += 1
                    else:
                        stalled_strokes = 0
                    if slipped:
                        break
                if reached(frac):
                    self.hold(10)
                    frac = fraction()
                    if reached(frac):
                        break
                    _trace(f"rebounded to {frac:.3f}: stroking again")
                    rebounds += 1
                    continue
                if vertical_hinge and not base_pull and stalled_strokes >= 2:
                    base_pull = True  # the pinch keeps slipping: tow with the base from now on
                    _trace("switching to base pull")
                # let gravity finish a drop-down door before re-grasping
                if opening and horizontal_hinge:
                    self.set_gripper(closed=False, steps=8)
                    self.hold(25)
                    frac = fraction()
                    if reached(frac):
                        break
                regrasps += 1
                self.set_gripper(closed=False, steps=8)
                with contextlib.suppress(RuntimeError):
                    tip = self.get_eef_pose()
                    back = self._pre_dir(styles[style_idx], self._outward_now(geo))
                    self.goto_pose(tip[:3] + back * 0.08, tip[3:7], tolerance_m=0.03, max_steps=120)
            self.set_gripper(closed=False, steps=8)
            with contextlib.suppress(Exception):
                # retreat along the panel's CURRENT outer normal: after a 90 degree swing the
                # grasp-time normal runs along the panel and the open fingers shoved the
                # cabinet door back from 0.98 to 0.77 (measured 2026-08-27)
                tip = self.get_eef_pose()
                back = self._pre_dir(styles[style_idx], self._outward_now(geo))
                self.goto_pose(tip[:3] + back * 0.12, tip[3:7], tolerance_m=0.03, max_steps=150)
        frac = fraction()
        if not reached(frac) and not opening and geo["type"] == "hinge":
            panel_mode = "edge-lift" if horizontal_hinge else "push-panel"
            if self._door_panel(geo) is None:
                panel_mode = "no-panel"
            frac, s2, r2 = self._close_hinge_by_panel(
                fixture,
                jn,
                geo,
                target,
                stroke_rad,
                max_strokes - strokes,
                horizontal_hinge,
                max_regrasps=max(0, max_regrasps - regrasps),
            )
            strokes += s2
            regrasps += r2
            # the mode names the path that actually MOVED the part last: a panel push
            # that never got a stroke in (its approach refused) leaves the mode of the
            # handle path, with the refusal in the history
            if s2 > 0 or panel_mode == "no-panel" or strokes == 0:
                mode = panel_mode
            history.append(f"{panel_mode}: {s2} strokes -> {fraction():.3f}")
        elif handle is None and strokes == 0 and not reached(frac):
            mode = "no-handle"  # nothing was attempted: no *handle* geom / lip on this joint
        frac = fraction()
        report = {
            "done": bool(reached(frac)),
            "fraction": round(frac, 3),
            "opening": bool(opening),
            "start_fraction": round(f0, 3),
            "strokes": strokes,
            "regrasps": regrasps,
            "rebounds": rebounds,
            "mode": mode,
            "history": history,
            "budget_exhausted": self._budget_left() <= 0,
        }
        if last_seat is not None:
            report["seat"] = last_seat
        if strokes == 0 and not reached(frac) and mode not in ("no-handle", "no-panel", "noop"):
            report["reason"] = (
                f"{jn}: every approach was refused (no standing spot with the footprint clear "
                "and the handle within the arm's reach, or the servo stalled before the "
                "handle) — nothing was stroked; try find_standing_pose + drive_to yourself"
            )
        if mode in ("no-handle", "no-panel"):
            report["reason"] = (
                f"{jn}: no graspable handle/lip (or pushable panel) was found under the "
                "joint body — nothing was attempted; use get_fixture_joints() axis/anchor "
                "and goto_pose/move_by_displacement directly"
            )
        return report

    def _outward_now(self, geo: dict[str, Any], fallback: Any = None) -> np.ndarray:
        """The handle's outer normal as the part stands now.  When the handle cannot be
        found the grasp-time normal ``fallback`` is returned (the joint anchor gives a
        zero radial vector, i.e. a retreat that goes nowhere); with no fallback either,
        world +z.
        """
        handle = self._handle_for_joint(geo)
        if handle is None:
            if fallback is not None:
                return np.asarray(fallback, dtype=float)
            return np.array([0.0, 0.0, 1.0])
        return self._outward(geo, np.asarray(handle["center"], dtype=float))

    def _remaining_stroke(self, geo: dict[str, Any], target: float, stroke: float) -> float:
        """A hinge stroke no longer than the travel left to ``target`` (+6 deg): towing a
        door a full stroke past its stop bounced the cabinet door from 1.00 back to 0.88
        (measured 2026-08-27); a tighter +3 deg margin left the microwave 0.01 short and
        the next tow slipped.  For targets >= 0.94 the stroke can still end ~3 deg past
        the stop (ask doors for 0.95, not 0.97).
        Assumes the scorer's fraction is the joint range normalised (true for every hinge
        fixture here; drawers use their own depth and are not clamped by this).
        """
        lo, hi = geo["range"]
        t = target if geo["open_sign"] > 0 else 1.0 - target
        q_target = lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))
        q_now = float(self.sim.data.get_joint_qpos(geo["name"]))
        # NOTE: an extra clamp to the physical stop (+1 deg) was tried 2026-08-27 and
        # reverted: it cost the certified OpenMicrowave 1042 frames instead of 398 (more,
        # shorter tows) without changing the outcome; the +6 deg margin is what the
        # certificates were earned with
        return float(max(min(stroke, abs(q_target - q_now) + 0.10), 0.05))

    def _base_pull(self, anchor: np.ndarray, axis: np.ndarray, th: float) -> bool:
        """Tow what the gripper pinches along a hinge's arc with the BASE: rotate the base
        pose about the hinge by ``th`` while the arm actively holds its base-relative pose
        (a zero arm delta would leave the OSC limp and the door would drag the arm behind).
        """
        base = self.get_base_pose()
        p = np.array([base[0], base[1], 0.0])
        p2 = anchor + self._rodrigues(p - anchor, axis, th)
        yaw2 = base[2] + th * float(np.sign(axis[2]) if abs(axis[2]) > 0.5 else 0.0)
        rep = self.drive_to(float(p2[0]), float(p2[1]), float(yaw2), max_steps=100, hold_arm=True)
        self.hold(3)
        return bool(rep.get("reached"))

    def _seat_pinch(
        self, geo: dict[str, Any], handle: dict[str, Any], style: dict[str, Any]
    ) -> dict[str, Any]:
        """Slide the open fingers deeper over a handle bar, along the tool axis, until the
        finger pads sit at the bar's REAR (panel-side) edge - or the fingertips meet the
        panel - and report the depth reached.

        A pinch at the bar's centre holds only by friction, and a door that rubs its frame
        (a HingeCabinet's panel touches the box's top, bottom and hinge-side walls and
        needs ~4 Nm at the hinge) slides straight out of it: measured 2026-08-27, the
        fingers left the bar after one 0.1 rad stroke, six regrasps never got past 0.13.
        With the pads seated behind the bar the pull is carried by the pads' proximal
        faces (a hook, not a friction pinch): the same door opened to 0.91 in eight
        strokes without a regrasp.  Only the STRAIGHT (tool along the outer normal,
        pitch 0) style of a real ``handle`` geom is seated; pitched, lip and top-down
        styles are left where they are (a pitched wrist puts one fingertip nearer the
        panel than the bound below assumes).  The nearest face of the moving part that
        faces the tool and lies behind the bar bounds the depth so the fingertips stop
        ~4 mm short of it — the face is found among the part's own geoms, not taken from
        the largest geom's thinnest axis (which is a drawer's inner bottom, and reported
        "no room" for every drawer); the bar's extent along the tool axis is its support
        width in that direction, not its thinnest side.  A bar with no room behind it is
        reported ``skipped``; ``seated`` False after a real attempt means the pull is a
        friction pinch (microwave / fridge bars stand < 1.5 cm off their panels).
        """
        if handle.get("kind") != "handle" or style.get("top") or float(style.get("pitch", 0)) != 0:
            return {"depth": 0.0, "skipped": True, "reason": "not the straight handle style"}
        import robosuite.utils.transform_utils as tu

        tip = self.get_eef_pose()
        tool = np.asarray(tu.quat2mat(_xyzw(tip[3:7])), dtype=float)[:, 2]
        centre = np.asarray(handle["center"], dtype=float)
        half = self._extent_along(handle, tool)
        want = self.PAD_BEHIND_SITE_M + half + 0.003
        limit = want + 0.01
        face = self._face_ahead(geo, centre, tool)
        if face is not None:
            limit = face - self.FINGERTIP_BEYOND_SITE_M - 0.004
        depth = float(min(want, limit))
        if depth <= 0.004:
            return {
                "depth": 0.0,
                "skipped": True,
                "reason": "no room behind the bar",
                "face_m": None if face is None else round(face, 4),
            }
        target = tip[:3] + tool * (depth - float((tip[:3] - centre) @ tool))
        with contextlib.suppress(RuntimeError):
            self.goto_pose(target, tip[3:7], tolerance_m=0.004, max_steps=120)
        got = float((self.get_eef_pose()[:3] - centre) @ tool)
        return {
            "depth": round(got, 4),
            "wanted": round(depth, 4),
            "face_m": None if face is None else round(face, 4),
            "bar_half_m": round(half, 4),
            "seated": bool(got >= self.PAD_BEHIND_SITE_M + half - 0.004),
        }

    @staticmethod
    def _extent_along(handle: dict[str, Any], direction: np.ndarray) -> float:
        """Half extent of a handle geom along a world ``direction`` (its support width:
        a box's projected half sizes, a capsule/cylinder's radius).
        """
        size = np.asarray(handle.get("size", []), dtype=float).reshape(-1)
        xmat = handle.get("xmat")
        if xmat is not None and size.size == 3 and int(handle.get("geom_type", 6)) == 6:
            return float(np.sum(np.abs(np.asarray(xmat, dtype=float).T @ direction) * size))
        return float(handle["thickness"]) / 2.0

    def _face_ahead(self, geo: dict[str, Any], point: np.ndarray, tool: np.ndarray) -> float | None:
        """Distance along ``tool`` from ``point`` to the nearest face of the joint's own
        moving part that FACES the tool (|n . tool| > 0.7), lies ahead of the point and
        whose lateral footprint contains the point (+3 cm); ``None`` when no such face
        exists.  Box and mesh (AABB) geoms only, handles excluded.
        """
        raw_m, raw_d = self._raw()
        bodies = self._subtree_bodies(geo["body"])
        best: float | None = None
        for gid in range(int(raw_m.ngeom)):
            if int(raw_m.geom_bodyid[gid]) not in bodies:
                continue
            if not (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid]):
                continue
            if "handle" in (self.sim.model.geom_id2name(gid) or ""):
                continue
            gtype = int(raw_m.geom_type[gid])
            xmat = np.asarray(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
            if gtype == 6:
                size = np.asarray(raw_m.geom_size[gid], dtype=float)
                centre = np.asarray(raw_d.geom_xpos[gid], dtype=float)
            elif gtype == 7:
                aabb = np.asarray(raw_m.geom_aabb[gid], dtype=float).reshape(2, 3)
                size = aabb[1]
                centre = np.asarray(raw_d.geom_xpos[gid], dtype=float) + xmat @ aabb[0]
            else:
                continue
            rel = centre - np.asarray(point, dtype=float)
            for i in range(3):
                n = xmat[:, i]
                s = float(n @ tool)
                if abs(s) < 0.7:
                    continue
                if s < 0.0:
                    n, s = -n, -s
                plane = float(rel @ n) - float(size[i])
                if plane <= 0.002:
                    continue  # the face is at or behind the point
                lateral_ok = all(
                    abs(float(rel @ xmat[:, j])) <= float(size[j]) + 0.03
                    for j in range(3)
                    if j != i
                )
                if not lateral_ok:
                    continue
                d = plane / s
                if best is None or d < best:
                    best = d
        return best

    @staticmethod
    def _grasp_styles(handle: dict[str, Any]) -> list[dict[str, Any]]:
        """Grasp frames to try on a bar, in order: the tool along the outer normal, then
        pitched down (vertical bars: keeps the wrist inside its joint range at counter
        height), then from above (a horizontal bar straddled front/back; the free top end of
        a vertical bar).  A lip only takes the top-down pinch.
        """
        if handle["kind"] == "lip":
            return [{"top": True, "pitch": 0.0}]
        vertical = abs(float(handle["axis"][2])) > 0.7
        if vertical:
            return [
                {"top": False, "pitch": 0.0},
                {"top": False, "pitch": 0.6},
                {"top": True, "pitch": 0.0},
            ]
        return [
            {"top": False, "pitch": 0.0},
            {"top": False, "pitch": 0.5},
            {"top": True, "pitch": 0.0},
        ]

    @staticmethod
    def _pre_dir(style: dict[str, Any], n: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0]) if style.get("top") else np.asarray(n, dtype=float)

    def _grasp_frame(
        self, geo: dict[str, Any], handle: dict[str, Any], style: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(grasp point, wxyz orientation, outward normal) to pinch ``handle`` right now."""
        style = style or {"top": False, "pitch": 0.0}
        point = np.asarray(handle["center"], dtype=float)
        bar = np.asarray(handle["axis"], dtype=float)
        n = self._outward(geo, point)
        up = np.array([0.0, 0.0, 1.0])
        if handle["kind"] == "lip":
            tool = -up
            close = n - tool * (n @ tool)
            point = point - up * 0.015
        elif style.get("top"):
            tool = -up
            if abs(bar @ up) > 0.7:  # vertical bar: pinch its free top end across the sides
                bar_up = bar * np.sign(bar @ up)
                point = point + bar_up * (handle["half_length"] - 0.010)
                close = np.cross(n, bar_up)
            else:  # horizontal bar: straddle it front/back
                close = n - tool * (n @ tool)
        else:
            pitch = float(style.get("pitch", 0.0))
            tool = -n * np.cos(pitch) - up * np.sin(pitch)
            close = np.cross(n, bar)
            if np.linalg.norm(close) < 1e-6:
                close = up
        quat = self.horizontal_grasp_quat(tool, close, nearest=True)
        return point, quat, n

    @staticmethod
    def _rotate_quat(quat_wxyz: Any, axis: np.ndarray, th: float) -> np.ndarray:
        """``quat`` rotated by ``th`` about the world ``axis`` (wxyz in, wxyz out)."""
        import robosuite.utils.transform_utils as tu

        mat = tu.quat2mat(_xyzw(quat_wxyz))
        k = np.asarray(axis, dtype=float)
        k /= max(np.linalg.norm(k), 1e-9)
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot = np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * (kx @ kx)
        return _wxyz_from_mat(rot @ mat)

    def _reach(self, point: np.ndarray, quat: np.ndarray, tolerance_m: float = 0.03) -> bool:
        """One arm attempt at a pose (no base motion); False when out of reach."""
        if self._shoulder_distance(point) > self.ARM_REACH_M:
            return False
        try:
            rep = self.goto_pose(point, quat, tolerance_m=tolerance_m, max_steps=200)
            _trace(f"reach {np.round(point, 3).tolist()}: {rep}")
            return True
        except RuntimeError as exc:
            _trace(f"reach {np.round(point, 3).tolist()} failed: {exc}")
            return False

    def _close_hinge_by_panel(
        self,
        fixture: Any,
        jn: str,
        geo: dict[str, Any],
        target: float,
        stroke_rad: float,
        max_strokes: int,
        horizontal_hinge: bool,
        max_regrasps: int = 4,
    ) -> tuple[float, int, int]:
        """Close a hinged door without its handle.  UNTESTED in sim as of 2026-08-27 (no
        certified closing of a hinged door yet; CloseMicrowave is blocked by reach).

        Vertical hinge (cabinet, microwave, fridge): push the panel's outer face near its
        free edge along the closing arc with the closed gripper, tool along the hinge axis.
        Horizontal hinge (oven, dishwasher, toaster oven): pinch the panel's free edge from
        the outside and swing it up, rotating the wrist with the door.
        """

        def fraction() -> float:
            return self._joint_fraction(fixture, jn)

        def reached(f: float) -> bool:
            return f <= max(target, 0.004)

        sense = -geo["open_sign"]
        panel = self._door_panel(geo)
        if panel is None:
            return fraction(), 0, 0
        strokes = regrasps = stalls = 0
        frac = fraction()
        grasped = False
        while not reached(frac) and strokes < max_strokes and regrasps <= max_regrasps:
            panel = self._door_panel(geo)
            ppos = panel["center"]
            r = ppos - geo["anchor"]
            r = r - geo["axis"] * (r @ geo["axis"])
            radius = float(np.linalg.norm(r))
            r_unit = r / max(radius, 1e-9)
            extent = float(np.sum(np.abs(panel["xmat"].T @ r_unit) * panel["size"]))
            n_out = self._outward(geo, ppos)  # outer face normal
            tangent = -n_out  # closing direction of the panel at ppos
            if horizontal_hinge:
                edge = geo["anchor"] + r_unit * (radius + extent - 0.02)
                quat = self.horizontal_grasp_quat(-r_unit, n_out, nearest=True)
                if not grasped:
                    self.set_gripper(closed=False, steps=8)
                    pre = edge + r_unit * 0.10
                    if not self._approach(pre, quat, r_unit):
                        break
                    try:
                        self.goto_pose(edge, quat, tolerance_m=0.012, max_steps=200)
                    except RuntimeError:
                        regrasps += 1
                        continue
                    self.set_gripper(closed=True)
                    grasped = True
                th = sense * stroke_rad
                tip = self.get_eef_pose()[:3]
                target_pos = geo["anchor"] + self._rodrigues(tip - geo["anchor"], geo["axis"], th)
                r2 = self._rodrigues(r_unit, geo["axis"], th)
                n2 = self._rodrigues(n_out, geo["axis"], th)
                quat2 = self.horizontal_grasp_quat(-r2, n2, nearest=True)
                before = frac
                with contextlib.suppress(RuntimeError):
                    self.goto_pose(
                        target_pos, quat2, tolerance_m=0.03, tolerance_rad=0.25, max_steps=150
                    )
                strokes += 1
                frac = fraction()
                if abs(frac - before) < 0.015:
                    stalls += 1
                    if stalls >= 2:
                        # re-grasp: release, back off, and let _approach reposition the base
                        self.set_gripper(closed=False, steps=8)
                        with contextlib.suppress(RuntimeError):
                            tip = self.get_eef_pose()
                            self.goto_pose(
                                tip[:3] + r_unit * 0.08, tip[3:7], tolerance_m=0.03, max_steps=100
                            )
                        grasped = False
                        regrasps += 1
                        stalls = 0
                else:
                    stalls = 0
            else:
                contact = geo["anchor"] + r_unit * (radius + 0.75 * extent)
                contact[2] = ppos[2]
                behind = contact - tangent * 0.06
                tool = np.array([0.0, 0.0, -1.0 if ppos[2] < self._shoulder()[2] + 0.35 else 1.0])
                quat = self.horizontal_grasp_quat(tool, tangent, nearest=True)
                self.set_gripper(closed=True, steps=6)
                if not grasped:
                    if not self._approach(behind + tool * -0.10, quat, -tangent):
                        break
                    try:
                        self.goto_pose(behind, quat, tolerance_m=0.02, max_steps=200)
                    except RuntimeError:
                        regrasps += 1
                        continue
                    grasped = True
                before = frac
                th = sense * stroke_rad
                tip = self.get_eef_pose()[:3]
                target_pos = geo["anchor"] + self._rodrigues(tip - geo["anchor"], geo["axis"], th)
                t2 = self._rodrigues(tangent, geo["axis"], th)
                quat2 = self.horizontal_grasp_quat(tool, t2, nearest=True)
                with contextlib.suppress(RuntimeError):
                    self.goto_pose(
                        target_pos, quat2, tolerance_m=0.03, tolerance_rad=0.25, max_steps=150
                    )
                strokes += 1
                frac = fraction()
                if abs(frac - before) < 0.015:
                    stalls += 1
                    if stalls >= 2:
                        with contextlib.suppress(RuntimeError):
                            tip = self.get_eef_pose()
                            self.goto_pose(
                                tip[:3] - tool * 0.10, tip[3:7], tolerance_m=0.03, max_steps=100
                            )
                        grasped = False
                        regrasps += 1
                        stalls = 0
                else:
                    stalls = 0
        self.set_gripper(closed=False, steps=6)
        with contextlib.suppress(Exception):
            tip = self.get_eef_pose()
            self.goto_pose(
                tip[:3] + np.array([0, 0, 0.10]), tip[3:7], tolerance_m=0.03, max_steps=120
            )
        return fraction(), strokes, regrasps

    def top_down_grasp_quat(self, opening_yaw: float, nearest: bool = True) -> np.ndarray:
        """Grip-site orientation (wxyz) pointing straight down with the finger-opening
        axis (site x) along world yaw ``opening_yaw`` (or ``opening_yaw + pi``, the same
        grasp, when ``nearest`` and that is the shorter wrist rotation from here).
        """
        x = np.array([np.cos(opening_yaw), np.sin(opening_yaw), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        if nearest:
            x = self._nearest_finger_axis(x, z)
        y = np.cross(z, x)
        return _wxyz_from_mat(np.column_stack([x, y, z]))

    def get_eef_pose(self) -> np.ndarray:
        """Grip-site pose ``[x, y, z, qw, qx, qy, qz]`` in the world frame (site z = tool axis)."""
        sid = self.sim.model.site_name2id(GRIP_SITE)
        pos = np.array(self.sim.data.site_xpos[sid], dtype=float)
        return np.concatenate([pos, _wxyz_from_mat(self.sim.data.site_xmat[sid])])

    def get_base_pose(self) -> np.ndarray:
        """Mobile-base pose ``[x, y, yaw]`` in the world frame."""
        import robosuite.utils.transform_utils as tu

        bid = self.sim.model.body_name2id(BASE_BODY)
        pos = np.array(self.sim.data.body_xpos[bid], dtype=float)
        yaw = tu.mat2euler(tu.quat2mat(_xyzw(self.sim.data.body_xquat[bid])))[2]
        return np.array([pos[0], pos[1], float(yaw)])

    def get_gripper_opening(self) -> float:
        """Distance between the finger pads, metres (≈0.009 closed on nothing, 0.084 open)."""
        geoms = self._robot.gripper[ARM].important_geoms
        left = self.sim.model.geom_name2id(geoms["left_fingerpad"][0])
        right = self.sim.model.geom_name2id(geoms["right_fingerpad"][0])
        return float(np.linalg.norm(self.sim.data.geom_xpos[left] - self.sim.data.geom_xpos[right]))

    def get_step_budget(self) -> dict[str, int]:
        """Environment steps used vs the task horizon (or ``max_steps``).

        The horizon is ADVISORY here: RoboCasa is built with ``ignore_done=True`` and
        nothing stops the physics at ``max`` (measured 2026-08-27: a probe ran to 15910
        against a max of 1000).  ``roboenv replay`` records ``budget_exhausted`` in the
        certificate and the door/drawer primitives report it too (they keep going: the
        benchmark's own success predicate is what ``completed`` means, an over-budget
        win is flagged on the board rather than counted).  Pace deliberately — a stroke
        costs ~100-200 frames.
        """
        return {"used": self._step_count, "max": int(self._max_steps or self.horizon)}

    def _budget_left(self) -> int:
        return int(self._max_steps or self.horizon) - int(self._step_count)

    # ------------------------------------------------------------ parts, torso, base pushes

    TORSO_JOINT = "mobilebase0_joint_torso_height"
    TORSO_RANGE_M: tuple[float, float] = (0.0, 0.34)
    #: top of the Omron pedestal (panda_omron.yaml: the pedestal box is 0.38 m tall) — a
    #: drop-down door whose panel bottom stands above it cannot be pushed by the base
    PEDESTAL_TOP_M: float = 0.38

    def task_fixture(self, hint: str | None = None) -> dict[str, Any]:
        """The fixture the benchmark spawns the robot in front of (``env.init_robot_base_ref``).

        With ``hint`` (a role such as ``"blender"``, a class such as ``"StandMixer"`` /
        ``"stand_mixer"``, or a name fragment such as ``"sink"``) the fixture is resolved the
        way every round-3 program did by hand: the ``task_fixtures()`` role first, then the
        fixture CLASS (case-insensitive, underscores ignored), then a name substring among
        the kitchen's fixtures (auxiliary parts last) — ``source`` says which matched, and a
        hint that matches nothing raises ``KeyError`` with the roles and classes available.

        For the atomic door/drawer/appliance tasks (Open*/Close*/TurnOn*/TurnOff*/Press*/
        Adjust*) that is the fixture the scorer reads.  It is NOT the scored fixture for
        every task: RoboCasa defines ``init_robot_base_ref`` as the fixture to place the
        robot near (kitchen.py), so for PickPlace tasks it is only one side — the SOURCE
        appliance for ``*ToCounter`` tasks, the target for ``CounterTo*`` tasks — and for
        NavigateKitchen it is the START fixture, not the scored ``target_fixture``
        (measured 2026-08-27: PickPlaceMicrowaveToCounter -> the microwave while the
        counter is scored; NavigateKitchen -> the source drawer stack).  Read the other
        roles from ``task_fixtures()``.  A string or FixtureType reference is resolved the
        way RoboCasa does (``env.get_fixture``).  Otherwise the registered ``fxtr``
        reference, then the only registered reference, then the first Fixture attribute
        of the environment (``source: "attribute"``, insertion order — arbitrary for a
        multi-fixture task).  Returns ``{"name", "role", "class", "source"}``; raises only
        when the environment has no ``init_robot_base_ref``, no ``fixture_refs`` and no
        Fixture attribute.
        """
        self._ensure_built()
        from robocasa.models.fixtures import Fixture

        refs = self.task_fixtures()
        if hint:
            key = str(hint).lower().replace("_", "").replace(" ", "")
            for role, name in refs.items():
                if role.lower().replace("_", "") == key:
                    return {
                        "name": str(name),
                        "role": role,
                        "class": self._class_of(name),
                        "source": "task_fixtures role",
                    }
            by_class = [
                n for n in self.env.fixtures if type(self.env.fixtures[n]).__name__.lower() == key
            ]
            by_class = [n for n in by_class if "auxiliary" not in n] or by_class
            if by_class:
                name = next((n for n in by_class if n in refs.values()), sorted(by_class)[0])
                role = next((k for k, v in refs.items() if v == name), "class")
                return {
                    "name": name,
                    "role": role,
                    "class": self._class_of(name),
                    "source": "fixture class",
                }
            by_name = [n for n in self.env.fixtures if key in n.lower().replace("_", "")]
            by_name = [n for n in by_name if "auxiliary" not in n] or by_name
            if by_name:
                name = next((n for n in by_name if n in refs.values()), sorted(by_name)[0])
                role = next((k for k, v in refs.items() if v == name), "name")
                return {
                    "name": name,
                    "role": role,
                    "class": self._class_of(name),
                    "source": "name substring",
                }
            classes = sorted({type(f).__name__ for f in self.env.fixtures.values()})
            raise KeyError(f"no fixture matches {hint!r}: roles {sorted(refs)}, classes {classes}")
        ref = getattr(self.env, "init_robot_base_ref", None)
        if ref is not None and not isinstance(ref, Fixture):
            try:
                ref = self.env.get_fixture(ref)
            except Exception:
                ref = None
        if isinstance(ref, Fixture) and ref.name in self.env.fixtures:
            role = next((k for k, v in refs.items() if v == ref.name), "init_robot_base_ref")
            return {
                "name": str(ref.name),
                "role": role,
                "class": type(ref).__name__,
                "source": "init_robot_base_ref",
            }
        registered = dict(self.episode_meta().get("fixture_refs") or {})
        if "fxtr" in registered:
            name = str(registered["fxtr"])
            return {
                "name": name,
                "role": "fxtr",
                "class": self._class_of(name),
                "source": "fixture_refs",
            }
        pool = registered or refs
        if not pool:
            raise RuntimeError(
                f"{self.task_name}: no task fixture (no init_robot_base_ref, no fixture_refs, "
                "no Fixture attribute) - use list_fixtures()"
            )
        role, name = next(iter(pool.items()))
        return {
            "name": str(name),
            "role": str(role),
            "class": self._class_of(str(name)),
            "source": "fixture_refs" if registered else "attribute (first Fixture attribute)",
        }

    def _class_of(self, name: str) -> str:
        fixture = self.env.fixtures.get(name)
        return type(fixture).__name__ if fixture is not None else "?"

    def joint_state(self, fixture: str, joint: str | None = None) -> dict[str, Any]:
        """A fresh read of one joint the way the scorer sees it.

        ``joint`` defaults to the fixture's first door joint.  Returns ``{"joint", "type",
        "fraction"`` (the benchmark's normalised value), ``"qpos", "open", "closed"`` (against
        the door/drawer thresholds), ``"handle_center", "open_direction"`` (fresh: a handle
        travels with its part), ``"axis", "anchor", "range", "door_fractions"`` (every door
        joint of the fixture)``}``.
        """
        fx = self._fixture(fixture)
        doors = self._default_door_joints(fx)
        jn = str(joint) if joint is not None else (doors[0] if doors else None)
        if jn is None:
            raise RuntimeError(f"{fixture}: no door joint; pass joint=<name>")
        if jn not in self.sim.model.joint_names:
            raise KeyError(f"{fixture}: joint {jn!r} is not in the model")
        geo = self._joint_geometry(jn)
        frac = self._joint_fraction(fx, jn)
        hi_th, lo_th = self._scorer_thresholds(fx, jn, doors)
        is_door = jn in doors
        handle = self._handle_for_joint(geo)
        door_fractions = {d: round(self._joint_fraction(fx, d), 4) for d in doors}
        door_th = self._scorer_thresholds(fx, doors[0], doors) if doors else (hi_th, lo_th)
        out: dict[str, Any] = {
            "joint": jn,
            "type": geo["type"],
            "fraction": round(frac, 4),
            "qpos": round(geo["qpos"], 4),
            # THIS joint against the thresholds; None for a joint the door thresholds do
            # not describe (a stove knob, a sink lever) — steer those on qpos/range
            "open": bool(frac > hi_th) if is_door or geo["type"] == "slide" else None,
            "closed": bool(frac < lo_th) if is_door or geo["type"] == "slide" else None,
            "is_door": is_door,
            "thresholds": [hi_th, lo_th],
            # the scorer's verdict for the FIXTURE: every default door joint at once
            "fixture_open": bool(all(f > door_th[0] for f in door_fractions.values()))
            if door_fractions
            else None,
            "fixture_closed": bool(all(f < door_th[1] for f in door_fractions.values()))
            if door_fractions
            else None,
            "axis": geo["axis"].round(4).tolist(),
            "anchor": geo["anchor"].round(4).tolist(),
            "range": [round(v, 4) for v in geo["range"]],
            "handle_center": None,
            "open_direction": None,
            "door_fractions": door_fractions,
        }
        if handle is not None:
            centre = np.asarray(handle["center"], dtype=float)
            out["handle_center"] = centre.round(4).tolist()
            out["open_direction"] = self._outward(geo, centre).round(4).tolist()
        return out

    def _scorer_thresholds(self, fixture: Any, jn: str, doors: list[str]) -> tuple[float, float]:
        """(open, closed) thresholds the benchmark applies to joint ``jn`` of ``fixture``.

        Doors 0.9 / 0.005 (``is_open``/``is_closed``); Drawer fixtures 0.95 / 0.05; a
        fridge's drawer joints 0.70 / 0.05 (``is_open(reg_type="drawer", th=0.70)``,
        measured on OpenFridgeDrawer / CloseFridgeDrawer 2026-08-27); other slide joints that
        are not door joints (dishwasher / oven racks) 0.95 / 0.05 (``get_state()["rack"]``).
        """
        if "Drawer" in type(fixture).__name__:
            return 0.95, 0.05
        if jn in (getattr(fixture, "_drawer_joint_names", None) or []):
            return 0.70, 0.05
        if jn not in doors and self._joint_geometry(jn)["type"] == "slide":
            return 0.95, 0.05
        return 0.9, 0.005

    def torso_height(self) -> float:
        """Height of the torso lift joint above its lowest position, metres (0 when the
        robot has no torso joint).
        """
        if not self._has_torso():
            return 0.0
        return float(self.sim.data.get_joint_qpos(self.TORSO_JOINT))

    def set_torso_height(
        self, height: float, *, max_steps: int = 80, tolerance_m: float = 0.004
    ) -> dict[str, Any]:
        """Raise or lower the arm's mounting torso to ``height`` metres above its lowest
        position (the Omron base's ``joint_torso_height``, 0 .. 0.34 m; RoboCasa resets it
        to 0).  The torso is a stiff position servo (kp 2000): a full 0.34 m lift takes
        ~22-30 control steps and raises the shoulder from z 1.03 to 1.37 (measured
        2026-08-27).  The arm rides with the torso (the eef rose 1.30 -> 1.63 m): release a
        held handle first.  A request outside the range is clamped and reported as such
        (``clamped``, ``reached`` False); a torso that stops moving with the error still
        open reports ``reason`` "stalled".  A request already within ``tolerance_m`` costs
        no steps.  Returns ``{"height", "requested", "clamped", "shoulder_z", "eef_z",
        "steps", "reached", "reason", "budget_exhausted"}``.
        """
        if not self._has_torso():
            raise RuntimeError("this robot has no torso lift joint")
        lo, hi = self.TORSO_RANGE_M
        requested = float(height)
        target = float(np.clip(requested, lo, hi))
        clamped = bool(abs(target - requested) > 1e-9)
        used = 0
        reason = ""
        recent: list[float] = []
        if abs(target - self.torso_height()) >= tolerance_m:
            while used < int(max_steps):
                if self._budget_left() <= 0:
                    reason = "budget exhausted"
                    break
                err = target - self.torso_height()
                if abs(err) < tolerance_m:
                    break
                self._step({"torso": np.array([float(np.clip(err * 25.0, -1.0, 1.0))])})
                used += 1
                recent.append(self.torso_height())
                # a torso that stops moving with the error still open is blocked (a wrist
                # under a counter overhang, a gripper closed on a fixed handle)
                if len(recent) >= 6 and abs(recent[-1] - recent[-6]) < 0.0005:
                    reason = f"stalled at {recent[-1]:.3f} m (blocked?)"
                    break
            self.hold(3)
        height_now = self.torso_height()
        reached = bool(abs(target - height_now) < tolerance_m) and not clamped
        if clamped and not reason:
            reason = f"requested {requested:.3f} m is outside {lo}..{hi}: clamped to {target:.3f}"
        bid = self.sim.model.body_name2id("robot0_link1")
        return {
            "height": round(height_now, 4),
            "requested": round(requested, 4),
            "clamped": clamped,
            "shoulder_z": round(float(self.sim.data.body_xpos[bid][2]), 4),
            "eef_z": round(float(self.get_eef_pose()[2]), 4),
            "steps": used,
            "reached": reached,
            "reason": reason,
            "budget_exhausted": self._budget_left() <= 0,
        }

    def _shoulder(self) -> np.ndarray:
        bid = self.sim.model.body_name2id("robot0_link1")
        return np.asarray(self.sim.data.body_xpos[bid], dtype=float)

    def stand_at(
        self,
        target: Any,
        approach_dir: Any | None = None,
        *,
        max_candidates: int = 3,
        prefer_m: float = 0.72,
    ) -> dict[str, Any]:
        """Tuck the arm and drive the base to a spot the arm can work ``target`` from.

        The candidates are :meth:`find_standing_pose` (0.5-0.8 m from the target along
        directions fanned around ``approach_dir``, footprint clear, target within reach);
        they are driven to in order until ``drive_to`` reports the pose reached (the arm
        is held at its tucked base-relative pose while driving).  The candidates read the
        LIVE shoulder height, so a torso lift counts.  A 2-D target raises.  Worst case
        ~1060 frames (tuck 160 + three 300-step drives).  Returns ``{"reached", "pose",
        "candidate", "tried", "drive", "tuck", "shoulder_dist", "steps",
        "budget_exhausted"}``; ``reached`` False with a ``reason`` when no candidate clears
        or none could be driven to (obstructed, or out of steps).
        """
        tgt = np.asarray(target, dtype=float).reshape(-1)
        if tgt.shape != (3,):
            raise ValueError("stand_at: target must be a 3-D world point [x, y, z]")
        s0 = self._step_count
        cands = self.find_standing_pose(
            tgt, approach_dir, max_candidates=max_candidates, prefer_m=prefer_m
        )
        if not cands:
            return {
                "reached": False,
                "pose": self.get_base_pose().round(3).tolist(),
                "tried": 0,
                "reason": "no standing pose: no 0.5-0.8 m spot around approach_dir has the "
                "footprint clear and the target within the arm's reach",
                "steps": 0,
                "budget_exhausted": self._budget_left() <= 0,
            }
        tuck = self.tuck_arm()
        drive: dict[str, Any] = {}
        for k, cand in enumerate(cands):
            x, y, yaw = cand["pose"]
            drive = self.drive_to(x, y, yaw, max_steps=300)
            _trace(f"stand_at candidate {cand}: {drive}")
            if drive.get("reached"):
                return {
                    "reached": True,
                    "pose": self.get_base_pose().round(3).tolist(),
                    "candidate": cand,
                    "tried": k + 1,
                    "drive": drive,
                    "tuck": {k2: tuck.get(k2) for k2 in ("reached", "reason") if k2 in tuck},
                    "shoulder_dist": round(self._shoulder_distance(tgt), 3),
                    "steps": self._step_count - s0,
                    "budget_exhausted": self._budget_left() <= 0,
                }
        if drive.get("blocked"):
            reason = "every candidate drive was obstructed (drive_to blocked)"
        else:
            reason = (
                f"every candidate drive ran out of steps (last error_m "
                f"{drive.get('error_m')}, {drive.get('steps')} steps)"
            )
        return {
            "reached": False,
            "pose": self.get_base_pose().round(3).tolist(),
            "tried": len(cands),
            "drive": drive,
            "reason": reason,
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }

    def _geom_outward_normal(self, gid: int) -> tuple[np.ndarray, np.ndarray, str]:
        """(centre, outward unit normal, how the sign was decided) of a fixture geom.

        The normal is the geom's thinnest axis (a capsule's/cylinder's own axis).  Its sign
        comes, in order, from: the slide joint that moves the geom's body (a button's
        press axis is definitive — the kettle switch, whose body root sits under it,
        used to be undetermined); else the direction from the fixture's collision-box
        centre to the geom in full 3-D (a button on the front panel faces the room, a
        cap on top faces up); else the old horizontal-offset rule.
        """
        raw_m, raw_d = self._raw()
        xmat = np.array(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
        gtype = int(raw_m.geom_type[gid])
        size = np.array(raw_m.geom_size[gid], dtype=float)
        centre = np.array(raw_d.geom_xpos[gid], dtype=float)
        if gtype == 7:
            aabb = np.array(raw_m.geom_aabb[gid], dtype=float).reshape(2, 3)
            size = aabb[1]
            centre = centre + xmat @ aabb[0]
        if gtype in (3, 5):  # capsule / cylinder: the axis is the press direction
            n = xmat[:, 2].copy()
        elif gtype == 2:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = xmat[:, int(np.argmin(size))].copy()
        body = int(raw_m.geom_bodyid[gid])
        root = int(raw_m.body_rootid[body])
        source = "geometry"
        # 1. a slide joint on the geom's body (or an ancestor inside the fixture) presses
        #    along its own axis: outward = against the fraction-raising sense
        k = body
        while k > 0 and k != root:
            for jid in range(int(raw_m.njnt)):
                if int(raw_m.jnt_bodyid[jid]) == k and int(raw_m.jnt_type[jid]) == 2:
                    jn = self.sim.model.joint_id2name(jid)
                    geo = self._joint_geometry(jn)
                    n = -geo["axis"] * geo["open_sign"]
                    return centre, n / max(np.linalg.norm(n), 1e-9), f"slide joint {jn}"
            k = int(raw_m.body_parentid[k])
        # 2. away from the fixture's collision-box centre, in 3-D
        members = [
            g
            for g in range(int(raw_m.ngeom))
            if int(raw_m.body_rootid[int(raw_m.geom_bodyid[g])]) == root
            and (raw_m.geom_contype[g] or raw_m.geom_conaffinity[g])
        ]
        if len(members) > 1:
            pts = np.array([raw_d.geom_xpos[g] for g in members], dtype=float)
            box_c = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            away = centre - box_c
            if abs(float(n @ away)) > 1e-6:
                if float(n @ away) < 0.0:
                    n = -n
                return centre, n / max(np.linalg.norm(n), 1e-9), "fixture box centre"
        # 3. the horizontal offset from the body root (fixtures are placed at their base)
        body_c = np.array(raw_d.xpos[root], dtype=float)
        away = centre - body_c
        away[2] = 0.0 if np.linalg.norm(away[:2]) > 0.02 else away[2]
        if abs(float(n @ away)) < 1e-6:
            source = "undetermined (geom axis sign)"
        elif float(n @ away) < 0.0:
            n = -n
        return centre, n / max(np.linalg.norm(n), 1e-9), source

    def press_part(
        self,
        fixture: str,
        part: str,
        *,
        increase: bool = True,
        depths: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03),
        pre_m: float = 0.06,
        retreat_m: float = 0.12,
        fraction_gain: float = 0.15,
    ) -> dict[str, Any]:
        """Press a button / lever / cap with the closed fist and come away again.

        ``part`` is a slide-joint name of the fixture (pressed along its own axis in the
        sense that raises the scorer's fraction, or lowers it with ``increase=False``; the
        contact point is the joint's handle/lip centre from ``get_fixture_joints``) or a
        collision geom name (a microwave's ``*start_button``: pressed along the geom's
        inward normal at its centre).  Position-only servoing (the wrist is what runs out of
        envelope at appliance height, not the reach): pre-point ``pre_m`` before the part,
        then the centre and ``depths`` past it until the joint fraction rose by
        ``fraction_gain`` (joint) or the gripper touched the geom (geom), then a retreat of
        ``retreat_m`` along the press normal (the microwave scorer wants the grip site
        > 0.15 m from the button; the blender toggles on the RELEASE of the contact).
        Measured and certified on flush button GEOMS of a vertical panel (the microwave
        start/stop buttons); the joint path is verified on the kettle lid button only —
        levers with a long travel (a toaster) and diagonal buttons (a blender) are NOT
        solved by it as of 2026-08-27.  ``part`` must belong to ``fixture``.  Returns
        ``{"kind", "center", "press_dir", "normal_source", "oriented", "fraction_before",
        "fraction_after" (before the retreat), "fraction_peak", "fraction_final" (after
        the retreat), "disturbed_by_approach", "touched", "depth_reached", "retreat_m",
        "reached_pre", "reached_contact", "ok", "reason", "steps", "budget_exhausted"}``.
        """
        fx = self._fixture(fixture)
        if not str(part).startswith(str(fx.name) + "_"):
            raise KeyError(
                f"{fixture}: {part!r} is not a part of this fixture (parts are named "
                f"{fx.name}_...; see get_fixture_parts)"
            )
        s0 = self._step_count
        kind: str
        normal_source = ""
        gripper_geoms = list(self._robot.gripper[ARM].contact_geoms)
        raw_m, _raw_d = self._raw()
        sense = 1.0 if increase else -1.0
        if part in self.sim.model.joint_names:
            geo = self._joint_geometry(part)
            if geo["type"] != "slide":
                raise RuntimeError(
                    f"{part}: press_part takes slide joints (got {geo['type']}); "
                    "use turn_knob for hinges"
                )
            handle = self._handle_for_joint(geo)
            centre = (
                np.array(handle["center"], dtype=float) if handle is not None else geo["anchor"]
            )
            press = geo["axis"] * geo["open_sign"] * sense
            kind = "joint"
            normal_source = f"slide joint {part}"
            part_bodies = self._subtree_bodies(geo["body"])
            part_geoms = [
                self.sim.model.geom_id2name(g)
                for g in range(int(raw_m.ngeom))
                if int(raw_m.geom_bodyid[g]) in part_bodies
                and (raw_m.geom_contype[g] or raw_m.geom_conaffinity[g])
                and self.sim.model.geom_id2name(g)
            ]

            def fraction() -> float | None:
                return float(self._joint_fraction(fx, part))

            def touched() -> bool:
                # gripper contact with any collision geom the joint moves (used to be a
                # stub returning False for joints)
                return bool(part_geoms) and bool(self.env.check_contact(gripper_geoms, part_geoms))

        elif part in self.sim.model.geom_names:
            gid = self.sim.model.geom_name2id(part)
            centre, n, normal_source = self._geom_outward_normal(gid)
            press = -n
            kind = "geom"

            def fraction() -> float | None:
                return None

            def touched() -> bool:
                return bool(self.env.check_contact(gripper_geoms, part))

        else:
            raise KeyError(f"{fixture}: {part!r} is neither a joint nor a geom of the model")
        press = press / max(np.linalg.norm(press), 1e-9)
        f0 = fraction()
        self.set_gripper(closed=True, steps=8)
        pre = centre - press * pre_m
        # a sideways press (a button on a vertical panel) wants the fingertips pointed at it:
        # the finger's side face pushed the surrounding panel 1.9 cm short of a flush
        # microwave button, the oriented fingertips pressed it (measured 2026-08-27); a
        # downward press keeps the wrist free (the kettle button stalled 4.5 cm short with a
        # top-down quaternion and was pressed position-only)
        quat: np.ndarray | None = None
        if abs(float(press[2])) < 0.7:
            quat = self.horizontal_grasp_quat(press, np.array([0.0, 0.0, 1.0]), nearest=True)
        reached_pre = False
        with contextlib.suppress(RuntimeError):
            reached_pre = bool(
                self.goto_pose(pre, quat, tolerance_m=0.02, max_steps=200)["reached"]
            )
        if not reached_pre and self._shoulder_distance(centre) > 0.6:
            stand = self.stand_at(centre, -press)
            _trace(f"press_part: repositioned {stand}")
            with contextlib.suppress(RuntimeError):
                reached_pre = bool(
                    self.goto_pose(pre, quat, tolerance_m=0.02, max_steps=200)["reached"]
                )
        if not reached_pre and quat is not None:
            quat = None  # the oriented wrist ran out of envelope: press with the wrist free
            with contextlib.suppress(RuntimeError):
                reached_pre = bool(
                    self.goto_pose(pre, None, tolerance_m=0.02, max_steps=200)["reached"]
                )
        f_pre = fraction()
        disturbed = bool(f_pre is not None and f0 is not None and abs(f_pre - f0) > 0.05)
        hit = False
        depth_reached = 0.0
        reached_contact = False
        f_peak = f0
        for d in depths:
            if self._budget_left() <= 0:
                break
            rep = None
            with contextlib.suppress(RuntimeError):
                rep = self.goto_pose(centre + press * d, quat, tolerance_m=0.006, max_steps=80)
            if rep is not None and d == 0.0:
                reached_contact = bool(rep.get("reached"))
            depth_reached = float((self.get_eef_pose()[:3] - centre) @ press)
            f1 = fraction()
            if f1 is not None and f0 is not None and sense * (f1 - f_peak) > 0:
                f_peak = f1
            # the stop rule follows the requested SENSE: increase=False used to test for a
            # rise and push a drawer 3 cm past its stop
            gained = f1 is not None and f0 is not None and sense * (f1 - f0) >= fraction_gain
            hit = touched() or gained
            _trace(f"press {part} depth {d}: fraction {f1} touched {hit}")
            if hit:
                break
        f_after = fraction()
        # come away until the grip site really is retreat_m from the part (the microwave
        # scorer wants > 0.15 m): a single goto stopped 9.6 cm out on a 12 cm request
        for extra in (0.0, 0.06, 0.12):
            with contextlib.suppress(RuntimeError):
                self.goto_pose(
                    centre - press * (retreat_m + extra), None, tolerance_m=0.02, max_steps=120
                )
            if float(np.linalg.norm(self.get_eef_pose()[:3] - centre)) >= retreat_m:
                break
        self.hold(5)
        f_final = fraction()  # after the retreat: a drawer can rebound off its stop
        retreat = float(np.linalg.norm(self.get_eef_pose()[:3] - centre))
        ok = bool(hit and retreat >= retreat_m)
        if kind == "joint" and f_final is not None and f_after is not None:
            ok = ok and abs(f_final - f_after) < 0.05
        reason = ""
        if not hit:
            reason = "no contact / no fraction change at any depth"
        elif retreat < retreat_m:
            reason = f"retreat stopped at {retreat:.3f} m of {retreat_m}"
        elif not ok:
            reason = f"the part rebounded after the retreat ({f_after} -> {f_final})"
        return {
            "kind": kind,
            "center": centre.round(4).tolist(),
            "press_dir": press.round(4).tolist(),
            "normal_source": normal_source,
            "oriented": quat is not None,
            "fraction_before": None if f0 is None else round(f0, 4),
            "fraction_after": None if f_after is None else round(f_after, 4),
            "fraction_peak": None if f_peak is None else round(f_peak, 4),
            "fraction_final": None if f_final is None else round(f_final, 4),
            "disturbed_by_approach": disturbed,
            "touched": bool(hit),
            "depth_reached": round(depth_reached, 4),
            "retreat_m": round(retreat, 4),
            "reached_pre": reached_pre,
            "reached_contact": reached_contact,
            "ok": ok,
            "reason": reason,
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }

    def close_door_with_base(
        self,
        fixture: str,
        joint: str | None = None,
        *,
        step_m: float = 0.10,
        max_pushes: int = 8,
        target_fraction: float = 0.0,
    ) -> dict[str, Any]:
        """Close a heavy drop-down door by driving the BASE into it.

        The arm's OSC delivers a few tens of newtons; a dishwasher door needs ~40 Nm at its
        hinge (22 kg) and an oven door ~21 Nm.  The Omron pedestal (0.38 m tall, 600 N of
        drive) can push a door whose panel reaches down to pedestal height: the base is
        aligned with the door panel's centre line, headed along the closing direction
        (minus the handle's ``open_direction``, flattened), and driven forward in
        ``step_m`` increments with the arm tucked and held, reading the scorer's fraction
        after each, until the door reads closed (``target_fraction``, the dishwasher
        scorer's 0.005) or the base stops moving with the fraction unchanged; then backs
        off 0.25 m and re-reads (a door can rebound).  Measured 2026-08-27: the dishwasher
        door 1.00 -> 0.00 in 5 pushes, 165 frames (the certified CloseDishwasher).
        Refuses BEFORE moving, with a ``reason`` and 0 frames, a vertical-hinge door (use
        ``push_door_shut``: driven into the open fridge door it rammed it further open),
        a panel whose bottom stands above the pedestal top (``PEDESTAL_TOP_M`` 0.38: the
        oven at 66 deg), a door open past ~72 deg (vertical outward normal), and an
        already-closed door (``done`` True, ``reason`` "already closed"); a line-up drive
        that is not reached stops it too.  Returns ``{"done", "fraction", "pushes",
        "history", "base_travel_m", "panel_bottom_z", "steps", "budget_exhausted"}``
        (+ ``reason``).
        """
        fx = self._fixture(fixture)
        doors = self._default_door_joints(fx)
        jn = joint or (doors[0] if doors else None)
        if jn is None:
            raise RuntimeError(f"{fixture}: no door joint")
        geo = self._joint_geometry(jn)
        if geo["type"] != "hinge" or abs(float(geo["axis"][2])) > 0.7:
            raise RuntimeError(
                f"{jn}: close_door_with_base is for hinged DROP-DOWN doors (horizontal "
                f"hinge); got {geo['type']} with axis {geo['axis'].round(2).tolist()} — "
                "use push_door_shut for a vertical hinge"
            )

        def fraction() -> float:
            return float(self._joint_fraction(fx, jn))

        panel = self._door_panel(geo)
        handle = self._handle_for_joint(geo)
        ref = np.asarray((handle or panel)["center"], dtype=float)
        n = self._outward(geo, ref)
        pc = np.asarray(panel["center"], dtype=float) if panel else ref
        z_ext = (
            float(np.sum(np.abs(panel["xmat"].T @ np.array([0, 0, 1.0])) * panel["size"]))
            if panel
            else 0.0
        )
        panel_bottom = float(pc[2] - z_ext)
        s0 = self._step_count
        f = fraction()

        def refusal(reason: str, done: bool = False) -> dict[str, Any]:
            return {
                "done": done,
                "fraction": round(f, 4),
                "pushes": 0,
                "history": [],
                "base_travel_m": 0.0,
                "panel_bottom_z": round(panel_bottom, 3),
                "reason": reason,
                "steps": self._step_count - s0,
                "budget_exhausted": self._budget_left() <= 0,
            }

        if f <= max(target_fraction, 0.004):
            return refusal("already closed", done=True)
        # the applicability check the docstring used to leave to the caller: a panel that
        # stands above the pedestal cannot be pushed (the oven at 66 deg: the base drove
        # 0.61 m under the open door and jammed, measured 2026-08-27)
        if panel_bottom > self.PEDESTAL_TOP_M:
            return refusal(
                f"panel bottom z {panel_bottom:.3f} is above the pedestal top "
                f"({self.PEDESTAL_TOP_M} m): the base cannot reach it"
            )
        push = -n.copy()
        push[2] = 0.0
        if np.linalg.norm(push) < 0.3:
            return refusal(
                f"the door's outward normal {n.round(2).tolist()} is vertical (open past "
                "~72 deg): nothing to drive into"
            )
        push /= np.linalg.norm(push)
        yaw = float(np.arctan2(push[1], push[0]))
        history: list[str] = []
        tuck = self.tuck_arm()
        history.append(f"tuck: reached {tuck.get('reached')}")
        base0 = self.get_base_pose()
        # line up 0.45 m before the panel centre (the pedestal front is 0.15 m ahead of the body)
        start = pc[:2] - push[:2] * 0.45
        rep = self.drive_to(float(start[0]), float(start[1]), yaw, max_steps=300, hold_arm=True)
        history.append(f"line up {np.round(start, 2).tolist()}: reached {rep.get('reached')}")
        if not rep.get("reached") and float(rep.get("error_m", 1.0)) > 0.06:
            out = refusal(
                f"line-up drive not reached (error_m {rep.get('error_m')}, "
                f"blocked {rep.get('blocked')})"
            )
            out["history"] = history
            out["base_travel_m"] = round(
                float(np.linalg.norm(self.get_base_pose()[:2] - base0[:2])), 3
            )
            return out
        f = fraction()
        pushes = 0
        stalled_after_contact = 0
        travelled = 0.0
        contact_seen = False
        while f > max(target_fraction, 0.004) and pushes < max_pushes and self._budget_left() > 0:
            bx, by, _ = self.get_base_pose()
            goal = np.array([bx, by]) + push[:2] * step_m
            before = f
            b0 = np.array([bx, by])
            rep = self.drive_to(
                float(goal[0]), float(goal[1]), yaw, max_steps=150, hold_arm=True, tolerance_m=0.02
            )
            pushes += 1
            self.hold(5)
            f = fraction()
            moved = float(np.linalg.norm(np.array(self.get_base_pose()[:2]) - b0))
            travelled += moved
            history.append(f"push {pushes}: base +{moved:.3f} m, fraction {before:.3f} -> {f:.3f}")
            _trace(history[-1])
            if abs(f - before) >= 0.005:
                contact_seen = True
                stalled_after_contact = 0
            elif contact_seen:
                stalled_after_contact += 1
            if moved < 0.01 and abs(f - before) < 0.005:
                break
            # the base keeps travelling but the door no longer follows: it slid past the
            # panel (CloseFridge: 8 pushes of +0.09 m at fraction 0.638, measured)
            if stalled_after_contact >= 2 or travelled > 0.45 + 2.0 * z_ext + 0.1:
                history.append("stop: base moving, fraction not")
                break
        bx, by, _ = self.get_base_pose()
        back = np.array([bx, by]) - push[:2] * 0.25
        self.drive_to(float(back[0]), float(back[1]), yaw, max_steps=150, hold_arm=True)
        self.hold(10)
        f = fraction()
        return {
            "done": bool(f <= max(target_fraction, 0.004)),
            "fraction": round(f, 4),
            "pushes": pushes,
            "history": history,
            "base_travel_m": round(float(np.linalg.norm(self.get_base_pose()[:2] - base0[:2])), 3),
            "panel_bottom_z": round(panel_bottom, 3),
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }

    def push_door_shut(
        self,
        fixture: str,
        joint: str | None = None,
        *,
        stroke_rad: float = 0.4,
        max_strokes: int = 10,
        contact_offset_m: float = 0.025,
        torso: float | None = None,
        target_fraction: float = 0.0,
    ) -> dict[str, Any]:
        """Close a vertical-hinge door (cabinet, microwave, fridge) by pushing its handle
        bar along the hinge arc with the closed fist, position-only.

        Leaner than ``close_door``'s push-handle path (which rotates the wrist with the
        panel and stalls when the orientation runs out: CloseMicrowave 1438 frames): the
        contact point is the bar's centre plus ``contact_offset_m`` along the panel's
        CURRENT outer normal, every stroke re-reads the bar and rotates that point about
        the hinge by up to ``stroke_rad`` (clamped to the travel left), and the wrist is
        left free.  When the approach point is beyond the arm (> 0.78 m from the shoulder)
        or the next stroke target is (> 0.85 m), or a stroke moved nothing twice, the hand
        retreats and the base is re-stood for the bar's current position (``stand_at``).
        ``torso`` (metres, default: auto = lift so the bar sits ~0.25 m above the
        shoulder, clamped to 0..0.34 — a LOW bar such as the fridge's at z 1.07 leaves it
        at 0) raises the arm for wall cabinets; the arm is tucked relative to the lifted
        torso, and the torso STAYS where it was lifted.  Strokes stop at the horizon.
        Measured 2026-08-27: microwave 0.99 -> 0.0 in 4 strokes; a wall-cabinet door
        beside the shoulder (bar z 1.57) 0.97 -> 0.0025 in 5 strokes with the torso at
        0.34; the fridge door 0.908 -> 0.0 in 3 strokes, 677 frames (round-3 review).
        Returns ``{"done", "fraction", "strokes", "repositions", "torso", "contact"
        ("handle" | "panel"), "history", "steps", "budget_exhausted"}`` on every path;
        ``reason`` names a refusal or why it stopped short.
        """
        fx = self._fixture(fixture)
        doors = self._default_door_joints(fx)
        jn = joint or (doors[0] if doors else None)
        if jn is None:
            raise RuntimeError(f"{fixture}: no door joint")
        geo = self._joint_geometry(jn)
        if geo["type"] != "hinge" or abs(geo["axis"][2]) < 0.7:
            raise RuntimeError(
                f"{jn}: push_door_shut is for vertical-hinge doors (got "
                f"{geo['type']}, axis {geo['axis'].round(2).tolist()})"
            )
        s0 = self._step_count
        rod = self._rodrigues
        sign = geo["open_sign"]

        def fraction() -> float:
            return float(self._joint_fraction(fx, jn))

        def reached(f: float) -> bool:
            return f <= max(target_fraction, 0.004)

        def contact_point() -> tuple[np.ndarray, np.ndarray] | None:
            handle = self._handle_for_joint(geo)
            if handle is not None:
                c = np.asarray(handle["center"], dtype=float)
                n = self._outward(geo, c)
                return c + n * contact_offset_m, n
            panel = self._door_panel(geo)
            if panel is None:
                return None
            pc = np.asarray(panel["center"], dtype=float)
            r = pc - geo["anchor"]
            r -= geo["axis"] * (r @ geo["axis"])
            r_unit = r / max(np.linalg.norm(r), 1e-9)
            extent = float(np.sum(np.abs(panel["xmat"].T @ r_unit) * panel["size"]))
            edge = geo["anchor"] + r_unit * (np.linalg.norm(r) + extent - 0.04)
            edge[2] = pc[2]
            n = self._outward(geo, edge)
            return edge + n * (contact_offset_m + 0.01), n

        f = fraction()
        base_report: dict[str, Any] = {
            "strokes": 0,
            "repositions": 0,
            "torso": round(self.torso_height(), 3),
            "contact": "handle" if self._handle_for_joint(geo) is not None else "panel",
            "history": [],
            "steps": 0,
            "budget_exhausted": self._budget_left() <= 0,
        }
        if reached(f):
            return {"done": True, "fraction": round(f, 4), "mode": "noop", **base_report}
        cp = contact_point()
        if cp is None:
            return {
                "done": False,
                "fraction": round(f, 4),
                "reason": f"{jn}: no handle and no panel geom under the joint body",
                **base_report,
            }
        point, n = cp
        history: list[str] = []
        if torso is None:
            torso = float(
                np.clip(
                    point[2] - 0.25 - self._shoulder()[2] + self.torso_height(), *self.TORSO_RANGE_M
                )
            )
        if abs(torso - self.torso_height()) > 0.01:
            t = self.set_torso_height(torso)
            history.append(f"torso {t['height']:.3f} (shoulder z {t['shoulder_z']:.3f})")
        repositions = 0
        strokes = stalls = 0
        self.set_gripper(closed=True, steps=8)
        radial = point - geo["anchor"]
        radial -= geo["axis"] * float(radial @ geo["axis"])
        door_radius = float(np.linalg.norm(radial))
        # the preferred standoff grows with the door: 0.55 m for a 0.3 m microwave door
        # (the certified stance; 0.7 re-stood the base after every stroke), 0.72 for the
        # 0.69 m fridge door (0.55 rammed it open, 0.8 left every stroke target beyond the
        # arm — measured 2026-08-27)
        prefer = float(np.clip(max(0.55, door_radius + 0.12), 0.5, 0.72))
        history.append(f"door radius {door_radius:.2f} m -> preferred standoff {prefer:.2f} m")

        moved = [False]

        def ahead_of_base(point: np.ndarray) -> bool:
            bx, by, byaw = self.get_base_pose()
            to = point[:2] - np.array([bx, by])
            return float(to @ np.array([np.cos(byaw), np.sin(byaw)])) > 0.5 * np.linalg.norm(to)

        def span_mid() -> tuple[np.ndarray, np.ndarray]:
            # the middle of the next <= 0.8 rad of travel: one standing spot serves two strokes
            point, n = contact_point()
            th = -sign * min(0.8, self._remaining_stroke(geo, target_fraction, 0.8)) / 2.0
            mid = geo["anchor"] + rod(point - geo["anchor"], geo["axis"], th)
            return mid, rod(n, geo["axis"], th)

        def approach(force_move: bool = False) -> bool:
            point, n = contact_point()
            far = self._shoulder_distance(point) > 0.78 or not ahead_of_base(point)
            if (far or force_move) and not moved[0]:
                mid, n_mid = span_mid()
                # an arc wants the shoulder close to the bar (~0.55 m: the certified
                # microwave stance) so the next stroke's target stays within reach — the
                # 0.72 m drawer preference re-stood the base after EVERY stroke once the
                # shoulder height was read live — but OUTSIDE the disc the door sweeps
                # about its hinge: at 0.55 m from a 0.6 m fridge door the base rammed it
                # fully open (measured 2026-08-27, both)
                st = self.stand_at(mid, n_mid, prefer_m=prefer)
                history.append(
                    f"stand_at {st.get('pose')}: {st.get('reached')} "
                    f"[{self._step_count - s0} steps]"
                )
                moved[0] = bool(st.get("reached"))
                self.set_gripper(closed=True, steps=4)
                point, n = contact_point()
            err = None
            toward = self._shoulder() - point
            toward[2] = 0.0
            toward /= max(np.linalg.norm(toward), 1e-9)
            with contextlib.suppress(RuntimeError):
                self.goto_pose(
                    point + n * 0.06 + toward * 0.06, None, tolerance_m=0.02, max_steps=250
                )
            with contextlib.suppress(RuntimeError):
                err = float(
                    self.goto_pose(point, None, tolerance_m=0.012, max_steps=250)["error_m"]
                )
            ok = err is not None and err <= 0.03
            history.append(
                f"approach {np.round(point, 3).tolist()}: "
                f"{'miss' if err is None else round(err, 3)} -> {ok} "
                f"[{self._step_count - s0} steps]"
            )
            return ok

        approached = approach()
        if not approached:
            repositions += 1
            approached = approach(force_move=True)
        reason = None
        while not reached(f) and strokes < max_strokes and repositions <= 3:
            if self._budget_left() <= 0:
                reason = f"{jn}: budget exhausted after {strokes} strokes"
                break
            point, n = contact_point()
            th = -sign * self._remaining_stroke(geo, target_fraction, stroke_rad)
            tgt = geo["anchor"] + rod(point - geo["anchor"], geo["axis"], th)
            if not approached:
                reason = f"{jn}: the bar could not be approached (see history)"
                break
            # a stroke whose end is beyond the arm is shortened before the base is moved:
            # halve it down to 0.15 rad and re-stand only when even that is out of reach
            while self._shoulder_distance(tgt) > 0.85 and abs(th) > 0.15:
                th *= 0.5
                tgt = geo["anchor"] + rod(point - geo["anchor"], geo["axis"], th)
            if self._shoulder_distance(tgt) > 0.85:
                with contextlib.suppress(RuntimeError):
                    tip = self.get_eef_pose()
                    self.goto_pose(tip[:3] + n * 0.10, None, tolerance_m=0.03, max_steps=80)
                repositions += 1
                moved[0] = False
                approached = approach(force_move=True)
                continue
            before = f
            try:
                self.goto_pose(tgt, None, tolerance_m=0.015, max_steps=120)
            except RuntimeError as exc:
                history.append(f"stroke {strokes + 1}: goto raised {str(exc)[:60]}")
            strokes += 1
            self.hold(3)
            f = fraction()
            history.append(
                f"stroke {strokes} th {th:+.2f}: {before:.3f} -> {f:.3f} "
                f"[{self._step_count - s0} steps]"
            )
            _trace(history[-1])
            if abs(f - before) < 0.01:
                stalls += 1
                if stalls >= 2:
                    stalls = 0
                    repositions += 1
                    with contextlib.suppress(RuntimeError):
                        tip = self.get_eef_pose()
                        self.goto_pose(tip[:3] + n * 0.10, None, tolerance_m=0.03, max_steps=80)
                    moved[0] = False
                    approached = approach(force_move=True)
            else:
                stalls = 0
        if reached(f):
            self.hold(10)
            f = fraction()
        with contextlib.suppress(RuntimeError):
            point, n = contact_point()
            tip = self.get_eef_pose()
            self.goto_pose(tip[:3] + n * 0.10, None, tolerance_m=0.03, max_steps=80)
        f = fraction()
        if not reached(f) and reason is None:
            if strokes >= max_strokes:
                reason = f"{jn}: max_strokes ({max_strokes}) reached at fraction {f:.3f}"
            elif repositions > 3:
                reason = f"{jn}: repositions exhausted at fraction {f:.3f} (see history)"
        out = {
            "done": bool(reached(f)),
            "fraction": round(f, 4),
            "strokes": strokes,
            "repositions": repositions,
            "torso": round(self.torso_height(), 3),
            "contact": base_report["contact"],
            "history": history,
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }
        if reason:
            out["reason"] = reason
        return out

    # ------------------------------------------------------------ goal, instruction, meta

    def instruction(self) -> str:
        """The episode's language instruction (``ep_meta["lang"]``) - the only place the
        benchmark names a direction (TurnSinkSpout left/right), a knob (TurnOnStove "front
        left"), a rack ("top"/"bottom") or a pull/push sense.
        """
        return self.goal_text()

    def episode_meta_public(self) -> dict[str, Any]:
        """RoboCasa's episode metadata minus the bulky scene dumps: ``lang``, ``layout_id``,
        ``style_id``, ``fixture_refs``, ``task_refs`` (a stove task's ``knob`` /
        ``cookware_burner``, a sink task's ``behavior``), ``refs``, ``should_pull``,
        ``rack_level``, ``initial_temp`` ... - everything the task class stored, verbatim.
        """
        meta = self.episode_meta()
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if k in ("cam_configs", "fixtures", "gen_textures", "object_cfgs"):
                continue
            out[k] = _plain(v)
        return out

    def goal(self) -> dict[str, Any]:
        """What the benchmark scores for this episode, read from the task class itself.

        Returns ``{"instruction", "task", "behavior", "decoded", "satisfied" (a LIVE
        ``_check_success``), "predicates": [...], "fixtures": {role: name},
        "objects": [...], "meta": {...}}``.  The decoder reads the SOURCE of the episode's
        own ``_check_success`` and emits a predicate only for what that body actually asks
        for, so a fixture the env merely carries never becomes a predicate.  Four kinds are
        decoded today:

        * ``joint_fraction`` — ``{"kind", "fixture", "joints", "test", "thresholds",
          "want", "value"}``: the door/drawer fractions with the scorer's own thresholds.
        * ``knob_qpos`` — ``{"kind", "fixture", "joint", "knob", "test", "target_qpos",
          "value" (the RAW joint position, which is what ``turn_knob`` takes),
          "value_mod_2pi" (what ``Stove.get_knobs_state`` reports and the scorer tests —
          a knob resting a hair below 0 reads 6.283 there, not 0)}``.
        * ``fixture_state`` — ``{"kind", "fixture", "key", "test", "value"}``: an
          appliance's ``get_state()`` key (``turned_on``, ``lid_on_blender``, ``lid``,
          ``head``, ``time``, ``temperature``) that the scorer reads.
        * ``gripper_far`` — ``{"kind", "fixture", "part", "test"}`` (no ``value``): the
          "let go and stand back" term of a button press or the blender lid.

        Anything else — pick-and-place containment, the sink rules, the toaster slots —
        is reported as ``kind: "unknown"`` with the class, its scoring attributes and the
        first 1200 characters of ``_check_success``, and ``decoded`` is False; read
        ``satisfied`` there.  Costs no simulation steps.
        """
        self._ensure_built()
        env = self.env
        cls = type(env).__name__
        meta = self.episode_meta_public()
        behavior = getattr(env, "behavior", None)
        preds: list[dict[str, Any]] = []
        fixtures = self.task_fixtures()

        def frac_pred(fx: Any, joints: list[str], want_open: bool, hi: float, lo: float) -> None:
            vals = {}
            for jn in joints:
                with contextlib.suppress(Exception):
                    vals[jn] = round(float(self._joint_fraction(fx, jn)), 4)
            preds.append(
                {
                    "kind": "joint_fraction",
                    "fixture": str(fx.name),
                    "joints": joints,
                    "test": f"every joint fraction {'>' if want_open else '<'} "
                    f"{hi if want_open else lo}",
                    "thresholds": {"open": hi, "closed": lo},
                    "want": "open" if want_open else "closed",
                    "value": vals,
                }
            )

        # RoboCasa's `_check_success` is per-class Python with no machine-readable
        # descriptor.  Reading its SOURCE is the only general decoder: every predicate
        # below is emitted because the scorer's own body asks for it, never because the
        # task is called something.  A task whose success test mentions nothing we know
        # falls through to the `unknown` predicate — which is why
        # PickPlaceCounterToMicrowave (which carries `env.microwave` but is scored on a
        # containment test) is no longer told it is scored on the microwave's button.
        src = ""
        with contextlib.suppress(OSError, TypeError):
            src = inspect.getsource(type(env)._check_success)
        fxtr = getattr(env, "fxtr", None)
        if fxtr is None:
            # a door task that keeps its fixture under its own name (the toaster oven):
            # take the attribute the success test calls is_open/is_closed on
            for attr, value in vars(env).items():
                if hasattr(value, "is_open") and f"self.{attr}.is_" in src:
                    fxtr = value
                    behavior = "open" if f"self.{attr}.is_open" in src else "close"
                    break
        if fxtr is not None and behavior in ("open", "close") and hasattr(fxtr, "is_open"):
            doors = self._default_door_joints(fxtr)
            # a fixture whose own drawer joints are what the region names (the fridge's
            # drawer tasks) is scored on those; the thresholds always come from the
            # scorer's own table, never from a number written here
            drawers = list(getattr(fxtr, "_drawer_joint_names", []) or [])
            if getattr(env, "reg_type", None) == "drawer" and drawers:
                doors = drawers
            doors = self._joints_named_by(fxtr, doors)
            hi, lo = self._scorer_thresholds(fxtr, doors[0], doors) if doors else (0.9, 0.005)
            frac_pred(fxtr, doors, behavior == "open", hi, lo)
        stove = getattr(env, "stove", None)
        knob = getattr(env, "knob", None)
        if stove is not None and knob is not None:
            jn = f"{stove.naming_prefix}knob_{knob}_joint"
            q = None
            with contextlib.suppress(Exception):
                q = float(self.sim.data.get_joint_qpos(jn))
            if "STOVE_HIGH_MIN" in src:
                hi_min = float(getattr(stove, "STOVE_HIGH_MIN", 1.3963))
                test = (
                    f"0.35 <= |qpos mod 2pi| < {hi_min:.4f} (Stove.STOVE_HIGH_MIN): low heat"
                )
                target = [0.35, round(hi_min, 4)]
            elif behavior == "turn_off":
                test = "NOT (0.35 <= |qpos mod 2pi| <= 2pi - 0.35): burner off"
                target = [0.0, 0.35]
            else:
                test = "0.35 <= |qpos mod 2pi| <= 2pi - 0.35: burner on"
                target = [0.35, 2 * np.pi - 0.35]
            preds.append(
                {
                    "kind": "knob_qpos",
                    "fixture": str(stove.name),
                    "joint": jn,
                    "knob": str(knob),
                    "test": test,
                    "target_qpos": target,
                    # `value` is the RAW joint position (what get_joint_state and
                    # turn_knob's target_rad speak); the scorer normalises it first
                    # (Stove.get_knobs_state: q mod 2pi, so a knob resting a hair below 0
                    # reads 6.283, not 0) and `value_mod_2pi` is that number
                    "value": None if q is None else round(q, 4),
                    "value_mod_2pi": None if q is None else round(float(q) % (2 * np.pi), 4),
                }
            )
        for attr, key, rule in (
            ("microwave", "turned_on", "Microwave.get_state()['turned_on'] (start/stop button)"),
            ("blender", "turned_on", "Blender.get_state()['turned_on'] (button released, lid on)"),
            ("blender", "lid_on_blender", "Blender.get_state()['lid_on_blender']"),
            ("electric_kettle", "turned_on", "ElectricKettle: switch fraction >= 0.95"),
            (
                "electric_kettle",
                "lid",
                "ElectricKettle lid fraction (open >= 0.95 / closed <= 0.01)",
            ),
            ("stand_mixer", "head", "StandMixer head fraction (open > 0.99 / closed < 0.01)"),
            (
                "dishwasher",
                "rack",
                "Dishwasher.get_state()['rack'] fraction (pulled out >= 0.95 / pushed in "
                "<= 0.05) -- the rack, NOT the door",
            ),
            ("toaster_oven", "time", "ToasterOven time knob fraction >= 0.1"),
            ("toaster_oven", "temperature", "ToasterOven temperature fraction, |delta| >= 0.15"),
            ("oven", "temperature", "Oven temperature fraction >= 0.25"),
            ("coffee_machine", "turned_on", "CoffeeMachine start button contact"),
            ("toaster", "turned_on", "Toaster lever of the slot holding the bread"),
        ):
            fx = getattr(env, attr, None)
            if fx is None or not hasattr(fx, "get_state"):
                continue
            if f"self.{attr}" not in src:
                # the env carries the fixture but the success test never looks at it
                # (PickPlaceCounterToMicrowave holds `self.microwave` and scores a
                # containment test): inventing a predicate here is a fabricated reading
                continue
            state = None
            with contextlib.suppress(Exception):
                try:
                    state = fx.get_state(env)
                except TypeError:
                    state = fx.get_state()
            if isinstance(state, dict) and key in state:
                preds.append(
                    {
                        "kind": "fixture_state",
                        "fixture": str(fx.name),
                        "key": key,
                        "test": rule,
                        "value": _plain(state[key]),
                    }
                )
        if "gripper_button_far" in src and getattr(env, "microwave", None) is not None:
            preds.append(
                {
                    "kind": "gripper_far",
                    "fixture": str(env.microwave.name),
                    "part": "start_button" if behavior == "turn_on" else "stop_button",
                    "test": "grip site > 0.15 m from the button geom after the press",
                }
            )
        if "blender_lid" in src:
            lid = getattr(getattr(env, "blender", None), "blender_lid", None)
            if lid is not None:
                preds.append(
                    {
                        "kind": "gripper_far",
                        "fixture": str(lid.name),
                        "part": f"{lid.name}_main",
                        "test": "grip site > 0.15 m from the lid; OpenBlenderLid also wants the "
                        "lid in contact with a Counter fixture",
                    }
                )
        if not preds:
            preds.append(
                {
                    "kind": "unknown",
                    "task": cls,
                    "test": "not decoded: read _check_success in robocasa/environments/kitchen/"
                    "atomic/*.py; `satisfied` below is that predicate, live",
                    "check_success_source": src[:1200],
                    "attributes": {
                        k: _plain(v)
                        for k, v in vars(env).items()
                        if k
                        in (
                            "behavior",
                            "should_pull",
                            "rack_level",
                            "knob",
                            "target_fixture",
                            "initial_temp",
                            "reg_type",
                            "drawer_side",
                        )
                    },
                }
            )
        satisfied = None
        with contextlib.suppress(Exception):
            satisfied = bool(env._check_success())
        return {
            "instruction": self.goal_text(),
            "task": cls,
            "behavior": _plain(behavior),
            "satisfied": satisfied,
            # False when nothing in the scorer's own body was recognised: `predicates`
            # then holds only the `unknown` entry and `satisfied` is the thing to read
            "decoded": not any(p["kind"] == "unknown" for p in preds),
            "predicates": preds,
            "fixtures": fixtures,
            "objects": self.list_objects(),
            "meta": {k: v for k, v in meta.items() if k not in ("lang", "fixture_refs")},
        }


    #: the scorer's own placement predicates, and which of their arguments name the object
    #: that has to move and the thing it has to end up in.  Read off
    #: ``robocasa/utils/object_utils.py``; the argument NAMES are what the tracer binds, so a
    #: task that passes them as keywords (``obj_name="meat_container"``) resolves the same.
    PLACEMENT_PREDICATES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], bool]] = {
        # predicate: (kind, the moved object's argument(s), the destination's, dest is a fixture)
        "check_obj_in_receptacle": ("in_receptacle", ("obj_name",), ("receptacle_name",), False),
        "obj_inside_of": ("inside_fixture", ("obj_name",), ("fixture_id",), True),
        "check_obj_fixture_contact": ("fixture_contact", ("obj_name",), ("fixture_name",), True),
        "check_obj_any_counter_contact": ("counter_contact", ("obj_name",), (), True),
    }
    #: the conjuncts that constrain the SAME object without moving it anywhere
    OTHER_PREDICATES: dict[str, tuple[str, tuple[str, ...]]] = {
        "gripper_obj_far": ("gripper_far", ("obj_name",)),
        "check_obj_upright": ("upright", ("obj_name",)),
        "check_obj_grasped": ("grasped", ("obj_name",)),
    }
    #: fixture METHODS the scorer calls the same way (``self.fridge.check_rack_contact``)
    FIXTURE_PREDICATES: dict[str, tuple[str, tuple[str, ...]]] = {
        "check_rack_contact": ("rack_contact", ("obj_name", "object_name")),
        "check_obj_location_on_stove": ("on_stove", ("obj_name",)),
    }
    #: how many single-predicate flips the tracer will try to reach a short-circuited branch
    TRACE_FLIPS = 12

    def _negated_predicate_calls(self, src: str) -> set[tuple[str, str | None]]:
        """``{(predicate, object literal or None)}`` for the calls the scorer NEGATES.

        ``UtensilShuffle`` wants the utensils in the drawer *and* ``not
        check_obj_any_counter_contact(utensil1)``.  The tracer sees the call and its real
        answer; only the source says the answer is wanted the other way round, so a
        predicate reading False there is a conjunct already SATISFIED, not work.
        """
        known = (set(self.PLACEMENT_PREDICATES) | set(self.OTHER_PREDICATES)
                 | set(self.FIXTURE_PREDICATES))
        out: set[tuple[str, str | None]] = set()
        try:
            tree = ast.parse(textwrap.dedent(src))
        except (SyntaxError, ValueError):
            return out

        def literal(node: Any) -> str | None:
            return node.value if isinstance(node, ast.Constant) and isinstance(
                node.value, str) else None

        def walk(node: Any, negated: bool) -> None:
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                walk(node.operand, not negated)
                return
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if negated and name in known:
                    arg = literal(node.args[1]) if len(node.args) > 1 else None
                    if arg is None:
                        for keyword in node.keywords:
                            if keyword.arg in ("obj_name", "object_name"):
                                arg = literal(keyword.value)
                    out.add((name, arg))
            for child in ast.iter_child_nodes(node):
                walk(child, negated)

        walk(tree, False)
        return out

    def placement_goal(self) -> dict[str, Any]:
        """Which object has to move and what it has to end up in — read OUT OF THE SCORER.

        ``goal()`` decodes a containment check as ``kind: "unknown"``, and the referent
        rules a program can write from outside — "the receptacle is ``<obj>_container``",
        "the object is ``objects[0]``", "the destination is the fixture the sentence names
        last" — do not survive the composite suite: a composite calls them ``marshmallow``
        and ``marshmallow_container`` where ``marshmallow_container`` is the plate it
        STARTED on, and ``PrepForTenderizing`` scores the rolling pin against the *meat's*
        cutting board.  Every one of those guesses is unnecessary, because
        ``_check_success`` names both ends itself.

        So this does not guess: it **runs the episode's own** ``_check_success`` with the
        benchmark's placement predicates wrapped, and records the arguments each one was
        actually called with together with the value it really returns.  Three things make
        the reading complete:

        * a wrapper returns ``True`` rather than the real answer, so ``and`` does not
          short-circuit past the conjuncts after the first false one;
        * the body is traced again with every wrapper honest, in case returning ``True``
          skipped a branch;
        * and then once per call site with **that one call flipped to False**, which is what
          reaches a conjunct hidden behind a negation (``UtensilShuffle``:
          ``... and not OU.obj_inside_of(self, "condiment", self.drawer) and
          OU.check_obj_any_counter_contact(...)``).

        The union of the records is the reading; the value stored against each is always the
        predicate's REAL answer, whatever the wrapper returned.  Nothing is stepped and
        nothing is written to the world — ``_check_success`` is a pure read, so this costs no
        simulation steps.

        Returns ``{"task", "instruction", "satisfied", "placements": [...], "pending":
        [...], "move": {...} | None, "single", "gripper_far": [names], "constraints": [...],
        "opaque", "source"}``.  A placement record is ``{"kind", "object", "destination",
        "destination_is_fixture", "satisfied", "support"}`` — ``support`` being the fixture
        the destination object is resting on right now, which is where the base has to stand
        to reach it.  ``pending`` is the placements that are FALSE at this instant, i.e. the
        work; a composite that stages one containment and scores two
        (``PrepForTenderizing``: the meat is already on its board) has one pending, and
        ``single`` says the referent is decided.  ``opaque`` is True when the scorer's
        conjunction contains a term this cannot see (``DryDrinkware`` tests a mug's Euler
        angle inline), so a conjunct may have been missed.
        """
        self._ensure_built()
        env = self.env
        from robocasa.utils import object_utils as ou

        src = ""
        with contextlib.suppress(OSError, TypeError):
            src = inspect.getsource(type(env)._check_success)
        negated_calls = self._negated_predicate_calls(src)
        records: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        others: dict[tuple[str, str | None], dict[str, Any]] = {}
        # `check_obj_any_counter_contact` calls `check_obj_fixture_contact` once per counter
        # in the kitchen: only the OUTERMOST call is a conjunct of the scorer, and a nested
        # one must see the predicate's real answer, not what the wrapper returns.
        depth = [0]
        seen = [0]  # call sites reached in this pass, in order
        flip = [-1]  # the index of the one call this pass answers False

        def as_name(value: Any) -> str | None:
            if isinstance(value, str):
                return value
            name = getattr(value, "name", None)
            return str(name) if isinstance(name, str) else None

        def as_fixture(value: Any) -> str | None:
            with contextlib.suppress(Exception):
                return str(env.get_fixture(value).name)
            return as_name(value)

        def first(bind: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
            for key in keys:
                if bind and key in bind:
                    return bind[key]
            return None

        def note(kind: str, bind: dict | None, spec: tuple, value: bool | None,
                 bound: Any) -> None:
            if kind == "placement":
                name, obj_keys, dest_keys, dest_is_fx = spec
                obj = as_name(first(bind, obj_keys))
                if obj is None:
                    return
                raw = first(bind, dest_keys)
                dest = None if raw is None else (as_fixture(raw) if dest_is_fx
                                                 else as_name(raw))
                key = (name, obj, dest)
                rec = records.setdefault(
                    key,
                    {"kind": name, "object": obj, "destination": dest,
                     "destination_is_fixture": bool(dest_is_fx), "satisfied": value},
                )
            elif kind == "fixture":
                name, obj_keys = spec
                obj = as_name(first(bind, obj_keys))
                if obj is None:
                    return
                dest = as_name(bound)
                key = (name, obj, dest)
                rec = records.setdefault(
                    key,
                    {"kind": name, "object": obj, "destination": dest,
                     "destination_is_fixture": True, "satisfied": value},
                )
            else:
                name, obj_keys = spec
                obj = as_name(first(bind, obj_keys))
                rec = others.setdefault((name, obj),
                                        {"kind": name, "object": obj, "satisfied": value})
            if value is not None:
                rec["satisfied"] = value

        def wanted(call: str, bind: dict | None, keys: tuple[str, ...]) -> bool:
            """What the scorer wants this conjunct to answer: False where it negates it."""
            obj = as_name(first(bind, keys))
            return not ((call, obj) in negated_calls or (call, None) in negated_calls)

        def wrap(real: Any, call: str, kind: str, spec: tuple, honest: bool,
                 bound: Any = None) -> Any:
            signature = None
            with contextlib.suppress(Exception):
                signature = inspect.signature(real)

            def probe(*args: Any, **kwargs: Any) -> Any:
                if depth[0]:
                    return real(*args, **kwargs)
                index = seen[0]
                seen[0] += 1
                value = None
                depth[0] += 1
                try:
                    with contextlib.suppress(Exception):
                        value = bool(real(*args, **kwargs))
                finally:
                    depth[0] -= 1
                bind = None
                if signature is not None:
                    with contextlib.suppress(Exception):
                        bound_args = signature.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        bind = bound_args.arguments
                note(kind, bind, spec, value, bound)
                if honest:
                    return value
                # answer every conjunct the way the scorer WANTS it, so the conjunction runs
                # to the end: True normally, False where the source negates this call
                return (index != flip[0]) and wanted(call, bind, spec[1])

            return probe

        def one_pass(honest: bool, flip_index: int) -> int:
            seen[0] = 0
            flip[0] = flip_index
            saved: list[tuple[Any, str, Any]] = []
            for name, spec in self.PLACEMENT_PREDICATES.items():
                real = getattr(ou, name, None)
                if real is not None:
                    saved.append((ou, name, real))
                    setattr(ou, name, wrap(real, name, "placement",
                                           (spec[0], spec[1], spec[2], spec[3]), honest))
            for name, spec in self.OTHER_PREDICATES.items():
                real = getattr(ou, name, None)
                if real is not None:
                    saved.append((ou, name, real))
                    setattr(ou, name, wrap(real, name, "other", spec, honest))
            # `self.fridge.check_rack_contact(self, "bowl")` is a bound method, so it is
            # shadowed on the INSTANCE (a plain attribute wins over a class function) and
            # removed again below — never on the class, which every episode shares.
            for fixture in list(getattr(env, "fixtures", {}).values()):
                for name, spec in self.FIXTURE_PREDICATES.items():
                    real = getattr(fixture, name, None)
                    if real is None:
                        continue
                    saved.append((fixture, name, None))
                    with contextlib.suppress(Exception):
                        setattr(fixture, name, wrap(real, name, "fixture", spec, honest,
                                                    bound=fixture))
            try:
                with contextlib.suppress(Exception):
                    env._check_success()
            finally:
                for holder, name, real in reversed(saved):
                    if real is None:
                        with contextlib.suppress(Exception):
                            delattr(holder, name)
                    else:
                        setattr(holder, name, real)
            return seen[0]

        calls = one_pass(honest=False, flip_index=-1)
        one_pass(honest=True, flip_index=-1)
        for index in range(min(int(calls), int(self.TRACE_FLIPS))):
            one_pass(honest=False, flip_index=index)

        placements = [records[k] for k in sorted(records)]
        for rec in placements:
            rec["support"] = (
                None if rec["destination_is_fixture"] or rec["destination"] is None
                else self.supporting_fixture(rec["destination"])
            )
        negated = negated_calls
        by_kind = {spec[0]: name for name, spec in self.PLACEMENT_PREDICATES.items()}
        by_kind.update({spec[0]: name for name, spec in self.FIXTURE_PREDICATES.items()})
        for rec in placements:
            call = by_kind.get(rec["kind"], rec["kind"])
            rec["negated"] = (call, rec["object"]) in negated or (call, None) in negated
        # the work is a placement the scorer wants TRUE and reads False; a negated one that
        # reads True is a violation to undo, which is a different job and never a carry
        pending = [r for r in placements if r["satisfied"] is False and not r["negated"]]
        avoid = [r for r in placements if r["satisfied"] is True and r["negated"]]
        satisfied = None
        with contextlib.suppress(Exception):
            satisfied = bool(env._check_success())
        # a conjunct the tracer cannot see (an inline geometry test, a helper that reads the
        # sim directly) can still short-circuit the scorer: say so rather than imply the
        # reading is complete
        stripped = src.split("\n", 1)[-1]
        for name in (tuple(self.PLACEMENT_PREDICATES) + tuple(self.OTHER_PREDICATES)
                     + tuple(self.FIXTURE_PREDICATES)):
            stripped = stripped.replace(name, "")
        opaque = bool(re.search(r"self\.sim\.|np\.|self\.objects\[", stripped))
        return {
            "task": type(env).__name__,
            "instruction": self.goal_text(),
            "satisfied": satisfied,
            "placements": placements,
            "pending": pending,
            "avoid": avoid,
            "move": pending[0] if pending else None,
            "single": len(pending) == 1,
            "gripper_far": [o["object"] for o in others.values()
                            if o["kind"] == "gripper_far" and o["object"]],
            "constraints": [o for o in others.values() if o["kind"] != "gripper_far"],
            "opaque": opaque,
            "source": src,
        }

    def supporting_fixture(self, object_name: str) -> str | None:
        """The fixture ``object_name`` is resting on right now, from MuJoCo's own contacts.

        What a placement needs is the fixture to STAND at, and the sentence is a bad way to
        find it (it names a counter that three fixtures share, or nothing at all).  The
        receptacle is touching its support, so ask the simulator.  Contacts with the robot
        and with other objects are ignored; a stack of counters returns the one whose top is
        nearest below the object.  Costs no simulation steps.
        """
        self._ensure_built()
        env = self.env
        obj = env.objects.get(object_name)
        if obj is None:
            return None
        best: tuple[float, str] | None = None
        for name, fixture in env.fixtures.items():
            hit = False
            with contextlib.suppress(Exception):
                hit = bool(env.check_contact(obj, fixture))
            if not hit:
                continue
            score = 0.0
            with contextlib.suppress(Exception):
                pos = self.get_object_pose(object_name)[0]
                fpos = np.asarray(self.get_object_pose(name)[0], dtype=float)
                score = float(np.linalg.norm(np.asarray(pos, dtype=float)[:2] - fpos[:2]))
            if best is None or score < best[0]:
                best = (score, str(name))
        return None if best is None else best[1]

    #: parts a RoboCasa sentence names, in the words its joints are named with
    PART_WORDS: tuple[str, ...] = ("drawer", "rack", "shelf", "lid", "door", "tray", "freezer")

    def _joints_named_by(self, fixture: Any, joints: list[str]) -> list[str]:
        """Narrow a fixture's decoded joints to the ones the EPISODE'S OWN SENTENCE names.

        RoboCasa's predicate names the fixture reliably and the joint only sometimes: on
        the fridge, ``Open/CloseFridgeDrawer`` reads
        ``fridge.is_closed(reg_type="drawer")`` while the fixture's default door joints are
        the doors, which the setup staged *open* so the drawer is reachable.  Driving the
        decoded joint there spent 905 frames and failed.  The sentence — "Fully close the
        fridge **drawer**" — picks the right one, and on RoboCasa the sentence is
        trustworthy for this: its decoys are visual, not linguistic.

        Only ever narrows, and only when the sentence names a part the fixture actually has
        a joint for AND the decoded set contains none of that part.
        """
        text = str(self.goal_text() or "").lower()
        named = [w for w in self.PART_WORDS if w in text]
        if not named or not joints:
            return joints
        try:
            available = list(self.fixture_joints(str(fixture.name)).get("joints") or {})
        except Exception:
            return joints
        for word in named:
            hits = [j for j in available if word in j.lower()]
            if hits and not any(word in j.lower() for j in joints):
                _trace(f"goal: the sentence says {word!r}; {joints} -> {hits}")
                return hits
        return joints

    # ------------------------------------------------------------ base route planning

    # ------------------------------------------------------------ driving the base around

    #: how many metres of driving a quarter-turn is worth when the router chooses between
    #: a longer straight route and a shorter one that has to turn (a turn in place costs
    #: 40-90 frames, a metre of driving 30-60, so they are within a factor of two)
    TURN_COST_M: float = 0.7

    def _route_search(
        self,
        clear: Any,
        s0: np.ndarray,
        goal: np.ndarray,
        headings: np.ndarray,
        cell_m: float,
        max_states: int,
    ) -> dict[str, Any]:
        """Uniform-cost search over ``(cell_x, cell_y, heading_index)`` for a drivable
        route from ``s0`` (at ``headings[0]``) to ``goal``.

        **Every edge is checked, not every cell.**  A translation edge is admitted only
        when the footprint clears all along it at that state's heading; a rotation edge
        only when the footprint clears at the new heading where it stands.  A path this
        search returns is therefore drivable by construction — the caller can shorten it
        but never has to reject it.
        """
        n_h = len(headings)
        diag = float(np.sqrt(2.0)) * cell_m

        def pos(k: tuple[int, int, int]) -> np.ndarray:
            return s0 + np.array([k[0], k[1]], dtype=float) * cell_m

        def seg(a: np.ndarray, b: np.ndarray, h: float) -> bool:
            d = b - a
            n = max(int(np.ceil(np.linalg.norm(d) / (cell_m / 2))), 1)
            return all(clear(*(a + d * t / n), h) for t in range(1, n + 1))

        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        st0 = (0, 0, 0)
        best: dict[tuple[int, int, int], float] = {st0: 0.0}
        came: dict[tuple[int, int, int], tuple[int, int, int] | None] = {st0: None}
        queue: list[tuple[float, int, tuple[int, int, int]]] = [(0.0, 0, st0)]
        tie = 0
        expanded = 0
        done: set[tuple[int, int, int]] = set()
        found: tuple[int, int, int] | None = None
        while queue and expanded < max_states:
            cost, _, cur = heapq.heappop(queue)
            if cur in done:
                continue
            done.add(cur)
            expanded += 1
            p = pos(cur)
            yaw = float(headings[cur[2]])
            # the goal counts only when the last leg into it is itself clear
            if float(np.linalg.norm(p - goal)) < cell_m * 0.75 and seg(p, goal, yaw):
                found = cur
                break
            for dx, dy in moves:
                nxt = (cur[0] + dx, cur[1] + dy, cur[2])
                step = diag if dx and dy else cell_m
                if best.get(nxt, np.inf) <= cost + step:
                    continue
                q = pos(nxt)
                if not (clear(q[0], q[1], yaw) and seg(p, q, yaw)):
                    continue
                best[nxt] = cost + step
                came[nxt] = cur
                tie += 1
                heapq.heappush(queue, (cost + step, tie, nxt))
            for dh in (1, -1):
                nxt = (cur[0], cur[1], (cur[2] + dh) % n_h)
                turn = self.TURN_COST_M * (2 * np.pi / n_h) / (np.pi / 2)
                if n_h == 1 or best.get(nxt, np.inf) <= cost + turn:
                    continue
                if not clear(p[0], p[1], float(headings[nxt[2]])):
                    continue
                best[nxt] = cost + turn
                came[nxt] = cur
                tie += 1
                heapq.heappush(queue, (cost + turn, tie, nxt))
        if found is None:
            return {"ok": False, "states": expanded}
        chain: list[tuple[int, int, int]] = []
        k: tuple[int, int, int] | None = found
        while k is not None:
            chain.append(k)
            k = came[k]
        chain.reverse()
        # [x, y, yaw] per state, the exact start replacing its (rounded) cell and the exact
        # goal appended at the heading it is entered on
        raw = [[float(s0[0]), float(s0[1]), float(headings[chain[0][2]])]]
        for k2 in chain[1:]:
            raw.append([float(pos(k2)[0]), float(pos(k2)[1]), float(headings[k2[2]])])
        raw.append([float(goal[0]), float(goal[1]), float(headings[chain[-1][2]])])
        # shorten each same-heading run greedily; a heading change stays as its own leg
        out = [raw[0]]
        i = 1
        while i < len(raw):
            j = i
            while j + 1 < len(raw) and raw[j + 1][2] == raw[i][2]:
                j += 1
            h = raw[i][2]
            run = [out[-1][:2]] + [w[:2] for w in raw[i : j + 1]]
            a = 0
            while a < len(run) - 1:
                b = len(run) - 1
                while b > a + 1 and not seg(np.array(run[a]), np.array(run[b]), h):
                    b -= 1
                if run[b] != out[-1][:2] or h != out[-1][2]:
                    out.append([run[b][0], run[b][1], h])
                a = b
            i = j + 1
        length = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(out)))
        turns = sum(1 for a, b in itertools.pairwise(out) if b[2] != a[2])
        return {
            "ok": True,
            "states": expanded,
            "waypoints": [[round(v, 3) for v in w] for w in out[1:]],
            "length_m": round(length, 3),
            "turns": turns,
        }

    def plan_base_route(
        self,
        x: float,
        y: float,
        yaw: float | None = None,
        *,
        cell_m: float = 0.25,
        max_cells: int = 4000,
        headings: int = 8,
        relax_goal_m: float = 0.35,
    ) -> dict[str, Any]:
        """Waypoints ``[[x, y, yaw], ...]`` from the base's current pose to ``(x, y[, yaw]]``
        over :meth:`base_pose_clear` (a free occupancy oracle: no simulation steps).

        Two passes, both of which check **the edge between two cells, not the cell**
        (measured 2026-08-28: checking only the cell admits a diagonal squeeze whose
        midpoint is blocked, which is what made this method refuse a route that exists —
        `AddMarshmallow`'s dining counter, 5.6 m away, with two such cells out of twenty):

        1. a uniform-cost search at the base's CURRENT heading, so nothing turns;
        2. failing that, the same search with the heading in the state — the base is
           holonomic (its two slides are unit world-axis translations that do not turn
           with the yaw), so a leg may be driven at any heading, and a turn in place is
           priced at :attr:`TURN_COST_M` per quarter-turn against the metres it saves.

        A goal whose own footprint is not clear is moved to the nearest pose that is,
        within ``relax_goal_m`` (32 of the 54 old refusals, and the median move that fixes
        one is **0.05 m**: RoboCasa's standing poses sit a couple of centimetres inside this
        footprint, they are not somewhere else).  And when nothing plans at the 0.03 m
        margin the whole search is repeated at **0.0** — still a real no-overlap test, and
        the margin the simulator itself spawns the robot at (the base footprint is not clear
        at 0.03 in 37 of those 95 episodes).  The margin used is reported.

        Each waypoint carries the heading its leg is driven at; the last one turns to
        ``yaw`` when given.  Returns ``{"ok", "waypoints", "length_m", "turns", "goal",
        "goal_moved_m", "margin", "cells_expanded", "reason"}``; an unreachable goal (no
        pose near it is clear, or nothing connects) is ``ok`` False.  Drive it with
        :meth:`drive_route`, which turns in place before a leg whose heading differs from
        the one it is at.
        """
        start = self.get_base_pose()
        goal = np.array([float(x), float(y)])
        goal_heading = float(yaw) if yaw is not None else float(start[2])
        s0 = np.array([start[0], start[1]])
        # The straight line first, at both margins.  It is one leg, it is exactly what
        # ``drive_to`` does unaided, and it is what every existing certificate already pays
        # for: measured 2026-08-28, routing a clear 0.38 m hop as a relaxed two-leg plan
        # cost 58 frames where the straight line cost 19, and that was the whole of an
        # `ArrangeVegetables` regression (961 frames, certified, to 943 with no frames left
        # for the retraction).  Plan a route when the line is blocked, not before.
        for margin in (0.03, 0.0):
            clear = self._clearance(margin)
            d = goal - s0
            n = max(int(np.ceil(float(np.linalg.norm(d)) / (cell_m / 2))), 1)
            if all(clear(*(s0 + d * t / n), goal_heading) for t in range(1, n + 1)):
                return {
                    "ok": True,
                    "waypoints": [[round(float(goal[0]), 3), round(float(goal[1]), 3),
                                   round(goal_heading, 3)]],
                    "length_m": round(float(np.linalg.norm(d)), 3),
                    "turns": 0,
                    "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                    "goal_moved_m": 0.0,
                    "margin": margin,
                    "cells_expanded": 0,
                    "reason": "",
                }
        for margin in (0.03, 0.0):
            out = self._plan_route_at(x, y, yaw, cell_m, max_cells, headings, relax_goal_m, margin)
            out["margin"] = margin
            if out["ok"] or margin == 0.0:
                return out
        raise AssertionError  # unreachable

    def _plan_route_at(
        self,
        x: float,
        y: float,
        yaw: float | None,
        cell_m: float,
        max_cells: int,
        headings: int,
        relax_goal_m: float,
        margin: float,
    ) -> dict[str, Any]:
        """:meth:`plan_base_route` at one footprint margin."""
        start = self.get_base_pose()
        goal = np.array([float(x), float(y)])
        goal_yaw = float(yaw) if yaw is not None else None
        s0 = np.array([start[0], start[1]])
        clear = self._clearance(margin)
        goal_heading = goal_yaw if goal_yaw is not None else float(start[2])
        moved = 0.0
        if not clear(goal[0], goal[1], goal_heading):
            # RoboCasa's own standing poses (``base_placement_for``) routinely park the
            # base right against the fixture they stand at, and this footprint (0.70 x
            # 0.50 with the origin 0.55 m in front of the tail, plus a 3 cm margin) calls
            # that blocked -- measured 2026-08-28: it is 28 of the 38 route refusals over
            # the composite suite, against 5 for the legs.  A standing pose that cannot be
            # stood in is a request for the nearest one that can: back off first (the
            # counter is what the base is usually inside), then fan outward.
            back = float(goal_heading + np.pi)
            relaxed = None
            for radius in np.arange(0.05, float(relax_goal_m) + 1e-9, 0.05):
                for off in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, np.pi):
                    phi = back + off
                    cand = goal + radius * np.array([np.cos(phi), np.sin(phi)])
                    if clear(cand[0], cand[1], goal_heading):
                        relaxed = (cand, float(radius))
                        break
                if relaxed is not None:
                    break
            if relaxed is None:
                return {
                    "ok": False,
                    "waypoints": [],
                    "length_m": 0.0,
                    "turns": 0,
                    "cells_expanded": 0,
                    "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                    "goal_moved_m": 0.0,
                    "reason": "the goal pose's footprint is not clear, and no pose within "
                    f"{relax_goal_m:.2f} m of it is either",
                }
            goal, moved = relaxed
        n_h = max(int(headings), 1)
        expanded = 0
        res: dict[str, Any] = {}
        for hs in (
            np.array([float(start[2])]),
            float(start[2]) + np.arange(n_h) * (2 * np.pi / n_h),
        ):
            res = self._route_search(clear, s0, goal, hs, cell_m, max_cells * len(hs))
            expanded += int(res["states"])
            if res["ok"]:
                break
            if len(hs) == 1 and n_h <= 1:
                break
        if not res.get("ok"):
            return {
                "ok": False,
                "waypoints": [],
                "length_m": 0.0,
                "turns": 0,
                "cells_expanded": expanded,
                "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                "goal_moved_m": round(float(moved), 3),
                "reason": "no route whose legs are clear at any heading",
            }
        waypoints = [list(w) for w in res["waypoints"]]
        if goal_yaw is not None and waypoints:
            last = waypoints[-1]
            dy = float(np.arctan2(np.sin(goal_yaw - last[2]), np.cos(goal_yaw - last[2])))
            if abs(dy) > 1e-6:
                # turn to the requested yaw where the route ends, as its own leg
                waypoints.append([last[0], last[1], round(goal_yaw, 3)])
        return {
            "ok": True,
            "waypoints": waypoints,
            "length_m": res["length_m"],
            "turns": res["turns"],
            "cells_expanded": expanded,
            "goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
            "goal_moved_m": round(float(moved), 3),
            "reason": "",
        }

    #: a leg whose heading differs from the base's by more than this is turned into first
    #: (``drive_to`` turns WHILE it translates, so the footprint along such a leg is at
    #: neither heading and the plan's clearance would not describe it)
    TURN_FIRST_RAD: float = 0.15

    def drive_route(
        self, x: float, y: float, yaw: float | None = None, *, hold_arm: bool = True
    ) -> dict[str, Any]:
        """Plan with :meth:`plan_base_route` and drive each waypoint with ``drive_to``
        (arm held at its base-relative pose), after :meth:`tuck_arm` — ``drive_to`` holds
        whatever pose the hand is in, so an extended arm is carried extended into the
        furniture the base drives past.

        A leg whose heading differs from the base's current one is **turned into on the
        spot first**: ``drive_to`` turns while it translates, so a leg driven straight at a
        new heading sweeps the footprint through orientations the plan never checked.
        Stops at the first leg that is not reached.  Returns ``{"reached", "legs": [drive
        reports], "plan", "pose", "tuck", "steps", "budget_exhausted"}``.
        """
        s0 = self._step_count
        tuck = self.tuck_arm()
        plan = self.plan_base_route(x, y, yaw)
        legs: list[dict[str, Any]] = []
        reached = False
        if plan["ok"]:
            last = len(plan["waypoints"]) - 1
            for i, (wx, wy, wyaw) in enumerate(plan["waypoints"]):
                bx, by, byaw = self.get_base_pose()
                leg_m = float(np.hypot(wx - bx, wy - by))
                dyaw = abs(float(np.arctan2(np.sin(wyaw - byaw), np.cos(wyaw - byaw))))
                # measured 2026-08-28 on a four-leg 5.4 m route: 107 frames for 3.4 m,
                # 43 for 1.1 m, 27 for 0.6 m -- about 25 frames plus 25 per metre.  Bound
                # each leg by what it needs: the 400-step default is a nine-leg route's
                # whole horizon when one leg creeps without tripping the stall detector.
                budget = int(np.clip(80 + 90 * leg_m, 80, 400))
                if self._budget_left() <= budget:
                    legs.append({"reached": False, "reason": "out of step budget"})
                    break
                if dyaw > self.TURN_FIRST_RAD and leg_m > 0.10:
                    turn = self.drive_to(bx, by, wyaw, max_steps=120, hold_arm=hold_arm)
                    keep = ("reached", "yaw_error", "steps")
                    legs.append({"turn": True, **{k: turn.get(k) for k in keep}})
                # only the LAST leg has to be accurate.  drive_to's own measurement of the
                # base ("a soft friction knee": 0.1 m of error moves it 0.01 cm/step) makes
                # the last few centimetres of every leg the expensive ones, and an
                # intermediate waypoint is a corner to round, not a place to be: 0.05 m ->
                # 0.10 m on the intermediate legs is what pays for the finish.
                rep = self.drive_to(
                    wx,
                    wy,
                    wyaw,
                    tolerance_m=0.05 if i == last else 0.10,
                    max_steps=budget,
                    hold_arm=hold_arm,
                )
                legs.append({k: rep.get(k) for k in ("reached", "error_m", "blocked", "steps")})
                if not rep.get("reached"):
                    break
            else:
                reached = True
        return {
            "reached": reached,
            "legs": legs,
            "plan": plan,
            "tuck": {k: tuck.get(k) for k in ("reached", "reason")},
            "pose": self.get_base_pose().round(3).tolist(),
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }

    # ------------------------------------------------------------ arm posture (IK)

    ARM_JOINTS: tuple[str, ...] = tuple(f"robot0_joint{i}" for i in range(1, 8))
    #: fallback reset posture of the Panda on the Omron, used only as an IK seed when
    #: the robot does not expose ``init_qpos`` (the real one is read at call time)
    HOME_Q: tuple[float, ...] = (0.0, -1.02, 0.0, -2.26, 0.03, 1.51, 0.71)

    def _home_q(self) -> np.ndarray:
        """The robot's own reset arm posture (``init_qpos``), else :attr:`HOME_Q`."""
        q = getattr(self._robot, "init_qpos", None)
        if q is not None and len(np.asarray(q, dtype=float).reshape(-1)) >= 7:
            return np.asarray(q, dtype=float).reshape(-1)[:7]
        return np.asarray(self.HOME_Q, dtype=float)

    def arm_joint_positions(self) -> np.ndarray:
        """The seven arm joint angles (rad), joint1..joint7."""
        return np.array([float(self.sim.data.get_joint_qpos(j)) for j in self.ARM_JOINTS])

    def _arm_limits(self) -> tuple[np.ndarray, np.ndarray]:
        raw_m, _ = self._raw()
        ids = [self.sim.model.joint_name2id(j) for j in self.ARM_JOINTS]
        rng = np.array([raw_m.jnt_range[j] for j in ids], dtype=float)
        return rng[:, 0], rng[:, 1]

    def _scratch(self) -> Any:
        """A private MjData for kinematics-only queries (no physics stepped)."""
        import mujoco

        raw_m, _ = self._raw()
        cache = getattr(self, "_scratch_cache", None)
        if cache is None or cache[0] is not raw_m:
            cache = (raw_m, mujoco.MjData(raw_m))
            self._scratch_cache = cache
        return cache[1]

    def _fk_site(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(grip-site position, rotation matrix) for arm joints ``q_arm`` with everything
        else as it stands now (kinematics only, zero simulation steps).
        """
        import mujoco

        raw_m, raw_d = self._raw()
        sc = self._scratch()
        sc.qpos[:] = raw_d.qpos
        for jn, q in zip(self.ARM_JOINTS, q_arm, strict=False):
            sc.qpos[raw_m.jnt_qposadr[self.sim.model.joint_name2id(jn)]] = float(q)
        mujoco.mj_kinematics(raw_m, sc)
        sid = self.sim.model.site_name2id(GRIP_SITE)
        return np.array(sc.site_xpos[sid], dtype=float), np.array(sc.site_xmat[sid]).reshape(3, 3)

    def _arm_path_collides(self, q_from: np.ndarray, q_to: np.ndarray, samples: int = 10) -> str:
        """Name of the first robot/world geom pair in contact along the straight joint-space
        line from ``q_from`` to ``q_to`` (kinematics + MuJoCo collision on the scratch data,
        no physics stepped), or ``""`` when the whole line is clear.  Contacts between the
        robot's own bodies and with the floor are ignored; penetration must exceed 1 mm.
        A body the gripper HOLDS is NOT exempt yet, so ``path_clear`` reads False for every
        candidate while something is carried (OPEN).
        """
        import mujoco

        raw_m, raw_d = self._raw()
        sc = self._scratch()
        sc.qpos[:] = raw_d.qpos
        ids = [self.sim.model.joint_name2id(j) for j in self.ARM_JOINTS]
        qadr = [int(raw_m.jnt_qposadr[j]) for j in ids]
        robot_prefix = ("robot0", "gripper0", "mobilebase0")
        for k in range(1, int(samples) + 1):
            qk = q_from + (q_to - q_from) * (k / float(samples))
            for a, v in zip(qadr, qk, strict=False):
                sc.qpos[a] = v
            mujoco.mj_kinematics(raw_m, sc)
            mujoco.mj_collision(raw_m, sc)
            for i in range(int(sc.ncon)):
                c = sc.contact[i]
                if c.dist > -0.001:
                    continue
                n1 = self.sim.model.geom_id2name(int(c.geom1)) or ""
                n2 = self.sim.model.geom_id2name(int(c.geom2)) or ""
                r1, r2 = n1.startswith(robot_prefix), n2.startswith(robot_prefix)
                if r1 == r2:
                    continue  # self-contact or world-world
                other = n2 if r1 else n1
                if other in ("floor", "") or other.startswith("floor"):
                    continue
                return f"{n1 if r1 else n2} x {other} at waypoint {k}/{samples}"
        return ""

    def solve_arm_ik(
        self,
        position: Any,
        quaternion_wxyz: Any | None = None,
        *,
        seeds: list[Any] | None = None,
        either_finger_side: bool = True,
        iters: int = 150,
        qpos: Any | None = None,
        quick: bool = False,
    ) -> dict[str, Any]:
        """Joint angles that put the grip site at a world pose - kinematics only, no steps.

        Damped-least-squares IK on the seven arm joints from several seeds (the current
        posture, the reset posture, and elbow/shoulder variants), inside the joint limits;
        with ``either_finger_side`` the half-turn twin of the requested finger axis is
        solved too (the gripper is symmetric).  Solutions are ranked by convergence, then
        by their margin to the joint limits and their distance from the current posture.
        Converged candidates whose straight joint-space line from the current posture sweeps
        the arm through a fixture (MuJoCo collision on a kinematic copy) are ranked after the
        clear ones and say so (``path_clear`` False, ``path_collision`` names the pair).
        Returns ``{"ok", "q", "position_error_m", "orientation_error_rad", "limit_margin_rad",
        "joint_distance_rad", "finger_flipped", "path_clear", "candidates"}``; ``ok`` False
        means no seed converged (< 5 mm, < 3 deg, > 0.03 rad from every limit) - the pose is
        outside the arm's kinematic envelope from where the base stands, not merely outside
        the OSC servo's basin.

        ``qpos`` asks the same question of a HYPOTHETICAL world: a full model qpos (see
        :meth:`hypothetical_qpos`, which moves the base and the torso without stepping) in
        place of the live one.  That is what prices a standing pose before driving to it;
        the arm-path collision check is skipped, because a path from the current posture is
        not a path in that world.  ``quick`` trims the seed list and the iteration count
        (~8x faster, still zero steps) for a scan over many candidates.
        """
        import mujoco
        import robosuite.utils.transform_utils as tu

        raw_m, raw_d = self._raw()
        target = np.asarray(position, dtype=float)
        lo, hi = self._arm_limits()
        q_now = self.arm_joint_positions()
        ids = [self.sim.model.joint_name2id(j) for j in self.ARM_JOINTS]
        qadr = [int(raw_m.jnt_qposadr[j]) for j in ids]
        dadr = [int(raw_m.jnt_dofadr[j]) for j in ids]
        sid = self.sim.model.site_name2id(GRIP_SITE)
        sc = self._scratch()
        sc.qpos[:] = np.asarray(raw_d.qpos if qpos is None else qpos, dtype=float)
        targets_r: list[tuple[np.ndarray | None, bool]] = [(None, False)]
        if quaternion_wxyz is not None:
            r = np.asarray(tu.quat2mat(_xyzw(quaternion_wxyz)), dtype=float)
            targets_r = [(r, False)]
            if either_finger_side:
                flip = np.diag([-1.0, -1.0, 1.0])  # half-turn about the tool (z) axis
                targets_r.append((r @ flip, True))
        home = self._home_q()
        seed_list = [q_now, home]
        for d1 in (-0.6, 0.6):
            s = home.copy()
            s[0] += d1
            seed_list.append(s)
        s = home.copy()
        s[3] = -1.6
        s[1] = -0.4
        seed_list.append(s)
        s = home.copy()
        s[5] = 2.6
        s[3] = -1.2
        seed_list.append(s)
        # tool-forward postures (wrist pitched to the horizontal: joint 6 near pi)
        for q7 in (0.78, -0.78, 2.3):
            seed_list.append(np.array([0.0, 0.5, 0.0, -1.9, 0.0, 2.9, q7]))
        seed_list.append(np.array([0.0, 0.2, 0.0, -2.3, 0.0, 2.5, 0.78]))
        rng = np.random.default_rng(0)  # deterministic restarts
        for _ in range(6):
            seed_list.append(lo + (hi - lo) * rng.uniform(0.15, 0.85, size=7))
        for extra in seeds or []:
            seed_list.append(np.asarray(extra, dtype=float))
        if quick:
            seed_list = seed_list[:7]
            iters = min(int(iters), 60)
        jacp = np.zeros((3, raw_m.nv))
        jacr = np.zeros((3, raw_m.nv))
        cands = []
        for r_t, flipped in targets_r:
            for seed in seed_list:
                q = np.clip(np.asarray(seed, dtype=float), lo + 0.02, hi - 0.02)
                e_p = e_r = np.zeros(3)
                for _ in range(int(iters)):
                    for a, v in zip(qadr, q, strict=False):
                        sc.qpos[a] = v
                    mujoco.mj_kinematics(raw_m, sc)
                    p = np.array(sc.site_xpos[sid])
                    e_p = target - p
                    if r_t is not None:
                        r_c = np.array(sc.site_xmat[sid]).reshape(3, 3)
                        e_r = np.asarray(tu.quat2axisangle(tu.mat2quat(r_t @ r_c.T)), dtype=float)
                    if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 2e-3:
                        break
                    mujoco.mj_comPos(raw_m, sc)
                    mujoco.mj_jacSite(raw_m, sc, jacp, jacr, sid)
                    if r_t is not None:
                        jac = np.vstack([jacp[:, dadr], jacr[:, dadr]])
                        err = np.concatenate([e_p, e_r])
                    else:
                        jac = jacp[:, dadr]
                        err = e_p
                    lam = 0.05
                    # a joint sitting on its limit cannot help in the direction that pushes
                    # it further: drop its column for this step (the classic clamping
                    # trick), so the other joints take the error instead of stalling on it
                    jac = jac.copy()
                    grad = jac.T @ err
                    for j in range(7):
                        if (q[j] <= lo[j] + 1e-6 and grad[j] < 0) or (
                            q[j] >= hi[j] - 1e-6 and grad[j] > 0
                        ):
                            jac[:, j] = 0.0
                    jjt = jac @ jac.T + lam * lam * np.eye(jac.shape[0])
                    dq = jac.T @ np.linalg.solve(jjt, err)
                    # nullspace: drift away from the joint limits
                    mid = (lo + hi) / 2.0
                    null = np.eye(7) - np.linalg.pinv(jac) @ jac
                    dq = dq + null @ (0.1 * (mid - q))
                    n = float(np.linalg.norm(dq))
                    if n > 0.35:
                        dq *= 0.35 / n
                    q = np.clip(q + dq, lo, hi)
                # the loop's e_p/e_r belong to the PREVIOUS iterate, while q, the limit
                # margin and `ok` belong to the final one: an overshooting last step used
                # to report `ok` for a posture that is not at the pose.  Re-run FK on the
                # posture actually returned.
                for a, v in zip(qadr, q, strict=False):
                    sc.qpos[a] = v
                mujoco.mj_kinematics(raw_m, sc)
                e_p = target - np.array(sc.site_xpos[sid])
                if r_t is not None:
                    r_c = np.array(sc.site_xmat[sid]).reshape(3, 3)
                    e_r = np.asarray(tu.quat2axisangle(tu.mat2quat(r_t @ r_c.T)), dtype=float)
                else:
                    e_r = np.zeros(3)
                pe = float(np.linalg.norm(e_p))
                re = float(np.linalg.norm(e_r))
                margin = float(np.min(np.minimum(q - lo, hi - q)))
                dist = float(np.linalg.norm(q - q_now))
                cands.append(
                    {
                        "q": q.round(4).tolist(),
                        "position_error_m": round(pe, 4),
                        "orientation_error_rad": round(re, 4),
                        "limit_margin_rad": round(margin, 3),
                        "joint_distance_rad": round(dist, 3),
                        "finger_flipped": flipped,
                        # 0.0 by construction, not a measurement, when no wrist was asked for
                        "orientation_requested": bool(r_t is not None),
                        "path_clear": None,
                        "path_collision": None,
                        # a solution pinned on a joint limit is not one the servo can hold
                        "ok": bool(pe < 0.005 and re < 0.05 and margin > 0.03),
                    }
                )

        def rank(c: dict[str, Any]) -> tuple:
            return (
                0 if c["ok"] else 1,
                c["position_error_m"] + c["orientation_error_rad"] if not c["ok"] else 0.0,
                -min(c["limit_margin_rad"], 0.4) + 0.15 * c["joint_distance_rad"],
            )

        cands.sort(key=rank)
        distinct: list[dict[str, Any]] = []
        for c in cands:
            if all(np.linalg.norm(np.array(c["q"]) - np.array(d["q"])) > 0.2 for d in distinct):
                distinct.append(c)
            if len(distinct) >= 8:
                break
        # a straight joint-space path that sweeps the arm through the counter is no route:
        # test the ok candidates and put the clear ones first
        for c in distinct:
            if c["ok"] and qpos is None:
                hit = self._arm_path_collides(q_now, np.array(c["q"], dtype=float))
                c["path_clear"] = not hit
                c["path_collision"] = hit
        distinct.sort(key=lambda c: (0 if c.get("path_clear") else 1, rank(c)))
        best = distinct[0]
        out = dict(best)
        out["candidates"] = distinct[:6]
        return out

    def _osc_controller(self) -> Any:
        cc = getattr(self._robot, "composite_controller", None)
        parts = getattr(cc, "part_controllers", None) or {}
        return parts.get(ARM)

    def set_nullspace_posture(self, q: Any | None) -> dict[str, Any]:
        """Point the OSC's null-space spring (kp 10, robosuite ``nullspace_torques``) at the
        arm posture ``q`` (seven joint angles) instead of the reset posture; ``None`` restores
        the reset posture.  A redundancy-resolution setting of the controller, not a
        teleport: the task-space servo still decides the hand pose.
        """
        ctrl = self._osc_controller()
        if ctrl is None or not hasattr(ctrl, "initial_joint"):
            return {"ok": False, "reason": "no OSC controller with initial_joint"}
        if not hasattr(self, "_reset_initial_joint") or self._reset_initial_joint is None:
            self._reset_initial_joint = np.array(ctrl.initial_joint, dtype=float)
        if q is None:
            ctrl.initial_joint = np.array(self._reset_initial_joint, dtype=float)
        else:
            ctrl.initial_joint = np.asarray(q, dtype=float).copy()
        return {"ok": True, "posture": np.asarray(ctrl.initial_joint).round(3).tolist()}

    def goto_pose_posture(
        self,
        position: Any,
        quaternion_wxyz: Any | None = None,
        *,
        waypoints: int = 16,
        tolerance_m: float = 0.01,
        tolerance_rad: float = 0.08,
        max_steps: int = 480,
        hold_posture: bool = True,
        candidate: int = 0,
    ) -> dict[str, Any]:
        """Reach a grip-site pose through a JOINT-SPACE path: solve the arm's IK for the pose
        (``solve_arm_ik``: the joint limits are respected, both finger sides tried), then
        servo the OSC through ``waypoints`` intermediate poses taken from the straight
        joint-space interpolation (forward kinematics of each), and finish on the requested
        pose.  This is the posture option for wrists the plain servo never finds - the
        tool-horizontal-forward pinch of a stove knob or a toaster-oven bar, the side-on
        fingers for a kettle switch: the OSC's own straight Cartesian line ends in a
        joint-limit / elbow local minimum at 0.5-0.9 rad of error while the IK posture
        exists.  With ``hold_posture`` the controller's null-space spring is pointed at the
        IK posture until ``tuck_arm`` (so the elbow stays where the IK put it).  Returns
        ``{"reached", "error_m", "error_rad", "ik" (the solution report), "joint_residual_rad",
        "waypoints_done", "steps", "budget_exhausted"}``; ``reached`` False with
        ``reason`` "no IK solution" when the pose is outside the envelope from here (move the
        base).  Costs ~waypoints x 30 steps.
        """
        s0 = self._step_count
        ik = self.solve_arm_ik(position, quaternion_wxyz)
        if candidate:
            oks = [c for c in ik["candidates"] if c["ok"]]
            if oks:
                if int(candidate) >= len(oks):
                    return {
                        "reached": False,
                        "reason": f"candidate {int(candidate)} of {len(oks)} ok IK "
                        "postures: no other posture to try from this base pose",
                        "error_m": float(
                            np.linalg.norm(np.asarray(position, float) - self.get_eef_pose()[:3])
                        ),
                        "ik": {k: v for k, v in ik.items() if k != "candidates"},
                        "error_rad": None,
                        "joint_residual_rad": None,
                        "waypoints_done": 0,
                        "steps": 0,
                        "budget_exhausted": self._budget_left() <= 0,
                    }
                pick = oks[int(candidate)]
                ik = {**ik, **pick}
        if not ik["ok"]:
            return {
                "reached": False,
                "reason": f"no IK solution from this base pose (best {ik['position_error_m']} m, "
                f"{ik['orientation_error_rad']} rad)",
                "error_m": float(
                    np.linalg.norm(np.asarray(position, float) - self.get_eef_pose()[:3])
                ),
                "ik": {k: v for k, v in ik.items() if k != "candidates"},
                "error_rad": None,
                "joint_residual_rad": None,
                "waypoints_done": 0,
                "steps": 0,
                "budget_exhausted": self._budget_left() <= 0,
            }
        q_t = np.array(ik["q"], dtype=float)
        q_0 = self.arm_joint_positions()
        quat_used = quaternion_wxyz
        if quaternion_wxyz is not None and ik.get("finger_flipped"):
            # the IK settled on the half-turn twin of the finger axis (the same grasp):
            # finish on THAT orientation, or the final servo winds joint 7 through 180 deg
            import robosuite.utils.transform_utils as tu

            r_req = np.asarray(tu.quat2mat(_xyzw(quaternion_wxyz)), dtype=float)
            quat_used = _wxyz_from_mat(r_req @ np.diag([-1.0, -1.0, 1.0]))
        done = 0
        per = max(25, int(max_steps) // max(int(waypoints), 1))
        path_residual: list[float] = []
        left_path = 0
        for k in range(1, int(waypoints) + 1):
            if self._budget_left() <= 0:
                break
            qk = q_0 + (q_t - q_0) * (k / float(waypoints))
            if hold_posture:
                # the null-space spring follows the path, so the redundancy is pulled along
                # the joint-space line instead of drifting into a wrist-wound twin
                self.set_nullspace_posture(qk)
            pk, rk = self._fk_site(qk)
            qk_wxyz = _wxyz_from_mat(rk)
            with contextlib.suppress(RuntimeError):
                self.goto_pose(pk, qk_wxyz, tolerance_m=0.012, tolerance_rad=0.08, max_steps=per)
            path_residual.append(round(float(np.linalg.norm(self.arm_joint_positions() - qk)), 3))
            done = k
            if path_residual[-1] > 0.25:
                # the servo left the joint path (a collision, or a wrist wound into another
                # branch): finishing on the pose from here winds joint 7 - stop and say so
                left_path = k
                break
        if hold_posture:
            # the spring stays on this posture until tuck_arm (which resets it) or the next
            # goto_pose_posture; every later Cartesian servo in the episode is biased by it.
            self.set_nullspace_posture(q_t)
        if left_path:
            pose = self.get_eef_pose()
            return {
                "reached": False,
                "reason": f"left the joint-space path at waypoint {left_path}/{waypoints} "
                f"(residual {path_residual[-1]} rad): a collision or another wrist branch; "
                "retry with candidate=1.. or from another base pose",
                "error_m": round(float(np.linalg.norm(np.asarray(position, float) - pose[:3])), 4),
                "error_rad": None,
                "finger_flipped": bool(ik.get("finger_flipped")),
                "ik": {k: v for k, v in ik.items() if k != "candidates"},
                "joint_residual_rad": round(
                    float(np.linalg.norm(self.arm_joint_positions() - q_t)), 3
                ),
                "waypoints_done": done,
                "path_residual_rad": path_residual,
                "steps": self._step_count - s0,
                "budget_exhausted": self._budget_left() <= 0,
            }
        rep: dict[str, Any] = {}
        try:
            rep = self.goto_pose(
                position,
                quat_used,
                tolerance_m=tolerance_m,
                tolerance_rad=tolerance_rad,
                max_steps=max(60, per),
            )
        except RuntimeError as exc:
            rep = {"reached": False, "reason": str(exc)[:120]}
        pose = self.get_eef_pose()
        err = float(np.linalg.norm(np.asarray(position, dtype=float) - pose[:3]))
        rot_err = None
        if quat_used is not None:
            import robosuite.utils.transform_utils as tu

            e_q = tu.quat_multiply(_xyzw(quat_used), tu.quat_inverse(_xyzw(pose[3:7])))
            if e_q[3] < 0.0:
                e_q = -e_q
            rot_err = float(np.linalg.norm(tu.quat2axisangle(e_q)))
        return {
            "reached": bool(rep.get("reached")),
            "error_m": round(err, 4),
            "error_rad": None if rot_err is None else round(rot_err, 4),
            "finger_flipped": bool(ik.get("finger_flipped")),
            "quaternion_used": None
            if quat_used is None
            else np.asarray(quat_used).round(4).tolist(),
            "ik": {k: v for k, v in ik.items() if k != "candidates"},
            "joint_residual_rad": round(float(np.linalg.norm(self.arm_joint_positions() - q_t)), 3),
            "waypoints_done": done,
            "path_residual_rad": path_residual,
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
            **({"reason": rep["reason"]} if "reason" in rep else {}),
        }

    # ------------------------------------------------------------ knobs

    def _knob_tab(self, geo: dict[str, Any]) -> dict[str, Any] | None:
        """The pinchable tab of a hinged knob: the BOX collision geom on the joint's body
        that reaches farthest along the joint axis (the ridge standing proud of the dial),
        with its world axes and half sizes; ``None`` when the body carries no box.
        """
        raw_m, raw_d = self._raw()
        bodies = self._subtree_bodies(geo["body"])
        best = None
        n_axis = geo["axis"]
        for gid in range(int(raw_m.ngeom)):
            if int(raw_m.geom_bodyid[gid]) not in bodies:
                continue
            if not (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid]):
                continue
            if int(raw_m.geom_type[gid]) != 6:
                continue
            xmat = np.array(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
            size = np.array(raw_m.geom_size[gid], dtype=float)
            centre = np.array(raw_d.geom_xpos[gid], dtype=float)
            ext = float(np.sum(np.abs(xmat.T @ n_axis) * size))
            reach = abs(float((centre - geo["anchor"]) @ n_axis)) + ext
            if best is None or reach > best[0]:
                best = (reach, gid, centre, xmat, size)
        if best is None:
            return None
        _, gid, centre, xmat, size = best
        return {
            "geom": self.sim.model.geom_id2name(gid),
            "center": centre,
            "axes": xmat,
            "half": size,
        }

    def _knob_face(self, geo: dict[str, Any], n: np.ndarray, tab_geom: str) -> float:
        """Signed distance along ``n`` (from the joint anchor) of the dial's face: the farthest
        extent along ``n`` of the joint body's other collision geoms (cylinder, mesh, box).
        """
        raw_m, raw_d = self._raw()
        bodies = self._subtree_bodies(geo["body"])
        face = -1e9
        for gid in range(int(raw_m.ngeom)):
            if int(raw_m.geom_bodyid[gid]) not in bodies:
                continue
            if not (raw_m.geom_contype[gid] or raw_m.geom_conaffinity[gid]):
                continue
            if self.sim.model.geom_id2name(gid) == tab_geom:
                continue
            gtype = int(raw_m.geom_type[gid])
            xmat = np.array(raw_d.geom_xmat[gid], dtype=float).reshape(3, 3)
            size = np.array(raw_m.geom_size[gid], dtype=float)
            centre = np.array(raw_d.geom_xpos[gid], dtype=float)
            if gtype == 7:
                aabb = np.array(raw_m.geom_aabb[gid], dtype=float).reshape(2, 3)
                size = aabb[1]
                centre = centre + xmat @ aabb[0]
                half = np.array([size[0], size[1], size[2]])
            elif gtype == 3:
                half = np.array([size[0], size[0], size[1] + size[0]])
            elif gtype == 5:
                half = np.array([size[0], size[0], size[1]])
            elif gtype == 2:
                half = np.array([size[0]] * 3)
            else:
                half = size
            ext = float(np.sum(np.abs(xmat.T @ n) * half))
            face = max(face, float((centre - geo["anchor"]) @ n) + ext)
        return face

    def turn_knob(
        self,
        fixture: str,
        joint: str,
        target_rad: float | None = None,
        *,
        delta_rad: float | None = None,
        stroke_rad: float = 0.4,
        tolerance_rad: float = 0.04,
        max_strokes: int = 12,
        max_regrasps: int = 3,
        pitch_rad: float = 0.8,
    ) -> dict[str, Any]:
        """Turn a hinged knob (a stove burner, an oven / toaster-oven dial) to ``target_rad``
        of raw joint position (or by ``delta_rad``) by pinching its tab and rolling the wrist.

        The tab is the box geom standing proud of the dial along the joint axis
        (``get_fixture_parts`` lists it as ``*_main`` on the knob body: the stove's is
        1.3 x 1.8 x 3.9 cm, 1.6 cm proud of a 4.7 cm dial at z 0.80 on the front face);
        the tool points INTO the dial along the axis, the fingers close across the tab's
        thinnest side, the pads sit on the proud part with the fingertips 2 mm short of the
        dial face.  The pinch pose is reached through ``goto_pose_posture`` (the
        tool-horizontal-forward wrist the plain servo never finds in front of a counter),
        then the wrist is rolled about the joint axis in strokes of ``stroke_rad`` clamped to
        the travel left, re-reading the joint after each; a stroke that turns the wrist but
        not the knob (slip) releases, backs off along the axis and re-grips with the fingers
        re-centred (up to ``max_regrasps``).  The wrist's remaining roll margin is measured
        and printed into ``history`` but does NOT trigger a re-grip on its own yet.  The
        scorer normalises a knob's qpos (``Stove.get_knobs_state``: q mod 2pi; a burner is on
        for 0.35 <= |q| <= 2pi - 0.35) — ``get_goal`` says which window the task wants, and
        ``target_rad`` here is the RAW joint value to reach.  ``done`` is the reading AFTER
        the release, the one the scorer takes; ``qpos_in_grip`` is the reading with the pinch
        still on, so a turn undone by letting go is visible.  Returns ``{"done", "qpos",
        "qpos_in_grip", "qpos_before", "target_rad", "strokes", "regrasps", "opening_m",
        "tab", "history", "steps", "budget_exhausted"}`` (+ ``reason``).
        """
        self._fixture(fixture)
        if joint not in self.sim.model.joint_names:
            raise KeyError(f"{fixture}: joint {joint!r} is not in the model")
        geo = self._joint_geometry(joint)
        if geo["type"] != "hinge":
            raise RuntimeError(f"{joint}: turn_knob takes hinge joints (got {geo['type']})")
        s0 = self._step_count
        lo, hi = geo["range"]
        q0 = float(geo["qpos"])
        if target_rad is None and delta_rad is None:
            raise ValueError("turn_knob: give target_rad or delta_rad")
        target = float(target_rad) if target_rad is not None else q0 + float(delta_rad)
        target = float(np.clip(target, lo + 0.01, hi - 0.01))
        history: list[str] = []
        tab = self._knob_tab(geo)
        if tab is None:
            return {
                "done": False,
                "qpos": round(q0, 4),
                "qpos_before": round(q0, 4),
                "target_rad": round(target, 4),
                "strokes": 0,
                "regrasps": 0,
                "reason": f"{joint}: no box geom (tab) on the knob body to pinch",
                "steps": 0,
                "budget_exhausted": self._budget_left() <= 0,
            }
        axis = geo["axis"]
        # the dial faces AWAY from the fixture: n along the axis, pointing from the fixture's
        # collision-box centre toward the knob
        raw_m, raw_d = self._raw()
        root = int(raw_m.body_rootid[geo["body"]])
        members = [
            g
            for g in range(int(raw_m.ngeom))
            if int(raw_m.body_rootid[int(raw_m.geom_bodyid[g])]) == root
            and (raw_m.geom_contype[g] or raw_m.geom_conaffinity[g])
        ]
        pts = np.array([raw_d.geom_xpos[g] for g in members], dtype=float)
        box_c = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        n = axis.copy() if float(axis @ (tab["center"] - box_c)) >= 0 else -axis
        up = np.array([0.0, 0.0, 1.0])
        # a knob on a vertical face below a counter: the tool-horizontal wrist only exists
        # with the forearm wound to joint 5's limit (the Omron-mounted Panda cannot pitch
        # joint 6 past 3.75), so the tool is pitched DOWN by ``pitch_rad`` toward the tab
        # (0.8 rad reached with 0.43 rad of limit margin and 0.06 rad of path residual,
        # measured on the stove 2026-08-27); a knob on a horizontal face keeps a vertical tool
        horizontal_face = abs(float(n @ up)) < 0.5
        pitch = float(pitch_rad) if horizontal_face else 0.0
        tool = -n * np.cos(pitch) - up * np.sin(pitch)
        tool /= max(np.linalg.norm(tool), 1e-9)
        # fingers close across the tab's thinnest axis perpendicular to the knob axis
        cands = [i for i in range(3) if abs(float(tab["axes"][:, i] @ n)) < 0.5]
        close_i = min(cands, key=lambda i: tab["half"][i])
        bar_i = max(cands, key=lambda i: tab["half"][i])
        close0 = tab["axes"][:, close_i].copy()
        tab_ext_n = float(np.sum(np.abs(tab["axes"].T @ n) * tab["half"]))
        tab_ext_up = float(np.sum(np.abs(tab["axes"].T @ up) * tab["half"]))
        tab_front = float((tab["center"] - geo["anchor"]) @ n) + tab_ext_n
        face = self._knob_face(geo, n, tab["geom"])
        # measured on the stove tab (2026-08-27): the pinch that held and turned the knob had
        # the grip site 2 mm inside the tab's tip plane along n and 1.2 cm above the tab
        # centre (0.6 of its half height); 4 mm deeper the finger's upper edge met the dial
        # face first and the approach stalled 4-5 cm short every time
        pinch_point = (
            geo["anchor"] + n * (tab_front - 0.002) + up * (0.6 * tab_ext_up * np.sin(pitch))
        )
        tip_depth = float((pinch_point + tool * 0.011 - geo["anchor"]) @ n)
        if tip_depth < face + 0.002:
            back = (face + 0.002 - tip_depth) / max(-float(tool @ n), 1e-6)
            pinch_point = pinch_point - tool * back
        pad_depth = float((pinch_point - tool * 0.004 - geo["anchor"]) @ n)
        pads_on_tab = bool(pad_depth <= tab_front + 0.004)
        history.append(
            f"tab {tab['geom']} half {tab['half'].round(3).tolist()} front {tab_front:.3f} "
            f"face {face:.3f} n {n.round(2).tolist()} pitch {pitch:.2f} -> site "
            f"{pinch_point.round(4).tolist()} pad depth {pad_depth:.3f} pads_on_tab {pads_on_tab}"
        )

        def qpos() -> float:
            return float(self.sim.data.get_joint_qpos(joint))

        def remaining() -> float:
            return target - qpos()

        strokes = regrasps = 0
        opening = 0.0
        last_residual = 0.0
        q_grip = q0
        reason = None
        q_lo, q_hi = self._arm_limits()
        while (
            abs(remaining()) > tolerance_rad and regrasps <= max_regrasps and strokes < max_strokes
        ):
            if self._budget_left() <= 0:
                reason = "budget exhausted"
                break
            # the tab turns with the knob: re-read its close axis, roll the pinch to match
            tab = self._knob_tab(geo) or tab
            close = tab["axes"][:, close_i].copy()
            quat = self.horizontal_grasp_quat(tool, close, nearest=True)
            self.set_gripper(closed=False, steps=6)
            pre = pinch_point - tool * 0.08
            if regrasps > 0 and last_residual > 0.25:
                # the last path left the IK route: start over from home, next candidate
                self.tuck_arm()
                history.append("arm left the IK path: tucked before the re-try")
            rep = self.goto_pose_posture(
                pre, quat, tolerance_m=0.012, tolerance_rad=0.1, candidate=regrasps
            )
            last_residual = float(rep.get("joint_residual_rad") or 0.0)
            history.append(
                f"pre-pose: reached {rep.get('reached')} err {rep.get('error_m')} rad "
                f"{rep.get('error_rad')} ik_ok {rep['ik'].get('ok')} q {rep['ik'].get('q')} "
                f"path {rep.get('path_residual_rad')} residual {rep.get('joint_residual_rad')}"
            )
            if not rep.get("reached") and (rep.get("error_m") or 1.0) > 0.03:
                regrasps += 1
                if not rep["ik"].get("ok"):
                    reason = "no IK posture for the pinch from this base pose (move the base)"
                    break
                continue
            try:
                self.goto_pose(
                    pinch_point, quat, tolerance_m=0.005, tolerance_rad=0.08, max_steps=150
                )
            except RuntimeError as exc:
                history.append(f"pinch approach stalled: {str(exc)[:80]}")
            opening = self.set_gripper(closed=True, steps=10)
            tip = self.get_eef_pose()
            history.append(
                f"pinch at {np.round(tip[:3], 3).tolist()} opening {opening:.4f} "
                f"(tab {2 * tab['half'][close_i]:.3f})"
            )
            # closed on nothing reads ~0.009 m (get_gripper_opening); the pads on the tab
            # read the tab's own thickness.  Halfway between the two is the honest test — a
            # fixed 0.006 was below the empty-close value and never fired.
            if opening < 0.5 * (0.009 + 2.0 * float(tab["half"][close_i])):
                regrasps += 1
                self.set_gripper(closed=False, steps=6)
                with contextlib.suppress(RuntimeError):
                    self.goto_pose(pinch_point - tool * 0.05, quat, tolerance_m=0.02, max_steps=60)
                pinch_point = pinch_point + tool * 0.003
                history.append("closed on nothing: re-grip 3 mm deeper")
                continue
            slipped = False
            while abs(remaining()) > tolerance_rad and strokes < max_strokes:
                if self._budget_left() <= 0:
                    reason = "budget exhausted"
                    break
                before = qpos()
                th = float(np.sign(remaining())) * min(stroke_rad, abs(remaining()) + 0.02)
                tip = self.get_eef_pose()
                quat2 = self._rotate_quat(tip[3:7], axis, th)
                # the pinch point rides with the tab: rotate it about the knob axis too
                site_now = geo["anchor"] + self._rodrigues(tip[:3] - geo["anchor"], axis, th)
                try:
                    rep = self.goto_pose(
                        site_now, quat2, tolerance_m=0.01, tolerance_rad=0.06, max_steps=100
                    )
                except RuntimeError as exc:
                    rep = {"reached": False, "error": str(exc)[:60]}
                strokes += 1
                after = q_grip = qpos()
                q_arm = self.arm_joint_positions()
                margin = float(np.min(np.minimum(q_arm - q_lo, q_hi - q_arm)))
                history.append(
                    f"stroke {strokes} th {th:+.2f}: q {before:.3f} -> {after:.3f} "
                    f"(reached {rep.get('reached')}, wrist margin {margin:.2f})"
                )
                _trace(history[-1])
                if abs(after - before) < 0.25 * abs(th):
                    slipped = True
                    break
            if abs(remaining()) <= tolerance_rad:
                break
            if slipped:
                regrasps += 1
                history.append("the knob did not follow the wrist: release and re-grip")
                self.set_gripper(closed=False, steps=6)
                with contextlib.suppress(RuntimeError):
                    tip = self.get_eef_pose()
                    self.goto_pose(tip[:3] - tool * 0.06, tip[3:7], tolerance_m=0.02, max_steps=60)
        # release and come away along the tool: the first centimetres with the wrist held
        # (a free wrist let the open fingers flick the tab another 0.3 rad, measured)
        self.set_gripper(closed=False, steps=6)
        with contextlib.suppress(RuntimeError):
            tip = self.get_eef_pose()
            self.goto_pose(tip[:3] - tool * 0.04, tip[3:7], tolerance_m=0.01, max_steps=40)
        with contextlib.suppress(RuntimeError):
            tip = self.get_eef_pose()
            self.goto_pose(tip[:3] - tool * 0.06, None, tolerance_m=0.02, max_steps=60)
        self.hold(3)
        q_final = qpos()
        # the release can nudge the knob a few hundredths, so the verdict is the FINAL
        # reading — the one the scorer takes; `qpos_in_grip` is reported beside it so a
        # caller can see a turn that was undone by letting go.
        done = bool(abs(target - q_final) <= tolerance_rad)
        if not done and reason is None:
            if strokes >= max_strokes:
                reason = f"max_strokes reached at qpos {q_final:.3f}"
            elif regrasps > max_regrasps:
                reason = f"regrasps exhausted at qpos {q_final:.3f}"
        out = {
            "done": done,
            "qpos": round(q_final, 4),
            "qpos_in_grip": round(q_grip, 4),
            "qpos_before": round(q0, 4),
            "target_rad": round(target, 4),
            "strokes": strokes,
            "regrasps": regrasps,
            "opening_m": round(opening, 4),
            "tab": {
                "geom": tab["geom"],
                "center": np.asarray(tab["center"]).round(4).tolist(),
                "half_size": np.asarray(tab["half"]).round(4).tolist(),
                "bar_axis": tab["axes"][:, bar_i].round(3).tolist(),
                "close_axis": close0.round(3).tolist(),
                "knob_normal": n.round(3).tolist(),
                "tool": tool.round(3).tolist(),
                "pitch_rad": round(pitch, 3),
                "pinch_point": pinch_point.round(4).tolist(),
            },
            "history": history,
            "steps": self._step_count - s0,
            "budget_exhausted": self._budget_left() <= 0,
        }
        if reason:
            out["reason"] = reason
        return out

    # ------------------------------------------------------------ control

    def _step(self, action_dict: dict[str, Any]) -> None:
        action = self._robot.create_action_vector(action_dict)
        self._obs, _, _, _ = self.env.step(action)
        self._step_count += 1
        if self._capture and self._render:
            self._frames.append(self.render())

    def hold(self, steps: int = 5) -> None:
        """Step the simulation with a zero action (let things settle)."""
        for _ in range(int(steps)):
            self._step({})

    def set_gripper(self, closed: bool, steps: int = 15) -> float:
        """Close (``True``) or open (``False``) the gripper, hold, return the opening."""
        for _ in range(int(steps)):
            self._step({GRIPPER: np.array([1.0 if closed else -1.0])})
        return self.get_gripper_opening()

    def move_base(
        self, forward: float = 0.0, left: float = 0.0, turn: float = 0.0, steps: int = 20
    ) -> np.ndarray:
        """Open-loop base velocities (heading frame, each in [-1, 1]) for ``steps`` steps."""
        cmd = np.clip(np.array([forward, left, turn], dtype=float), -1.0, 1.0)
        for _ in range(int(steps)):
            self._step({BASE: cmd})
        self.hold(3)
        return self.get_base_pose()

    def drive_to(
        self,
        x: float,
        y: float,
        yaw: float | None = None,
        *,
        tolerance_m: float = 0.05,
        tolerance_rad: float = 0.05,
        max_steps: int = 400,
        hold_arm: bool = False,
    ) -> dict[str, Any]:
        """Closed-loop drive of the base to a world ``(x, y[, yaw])``.

        Velocities are proportional to the remaining error (clipped to ±1),
        expressed in the base's heading frame.  Stops early when the base
        stops making progress (blocked by furniture) and reports it.
        """
        target = np.array([x, y], dtype=float)
        last = float("inf")
        stalled = 0
        yaw_err = 0.0
        step = -1
        rel = self._eef_in_base_frame() if hold_arm else None
        for step in range(int(max_steps)):
            bx, by, byaw = self.get_base_pose()
            err_w = target - np.array([bx, by])
            dist = float(np.linalg.norm(err_w))
            yaw_err = (
                0.0 if yaw is None else float(np.arctan2(np.sin(yaw - byaw), np.cos(yaw - byaw)))
            )
            if dist < tolerance_m and abs(yaw_err) < tolerance_rad:
                self.hold(3)
                return {"reached": True, "error_m": dist, "yaw_error": yaw_err, "steps": step}
            progress = dist + abs(yaw_err)
            stalled = stalled + 1 if progress > last - 1e-3 else 0
            last = progress
            if stalled > 30:
                break  # not making progress: blocked by furniture
            err_b = _rot_z(-byaw)[:2, :2] @ err_w  # world → heading frame
            lin = np.clip(err_b * 3.0, -1.0, 1.0)
            ang = np.clip(yaw_err * 2.0, -1.0, 1.0)
            # the base's response is a soft friction knee, not a hard deadband (measured
            # 2026-08-27, cm/step over 20-step bursts: 0.1 → 0.01, 0.2 → 0.02, 0.3 → 0.2,
            # 0.4 → 0.6, 0.5 → 1.0, 1.0 → 2.2-3.5); the turn channel is linear (0.3 →
            # 0.8 deg/step).  Keep the linear command at or above BASE_MIN_CMD
            if dist >= tolerance_m and np.linalg.norm(lin) < BASE_MIN_CMD:
                lin = lin / max(np.linalg.norm(lin), 1e-9) * BASE_MIN_CMD
            if abs(yaw_err) >= tolerance_rad and abs(ang) < BASE_MIN_TURN:
                ang = np.sign(ang) * BASE_MIN_TURN
            action = {BASE: np.array([lin[0], lin[1], ang])}
            if rel is not None:  # servo the arm onto its base-relative pose while moving
                action[ARM] = self._arm_delta_to(
                    np.array([bx, by, 0.0]) + _rot_z(byaw) @ rel[0],
                    self._rotate_quat(rel[1], np.array([0.0, 0.0, 1.0]), byaw),
                )
            self._step(action)
        bx, by, byaw = self.get_base_pose()
        dist = float(np.linalg.norm(target - np.array([bx, by])))
        return {
            "reached": False,
            "blocked": stalled > 30,
            "error_m": dist,
            "yaw_error": yaw_err,
            "steps": step + 1,
            "hint": "the straight line is obstructed: drive around via waypoints "
            "(use get_fixture_extent to find free floor)",
        }

    def _arm_delta_to(self, position: Any, quaternion_wxyz: Any, gain: float = 10.0) -> np.ndarray:
        """One OSC delta (base frame) that servos the grip site toward a world pose."""
        import robosuite.utils.transform_utils as tu

        pose = self.get_eef_pose()
        delta_w = np.asarray(position, dtype=float) - pose[:3]
        err_q = tu.quat_multiply(_xyzw(quaternion_wxyz), tu.quat_inverse(_xyzw(pose[3:7])))
        if err_q[3] < 0.0:
            err_q = -err_q
        axis_angle = np.asarray(tu.quat2axisangle(err_q), dtype=float)
        to_base = _rot_z(-self.get_base_pose()[2])
        lin = np.clip((to_base @ delta_w) * gain, -1.0, 1.0)
        rot = np.clip(to_base @ axis_angle * 2.0, -0.5, 0.5)
        return np.concatenate([lin, rot])

    def goto_pose(
        self,
        position: Any,
        quaternion_wxyz: Any | None = None,
        *,
        tolerance_m: float = 0.01,
        tolerance_rad: float = 0.05,
        max_steps: int = 400,
        gain: float = 10.0,
    ) -> dict[str, Any]:
        """Servo the grip site to a world pose with clipped OSC deltas.

        Returns ``{"reached": bool, "error_m": float, "steps": int}``; raises
        ``RuntimeError`` when the target is not reached within ``max_steps``
        (typically: out of reach — drive the base closer first).
        """
        import robosuite.utils.transform_utils as tu

        target = np.asarray(position, dtype=float)
        target_q = None if quaternion_wxyz is None else _xyzw(quaternion_wxyz)
        last_error = float("inf")
        stalled = 0
        rot_error = 0.0
        for step in range(int(max_steps)):
            pose = self.get_eef_pose()
            delta_w = target - pose[:3]
            error = float(np.linalg.norm(delta_w))
            axis_angle = np.zeros(3)
            if target_q is not None:
                err_q = tu.quat_multiply(target_q, tu.quat_inverse(_xyzw(pose[3:7])))
                if err_q[3] < 0.0:  # shortest rotation (quaternion double cover)
                    err_q = -err_q
                axis_angle = np.asarray(tu.quat2axisangle(err_q), dtype=float)
                rot_error = float(np.linalg.norm(axis_angle))
            if error < tolerance_m and rot_error < tolerance_rad:
                self.hold(2)
                return {"reached": True, "error_m": error, "error_rad": rot_error, "steps": step}
            progress = error + 0.1 * rot_error
            stalled = stalled + 1 if progress > last_error - 1e-4 else 0
            last_error = progress
            if stalled > 40:
                break
            yaw = self.get_base_pose()[2]
            to_base = _rot_z(-yaw)
            lin = np.clip((to_base @ delta_w) * gain, -1.0, 1.0)
            rot = np.clip(to_base @ axis_angle * 2.0, -0.5, 0.5)
            self._step({ARM: np.concatenate([lin, rot])})
        error = float(np.linalg.norm(target - self.get_eef_pose()[:3]))
        if error <= 0.03 and rot_error <= 0.3:
            # a near miss is information, not a failure: let the program decide
            self.hold(2)
            return {"reached": False, "error_m": error, "error_rad": rot_error, "steps": step + 1}
        raise RuntimeError(
            f"goto_pose did not reach the target (error {error:.3f} m, {rot_error:.2f} rad after "
            f"{step + 1} steps); out of reach or in contact: drive the base closer, lower the "
            "target or approach from above"
        )

    # ------------------------------------------------------------ media

    def render(self, camera: str | None = None) -> np.ndarray | None:
        """An RGB frame from ``camera`` (default: the first configured camera)."""
        if not self._render:
            return None
        name = camera or self._camera_names[0]
        h, w = self._camera_size
        frame = self.sim.render(camera_name=name, height=h, width=w)
        return np.asarray(frame)[::-1].copy()

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        """Record a frame after every environment step."""
        self._capture = bool(enabled)
        if clear:
            self._frames.clear()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        """Frames captured since the last reset/clear."""
        frames = list(self._frames)
        if clear:
            self._frames.clear()
        return frames
