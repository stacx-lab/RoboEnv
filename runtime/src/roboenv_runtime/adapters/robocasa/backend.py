"""One RoboCasa world, native control intervals, and permitted measurements."""

from __future__ import annotations

from typing import Any
import os
import contextlib
import numpy as np

DEFAULT_ROBOT = "PandaOmron"
DEFAULT_TASK = "PickPlaceCounterToCabinet"
DEFAULT_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def _ensure_offscreen_rendering():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def list_robocasa_tasks():
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS

    return sorted(ALL_KITCHEN_ENVIRONMENTS)


def _xyzw(q):
    return np.asarray(q)[[1, 2, 3, 0]]


class RoboCasaEnv:
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
        camera_names: tuple[str, ...] = DEFAULT_CAMERAS,
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
            "camera_depths": bool(enable_render),
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

    def close(self) -> None:
        """Close the simulation."""
        self.env.close()

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

    def task_completed(self) -> bool:
        """The benchmark's own success check."""
        return bool(self.env._check_success())

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

    def fixture_regions(self, name: str) -> dict[str, Any]:
        """Interior placement regions of a fixture (tray, shelf, rack ...) as world boxes:
        the exact volumes RoboCasa samples and checks objects in.
        """
        import robosuite.utils.transform_utils as tu

        fixture = self.env.fixtures[name]
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

    def render(self, camera: str | None = None) -> np.ndarray | None:
        """An RGB frame from ``camera`` (default: the first configured camera)."""
        if not self._render:
            return None
        name = camera or self._camera_names[0]
        h, w = self._camera_size
        frame = self.sim.render(camera_name=name, height=h, width=w)
        return np.asarray(frame)[::-1].copy()

    def render_rgbd(self, camera: str):
        """Aligned upright RGB and optical-axis depth in metres, one render."""
        from robosuite.utils.camera_utils import get_real_depth_map

        if not self._render or camera not in self._camera_names:
            raise ValueError("Camera must be enabled in the task's simulator configuration")
        h, w = self._camera_size
        rgb, depth_buffer = self.sim.render(camera_name=camera, height=h, width=w, depth=True)
        raw = np.asarray(depth_buffer, dtype=float)
        # Far-plane/background pixels and invalid renderer values have no depth.
        valid = np.isfinite(raw) & (raw >= 0) & (raw < 1)
        metres = get_real_depth_map(self.sim, np.where(valid, raw, 0))
        metres = np.where(valid, metres, 0)
        return np.asarray(rgb)[::-1].copy(), metres[::-1].copy()

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed = seed
            self.env.rng = np.random.default_rng(seed)
        self._obs = self.env.reset()
        self._step_count = 0
        self._frames.clear()
        return self._obs

    def step(self, commands):
        self.step_vector(self._robot.create_action_vector(commands))

    def step_vector(self, action):
        # PandaGripper accumulates normalized commands internally. Start every
        # bounded interval from neutral, including intervals for other devices.
        self._robot.gripper["right"].current_action[:] = 0
        self._obs, _, _, _ = self.env.step(action)
        self._step_count += 1
