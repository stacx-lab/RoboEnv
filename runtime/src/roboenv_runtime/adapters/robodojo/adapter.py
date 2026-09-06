"""RoboDojo (Isaac Sim; dual ARX X5 or dual I2RT YAM) adapter.

The heavy lifting — in-process Isaac boot, IK/servo motion stack, fingertip
calibration, score pumping — lives in :mod:`roboenv_runtime.adapters.robodojo.backend`
(``RoboDojoEnv``).  This adapter imports it lazily and exposes its
capabilities through the ``@tool`` registry, with docstrings as the single
source of API documentation; prompt text comes from the embodiment profile.

Requires the ``RoboDojo`` conda environment on an RTX node.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from roboenv_runtime.adapters.base import ensure_capx_importable
from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.tools import tool
from roboenv_runtime.core.types import Budget, Capability, ScoreReport
from roboenv_runtime.embodiments import load_embodiment

__all__ = ["RoboDojoAdapter"]

EMBODIMENT_BY_ENV_CFG = {"arx_x5": "arx_x5", "yam": "yam"}


@register_adapter
class RoboDojoAdapter(AdapterBase):
    """RoboDojo task ``task_name``; ``env_cfg_type`` selects the embodiment."""

    name: ClassVar[str] = "robodojo"
    embodiment: ClassVar[str] = "arx_x5"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            Capability.PRIVILEGED,
            Capability.VIDEO,
            Capability.VLA,
            Capability.FUNCTIONAL_POINTS,
            Capability.OFFICIAL_BUDGET,
        }
    )

    def __init__(self, task_name: str = "pack_objects_into_box", **env_kwargs: Any):
        ensure_capx_importable()
        from roboenv_runtime.adapters.robodojo.backend import RoboDojoEnv  # heavy: boots Isaac

        self.task_name = task_name
        self._env = RoboDojoEnv(task_name=task_name, **env_kwargs)
        env_cfg = getattr(self._env, "env_cfg_type", "arx_x5")
        self.embodiment = EMBODIMENT_BY_ENV_CFG.get(env_cfg, env_cfg)
        self._home_poses: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- context

    @property
    def is_yam(self) -> bool:
        """True when the YAM embodiment is loaded."""
        return self.embodiment == "yam"

    @property
    def skill_domain(self) -> str:
        """Skill domain (kept for pre-refactor callers; equals ``embodiment``)."""
        return self.embodiment

    @property
    def profile(self):
        """The embodiment profile (``roboenv/embodiments/<embodiment>.yaml``)."""
        return load_embodiment(self.embodiment)

    def situation(self) -> str:
        """Embodiment situation text plus the task's language instruction (after reset)."""
        base = self.profile.render_situation().rstrip()
        goal = self.goal()
        return base + (f"\nBenchmark instruction: {goal}" if goal else "") + "\n"

    def goal(self) -> str:
        """The task's own language instruction (``gen_instruction``), when it has one."""
        try:
            task_env = self._env._env
            templates = task_env.gen_instruction(0)
        except Exception:
            return ""
        if isinstance(templates, str):
            return templates
        return " / ".join(str(t) for t in templates) if templates else ""

    def facts(self) -> str:
        """Embodiment facts plus the auto-measured world digest."""
        return self.profile.render_facts() + self.world_digest()

    def world_digest(self) -> str:
        """Auto-generated per-episode WORLD DIGEST.

        Everything an agent always spends its first turns discovering —
        objects with pose, extent, semantic category, and a per-arm
        reachability probe of each object's top.  Costs seconds at reset,
        saves 3-8 exploration turns every episode.  Never raises.
        """
        try:
            lines = ["", "WORLD DIGEST (auto-measured at reset; verify before critical moves):"]
            for name in self._env.list_objects():
                try:
                    pos, _ = self._env.get_object_pose(name)
                    ext, top_z = self._env._bbox_ext_top(name)
                    cat = ""
                    try:
                        fp = self._env._env.reward_manager.func_parser
                        category = fp.get_category_by_label(label=name, env_idx=0)
                        if category == "number":
                            idx = fp.get_label_cat_index(labels=[name])[0]
                            idx = idx[0] if hasattr(idx, "__len__") else idx
                            cat = f" sem=number:{int(idx) % 10}"
                        elif category:
                            cat = f" sem={category}"
                    except Exception:
                        pass
                    reach = []
                    top = [float(pos[0]), float(pos[1]), top_z + 0.09]
                    for arm in ("left", "right"):
                        try:
                            r = self._env.check_reachable(arm, top)
                            reach.append(arm[0].upper() if r.get("reachable") else "-")
                        except Exception:
                            reach.append("?")
                    lines.append(
                        f"- {name}: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                        f" ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}){cat}"
                        f" hover-reach={''.join(reach)}"
                    )
                except Exception as exc:
                    lines.append(f"- {name}: (read failed: {type(exc).__name__}: {exc})")
            try:
                fixtures = [r.get("label") for r in self._env._records(self._env.FIXTURE_GROUPS)]
                for name in dict.fromkeys(n for n in fixtures if n):
                    try:
                        pos, _ = self._env.get_object_pose(name)
                        ext, top_z = self._env._bbox_ext_top(name)
                        lines.append(
                            f"- {name} [FIXED prop, absent from list_objects() but queryable by "
                            f"this exact name]: pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                            f" ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f}) top_z={top_z:.3f}"
                        )
                    except Exception as exc:
                        lines.append(f"- {name}: (fixture read failed: {type(exc).__name__})")
            except Exception:
                pass
            return "\n".join(lines) + "\n"
        except Exception:
            return ""

    # ---------------------------------------------------------------- runtime hooks

    def score_authority(self, run_dir: Path, **options: Any) -> Any:
        """The official reward clock (``ScoreClock``) for the current episode."""
        from roboenv_runtime.adapters.robodojo.score_clock import ScoreClock

        return ScoreClock(self, collect_interval=options.get("collect_interval"))

    def vla_manager(self, scorer: Any, run_dir: Path) -> Any:
        """Bounded VLA-as-a-tool manager owned by a worker (``Capability.VLA``)."""
        from roboenv_runtime.adapters.robodojo.vla_manager import VlaManager

        return VlaManager(self, scorer, run_dir)

    def observe(self) -> dict[str, Any]:
        """Ground truth plus geometry, semantics, gripper calibration and home poses.

        This is the observation primitives and remote clients see; it carries the
        per-object ``geometry`` (dimensions, half extents, semantics) and the
        embodiment's ``gripper_calibration`` so nothing has to reach past the
        tool boundary for them.
        """
        objects: dict[str, Any] = {}
        geometry: dict[str, Any] = {}
        for label in self._env.list_objects():
            item: dict[str, Any] = {}
            try:
                pos, quat = self._env.get_object_pose(label)
                item["position"] = [float(v) for v in list(pos)[:3]]
                item["quaternion_wxyz"] = [float(v) for v in list(quat)[:4]]
            except Exception as exc:
                item["pose_error"] = f"{type(exc).__name__}: {exc}"
            try:
                extent = [float(v) for v in self.get_object_bbox_extent(label)]
                item["dimensions_m"] = extent
            except Exception as exc:
                item["extent_error"] = f"{type(exc).__name__}: {exc}"
            with contextlib.suppress(Exception):
                item["semantics"] = self.get_object_semantics(label)
            objects[label] = item
            geom = {
                k: item[k] for k in ("position", "quaternion_wxyz", "dimensions_m") if k in item
            }
            if "dimensions_m" in geom:
                geom["half_extents_m"] = [v / 2.0 for v in geom["dimensions_m"]]
            if "semantics" in item:
                geom["semantics"] = item["semantics"]
            geometry[label] = geom
        grippers: dict[str, Any] = {}
        calibration = self._gripper_calibration()
        for arm in ("left", "right"):
            entry: dict[str, Any] = {}
            try:
                pose = [float(v) for v in self._env.get_ee_pose(arm)]
                entry["tip_position"], entry["quaternion_wxyz"] = pose[:3], pose[3:7]
            except Exception as exc:
                entry["pose_error"] = f"{type(exc).__name__}: {exc}"
            try:
                state = self.get_gripper_state(arm)
                entry["finger_a"] = [float(v) for v in state["finger_a"]]
                entry["finger_b"] = [float(v) for v in state["finger_b"]]
                width = float(state["width"])
                entry["width_m"] = width
                cal = calibration[arm]
                slope = cal.get("affine_slope_m")
                if slope:
                    fraction = (width - float(cal["affine_intercept_m"])) / float(slope)
                else:
                    span = float(cal["open_width_m"]) - float(cal["closed_width_m"])
                    fraction = (width - float(cal["closed_width_m"])) / span if span > 0 else 0.0
                entry["open_fraction"] = max(0.0, min(1.0, fraction))
            except Exception as exc:
                entry["state_error"] = f"{type(exc).__name__}: {exc}"
            grippers[arm] = entry
        return {
            "objects": objects,
            "geometry": geometry,
            "grippers": grippers,
            "gripper_calibration": calibration,
            "home_poses": dict(self._home_poses),
            "budget": self.budget().as_dict(),
        }

    def _gripper_calibration(self) -> dict[str, dict[str, Any]]:
        """Jaw-width model of the loaded embodiment, from its profile."""
        gripper = self.profile.gripper
        if self.embodiment == "arx_x5":
            cal: dict[str, Any] = {
                "closed_width_m": float(gripper.get("gw_saturated_m", 0.0498)),
                "open_width_m": 0.1378,
                "command_fraction_saturation": 0.2,
                "affine_intercept_m": float(gripper.get("gw_intercept_m", 0.0299)),
                "affine_slope_m": float(gripper.get("gw_per_frac_m", 0.1079)),
                "width_model": "affine_with_closed_saturation",
                "source": "arx_x5 profile (V20 controlled measurement)",
            }
        else:
            cal = {
                "closed_width_m": 0.0,
                "open_width_m": float(gripper.get("max_opening_m", 0.094)),
                "width_model": "linear_endpoint_estimate",
                "source": f"{self.embodiment} profile (documented open span)",
            }
        return {"left": dict(cal), "right": dict(cal)}

    def _capture_home_poses(self) -> None:
        self._home_poses = {}
        for arm in ("left", "right"):
            try:
                pose = [float(v) for v in self._env.get_ee_pose(arm)]
            except Exception:
                continue
            self._home_poses[arm] = {"position": pose[:3], "quaternion_wxyz": pose[3:7]}

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int) -> None:
        """Start a fresh episode with video capture on.

        Capture is enabled AFTER the environment reset: the backend records the start
        frame the moment capture is switched on and its reset clears the buffer, so the
        old order lost the initial-layout frame of every episode and left ``capture()``
        with nothing until the first motion.
        """
        self._env.reset(seed=seed)
        with contextlib.suppress(Exception):
            self._env.enable_video_capture(True, clear=True)
        self._capture_home_poses()

    def score(self) -> ScoreReport:
        """Reward, graded tier and the official completion flag."""
        reward = float(self._env.compute_reward())
        rm = self._env._env.reward_manager
        tier = int(rm.score_completed_count[0]) if rm.score_completed_count else 0
        completed = bool(self._env.task_completed())
        # some tasks define only the binary completion check and no graded
        # get_score stages — completion there IS full marks, not zero
        if completed and reward == 0.0:
            reward = 1.0
        return ScoreReport(reward=reward, raw_score=reward * 100.0, tier=tier, completed=completed)

    def budget(self) -> Budget:
        """Simulation steps consumed vs the episode cap."""
        raw = self._env.get_step_budget()
        return Budget(used=int(raw["used"]), max=int(raw["max"]), unit="sim_steps")

    def frames(self) -> list[Any]:
        """Captured head-camera frames for the episode."""
        try:
            return list(self._env.get_video_frames())
        except Exception:
            return []

    def capture(self) -> Any | None:
        """A head-camera image of the CURRENT scene: a fresh render (no physics step, no
        budget) or, only when rendering fails, the most recent recorded video frame
        (recorded every 40 sim steps, so it can lag the arm by up to 40 steps).  None
        when neither exists; the backend prints the reason once.
        """
        frame = None
        try:
            frame = self._env.render()
        except Exception as exc:
            print(f"[robodojo] live render failed: {exc!r}", flush=True)
        if frame is not None:
            return frame
        try:
            return self._env.last_video_frame()
        except Exception:
            return None

    def close(self) -> None:
        """Stop the VLA policy-server subprocess (Isaac itself is per-process)."""
        with contextlib.suppress(Exception):
            self._env._stop_vla()

    # ---------------------------------------------------------------- tools
    # Docstrings below ARE the agent-facing API documentation.

    @tool
    def list_objects(self) -> list[str]:
        """List the names of the manipulable objects in the scene.

        Fixed scene props are not listed (but remain queryable by name).

        Returns:
            Names usable with get_object_pose()/get_object_bbox_extent().
        """
        return self._env.list_objects()

    @tool
    def get_object_pose(self, object_name: str):
        """Ground-truth pose of an object.

        Returns:
            (position (3,), quaternion_wxyz (4,)) in world frame.
        """
        return self._env.get_object_pose(object_name)

    @tool
    def get_object_category(self, object_name: str) -> int:
        """The object's asset CATEGORY INDEX from the scene's privileged
        metadata — its semantic identity as an integer. Same asset family
        shares an index scheme (e.g. digit assets encode their face value
        as index % 10). Use it to tell semantically different objects
        apart when names alone (digit_0, digit_1...) don't say what they
        show.
        """
        fp = self._env._env.reward_manager.func_parser
        idx = fp.get_label_cat_index(labels=[object_name])[0]
        try:
            return int(idx[0])
        except (TypeError, IndexError):
            return int(idx)

    @tool
    def get_object_bbox_extent(self, object_name: str):
        """Axis-aligned bounding-box side lengths (3,) of an object, meters.
        Full side lengths: half-height is extent[2]/2.
        """
        verts = np.asarray(self._env.get_object_bbox(object_name))
        return verts.max(axis=0) - verts.min(axis=0)

    @tool
    def get_gripper_pose(self, arm: str):
        """Current fingertip pose of one arm ("left"/"right"):
        (7,) [x, y, z, qw, qx, qy, qz] world frame — the same frame
        goto_pose() targets.
        """
        return self._env.get_ee_pose(arm)

    @tool
    def get_gripper_state(self, arm: str) -> dict:
        """Observe the physical gripper: both finger world positions and the
        jaw width. Use it to CALIBRATE the gripper frame empirically — command
        a test orientation, read this, and the finger-to-finger direction is
        the jaw opening axis while wrist-to-tips is the tool axis.

        Returns:
            {"finger_a": (3,), "finger_b": (3,), "width": float,
             "tip_pose": (7,)}
        """
        fa, fb = self._env.get_finger_positions(arm)
        pose = self._env.get_ee_pose(arm)
        return {
            "finger_a": fa,
            "finger_b": fb,
            "width": float(np.linalg.norm(np.asarray(fa) - np.asarray(fb))),
            "tip_pose": pose,
        }

    @tool
    def goto_pose(
        self,
        arm: str,
        position,
        quaternion_wxyz,
        max_tilt_deg: float = 0.0,
        max_yaw_deg: float = 0.0,
    ):
        """Move one arm so its FINGERTIPS reach a world-frame pose.

        The motion follows an approximately STRAIGHT Cartesian path with NO
        automatic obstacle avoidance: plan your own clearance waypoints (up,
        across, then down), including the object hanging below the jaws.

        WARNING: this gripper's tool-frame convention is NOT the common
        "[0,1,0,0] points down" one — calibrate with get_gripper_state().
        This small 6-DoF arm often cannot reach an exactly vertical wrist
        away from its base: pass max_tilt_deg=25..30 when approaching or
        grasping, and the motion uses the nearest reachable orientation.
        Reach is also strongly YAW-dependent: when the jaw yaw is free
        (repositioning moves, symmetric objects) pass max_yaw_deg=90 or 180
        and unreachable edge poses often become reachable — the sweep is
        automatic, nearest-yaw first.

        Returns {'position_error_m', 'orientation_dev_deg'}. POSITION is
        asserted (RuntimeError beyond 3 cm, or when no IK solution exists
        from the arm's current posture — catch it and try a nearby pose or
        the other arm); ORIENTATION is only REPORTED: read
        orientation_dev_deg after every move, a value above ~10 means the
        wrist could not take that orientation there (measured: 35-67 deg at
        the workspace edge while the position was met).
        """
        return self._env.goto_pose(
            arm, position, quaternion_wxyz, max_tilt_deg=max_tilt_deg, max_yaw_deg=max_yaw_deg
        )

    @tool
    def move_by_displacement(self, arm: str, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Move one arm's fingertips by a world-frame offset, keeping the
        current orientation. For pure upward lifts, progressively shortens
        the lift instead of failing outright.
        """
        pose = self._env.get_ee_pose(arm)
        last_err: Exception | None = None
        scales = [1.0, 0.6, 0.35] if (x == 0.0 and y == 0.0 and z > 0.0) else [1.0]
        for sc in scales:
            try:
                self._env.goto_pose(
                    arm,
                    np.asarray(pose[:3]) + np.array([x * sc, y * sc, z * sc]),
                    pose[3:7],
                    max_tilt_deg=30.0,
                )
                return
            except RuntimeError as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    @tool
    def open_gripper(self, arm: str, fraction: float = 1.0):
        """Open one gripper (1.0 = fully open)."""
        self._env.set_gripper(arm, fraction)

    @tool
    def close_gripper(self, arm: str, fraction: float = 0.0):
        """Close one gripper onto an object (0.0 = fully closed)."""
        self._env.set_gripper(arm, fraction)

    @tool
    def check_reachable(self, arm: str, position, quaternion_wxyz=None) -> dict:
        """Plan-only probe (nothing moves): can this arm's fingertips reach
        the position? Tries the requested (or default top-down) orientation
        plus tilts and the wrist-flip branch — the same ladder goto_pose
        uses. Returns {'reachable': bool, 'quat_wxyz': working orientation}:
        compare quat_wxyz with what you asked for — 'reachable' can be True
        through a TILTED candidate while the exact orientation has no
        solution. Judged from the arm's CURRENT posture (a seeded solve), so
        the answer can change after go_home. Probe BEFORE committing an arm
        so failed motions never burn steps.
        """
        return self._env.check_reachable(arm, position, quaternion_wxyz)

    @tool
    def pick(self, arm: str, object_name: str, lift: float = 0.10) -> dict:
        """Pick a rigid object in ONE call: bbox-informed pinch ladder
        (center pinch, then one-rim pinch on wide objects), reachability
        pre-probe, top-down grasp with tilt/branch fallback, then an HONEST
        lift verify — returns only if the object actually rose with the
        gripper. Returns {'grasped': True, 'offset', 'quat_wxyz'}; raises
        with guidance if no pinch holds (try the other arm).
        """
        return self._env.pick(arm, object_name, lift=lift)

    @tool
    def place_at(self, arm: str, position, hover: float = 0.08, release_dz: float = 0.02):
        """Place the held object: carry above the target, descend, open,
        retreat. position = world xyz where the object's bottom should land.
        Verify the object's final pose yourself afterwards.
        """
        return self._env.place_at(arm, position, hover=hover, release_dz=release_dz)

    @tool
    def get_object_semantics(self, object_name: str) -> dict:
        """The object's SEMANTIC identity from privileged scene metadata:
        {"category": <str>} plus {"digit": 0-9} for number tiles.

        IMPORTANT: a label like num_3 does NOT mean digit 3 — digit identity
        lives only in this metadata. Language/number tasks (equations,
        arranging, pressing by number, classifying) are scored against THIS
        identity; always decode every candidate object before reasoning.
        """
        fp = self._env._env.reward_manager.func_parser
        out: dict[str, Any] = {"category": fp.get_category_by_label(label=object_name, env_idx=0)}
        if out["category"] == "number":
            idx = fp.get_label_cat_index([object_name])[0]
            idx = idx[0] if hasattr(idx, "__len__") else idx
            if idx is not None:
                out["digit"] = int(idx) % 10
        return out

    @tool
    def vla_run(
        self,
        n_actions: int = 25,
        mode: str = "policy",
        allow_midstate: bool = False,
        reserve_sim_steps: int = 0,
        stop_when_latched: bool = True,
    ):
        """Run one of two bounded, allowlisted native-action modes.

        mode="policy" hands closed-loop control to G05 - a pretrained
        vision-language-action policy - for up to n_actions frames and returns
        its before/after report. Hold-only switches must retain their defaults.

        mode="hold" sends only the finite canonical X5 joint target already
        present in the current public observation. It cannot accept raw actions,
        trajectories, replay, sub-goals, or hidden support choices. This advances
        task-native support/opponent callbacks while keeping both robot arms still.
        It refuses a mid-episode handoff unless allow_midstate=True, stops before
        reserve_sim_steps, and stops on a success latch by default. The sanitized
        result reports actions, public object motion, budget, stop reason, and a hash
        of the observed hold actions.

        When to use it: contact-rich dexterous stretches where coded primitives
        keep failing - tower/block stacking, precision insertion, pouring into
        narrow mouths, bottle drops. It runs the task's TRAINED instruction; it
        cannot be told sub-goals, and it is BAD at language grounding (which
        button/number/category) - do the reasoning yourself, use it for motor
        skill. Typical use: let it run 20-40 frames, verify object poses with
        ground truth, take over for the endgame (go_home etc.) yourself.
        First call boots the policy server (~2-4 min, once per episode).
        """
        if (
            isinstance(n_actions, bool)
            or not isinstance(n_actions, int)
            or not 1 <= n_actions <= 2_000
        ):
            raise ValueError("n_actions must be an integer in 1..2000")
        if mode == "hold":
            hold = getattr(self._env, "neutral_hold", None)
            if not callable(hold):
                raise RuntimeError("this RoboDojo backend has no neutral-hold capability")
            return hold(
                n_actions=n_actions,
                allow_midstate=allow_midstate,
                reserve_sim_steps=reserve_sim_steps,
                stop_when_latched=stop_when_latched,
            )
        if mode != "policy":
            raise ValueError("vla_run mode must be exactly 'policy' or 'hold'")
        if allow_midstate is not False or reserve_sim_steps != 0 or stop_when_latched is not True:
            raise ValueError(
                "allow_midstate, reserve_sim_steps, and stop_when_latched are "
                "available only with mode='hold'"
            )
        return self._env.vla_run(n_actions=n_actions)

    @tool
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
    ) -> dict:
        """Run bounded continuous G05 replanning with public per-action event guards.

        ``candidate_labels`` enables a conservative stable-owner witness based
        only on public object/gripper poses, object rise, jaw width, nearest-arm
        separation, and consecutive object-in-hand SE(3) consistency.
        ``protected_labels`` stops the plan as soon as a protected public pose
        exceeds its displacement or rotation bound. Protection is checked first.

        The allowlist accepts only exact labels returned by ``list_objects`` and
        at most 110 total actions. Each non-empty returned plan is consumed action by
        action under the same guards; if it is exhausted before an event, the latest
        public observation is submitted for another plan without resetting the policy
        client. It returns
        the actual inference count plus a sanitized witness, counts, hashes, public
        motion, and budget; never raw policy actions or hidden scorer state. At
        least one candidate or protected label is required.
        """
        event_run = getattr(self._env, "vla_run_until_event", None)
        if not callable(event_run):
            raise RuntimeError("this RoboDojo backend has no policy-event capability")
        return event_run(
            n_actions=n_actions,
            candidate_labels=candidate_labels,
            protected_labels=protected_labels,
            allow_midstate=allow_midstate,
            reserve_sim_steps=reserve_sim_steps,
            stop_when_latched=stop_when_latched,
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

    @tool
    def press(self, arm: str, label: str, depth: float = 0.015):
        """Press down on a button/key/switch object with the closed gripper
        tip and retreat; returns {"pressed": bool, "dz": how far it sank}.
        """
        return self._env.press(arm, label, depth=depth)

    @tool
    def pour(self, arm: str, angle_deg: float = 110.0, axis: int = 0):
        """Pour from the held container: rotate it in place by angle_deg
        about world X (axis=0) or Y (axis=1), dwell, wiggle, rotate back.
        Position the container 2-6 cm ABOVE the receiver mouth first.
        max_tilt on goto_pose does NOT pour — this does.
        """
        return self._env.pour(arm, angle_deg=angle_deg, axis=axis)

    @tool
    def wait(self, steps: int = 30):
        """Let the simulation settle for N control steps (e.g. after a
        release, before re-reading poses).
        """
        self._env.settle(steps)

    @tool
    def go_home(self, arm: str):
        """Return one arm to its initial configuration. Scoring end-state
        gates require BOTH arms home and both grippers open.
        """
        self._env.go_home(arm)

    # ------------------------------------------------------------ closed-loop primitives
    @property
    def _prims(self):
        from roboenv_runtime.adapters.robodojo.primitives import DojoPrimitives

        cache = getattr(self, "_prims_cache", None)
        if cache is None or cache.env is not self._env:
            cache = self._prims_cache = DojoPrimitives(self._env)
        return cache

    @tool
    def get_joint_states(self, object_name: str | None = None) -> dict:
        """Joints of ARTICULATED objects (lids, doors, drawers, levers, buttons):
        {label: {"joints": {name: {position, lower, upper, ratio}}, "functional_joints":
        {tag: joint}}}. The scorer's "open/closed" checks are joint-RATIO thresholds
        (e.g. ratio > 0.85), and the tag names are what the task text refers to.
        """
        return self._prims.joint_states(object_name)

    @tool
    def get_functional_points(self, object_name: str) -> dict:
        """World poses [x,y,z,qw,qx,qy,qz] of the object's functional frames, exactly as
        the scorer resolves them: "passive" = where things go (slots, cradles, hinges),
        "active" = the object's own points (bottom, bridge, tip). Distance/alignment
        checks in the task are between these points, not object centres.
        """
        return self._prims.functional_points(object_name)

    @tool
    def get_support_points(self, object_name: str) -> dict:
        """World poses and radii of generic active/passive support metadata tags.

        Returns only active/passive tag mappings with poses and radii. All tags for the
        exact public object label are included; articulated base-link annotations are
        resolved by the layout through the object's USD path. No reward/scorer predicate,
        selected target, object ID, private instance name, metadata, or action queue is
        exposed.
        """
        raw = self._prims.support_points(object_name)
        if not isinstance(raw, dict):
            raise RuntimeError("support-point primitive returned a non-mapping result")
        result = {"active": {}, "passive": {}}
        for kind in ("active", "passive"):
            section = raw.get(kind)
            if not isinstance(section, dict):
                raise RuntimeError(f"support-point primitive omitted {kind!r}")
            for tag, item in section.items():
                if not isinstance(tag, str) or not isinstance(item, dict):
                    raise RuntimeError(
                        "support-point primitive returned malformed tag data"
                    )
                result[kind][tag] = {
                    "poses": item.get("poses"),
                    "radii": item.get("radii"),
                }
        return result

    @tool
    def get_garment_landmarks(self, object_name: str) -> dict:
        """Live env-relative XYZ centroids of a GARMENT's named semantic points.

        Returns only {tag: [x, y, z]} for the asset's non-empty, generic
        landmarks (for example sleeves, chest, hem and shoulders). Every call
        resamples the deforming cloth, so use it to verify whether a pinched point
        followed the gripper. Non-garments and malformed annotations are refused.
        Vertex IDs, qpos/orientations, the full mesh, scorer predicates, thresholds,
        target pairings and residuals are never exposed.
        """
        return self._prims.garment_landmarks(object_name)

    @tool
    def get_fluid_particles(self, object_name: str, cap: int = 512) -> dict:
        """Read a bounded live particle cloud for a FLUID public label.

        Returns ``{"frame": "env_world", "count", "sample_count", "downsampled",
        "points", "centroid", "aabb": {"min", "max"}}``. ``points`` is finite,
        deterministic, and capped; count/centroid/AABB summarize the complete live
        cloud. The tool returns no particle IDs, internal instance/prim names, target
        association, task predicate, or reward state. ``cap`` must be an integer in
        ``1..4096``. Non-fluid and malformed labels fail closed.
        """
        return self._prims.fluid_particles(object_name, cap=cap)

    @tool
    def get_mass_properties(self, object_name: str) -> dict:
        """Mass and centre of mass of a rigid object from the physics engine:
        {"mass_kg", "com_world", "com_local", "com_offset" (world vector from the
        object's origin - the pose get_object_pose reports, the origin of its
        functional-point frames - to its centre of mass)}. A friction pinch holds a flat tool
        level only when the pinch line passes within a few mm of the centre of mass
        (measured: the wrench's is 1.95 cm from its origin toward the head - pinched at
        the origin it turned 14-26 deg in the pads, pinched at the centre of mass 0.2
        deg). pick_flat(along="com") uses this. Raises for fixed props/articulations.
        """
        return self._prims.mass_properties(object_name)

    @tool
    def get_object_profile(self, object_name: str, step: float = 0.01) -> dict:
        """Cross-section profile of an object from its mesh along ITS OWN longest
        horizontal bounding-box edge, whatever its yaw: {"along_dir", "across_dir" (world
        [x, y] unit vectors - along = cross(z, jaw), pick_flat's along_axis convention, so
        a band's "along" can be passed to pick_flat(along=) directly), "bands":
        [{"along" (from the origin), "across_min", "across_max", "width", "z_min",
        "z_max"}] (empty bands are skipped), "origin", "com_along" (None, with a
        "com_error", for props without mass properties)}. Use it to choose a pinch site:
        parallel flat sides (a constant width over the 6 cm pad span) hold; diverging or
        rounded parts touch each pad at one point and roll about the jaw axis (pliers
        handles, a round joint boss - measured). Nothing moves; ~0.5 s for a 200k-vertex
        mesh; step must be > 0.
        """
        return self._prims.object_profile(object_name, step=step)

    @tool
    def reach_orientations(
        self,
        arm: str,
        position,
        jaw: str | list = "x",
        tilts: list | None = None,
        yaws: list | None = None,
    ) -> dict:
        """Which wrist LEANS (tool pitch about the jaw axis, degrees; leaning toward
        the arm where the jaw allows, sideways when the jaw points at the arm - read
        "tool_dir") and jaw YAWS have an IK solution for the fingertips at `position`
        ([x, y, z] or a 7-vector pose) - plan-only, EXACT orientation, judged from the
        arm's current posture. Returns {"feasible": [{"tilt_deg", "yaw_deg",
        "quat_wxyz", "tool_dir"}], "feasible_flipped": entries the probe could only
        solve with the wrist flipped 180 deg (they lean the OTHER way - not feasible as
        asked), "infeasible": [...], "best": the smallest feasible lean}. Defaults:
        tilts 0/10/20/30/40, yaws 0; a negative tilt leans away from the arm; each
        candidate costs ~0.3-0.5 s, keep tilts x yaws to ~15. "IK exists" is what this
        reports; goto_pose's orientation_dev_deg afterwards says whether the controller
        got there. Probe a point BEFORE planning a flat pick or seat there (the exact
        tool-down wrist has no solution over the table middle for either X5 arm; a
        20-30 deg lean has).
        """
        kwargs: dict[str, Any] = {}
        if tilts is not None:
            kwargs["tilts"] = [float(t) for t in np.atleast_1d(np.asarray(tilts, dtype=float))]
        if yaws is not None:
            kwargs["yaws"] = [float(y) for y in np.atleast_1d(np.asarray(yaws, dtype=float))]
        return self._prims.reach_orientations(arm, position, jaw=jaw, **kwargs)

    @tool
    def get_tool_frame(self, arm: str) -> dict:
        """Measured gripper frame right now: jaw_axis (the fingers close along it),
        tool_axis (link6 -> fingertips), width, tip, quat. Use it to verify that an
        orientation you asked for was actually achieved.
        """
        return self._prims.tool_frame(arm)

    @tool
    def orientation(self, arm: str, tool: str = "down", jaw: str = "y") -> list:
        """A goto_pose quaternion from words: tool pointing "down"/"forward"(+y)/"back"/
        "left"(-x)/"right"(+x)/"up" (or a vector) with the jaw closing along "x"/"y"/"z"
        (or a vector). Horizontal-tool rolls are honoured only inside the arm's comfortable
        workspace (y <= ~-0.1 for the X5): check get_tool_frame after the move.
        """
        return self._prims.orientation(arm, tool, jaw)

    @tool
    def reach_map(self, arm: str, xs: list, ys: list, z: float, quaternion_wxyz=None) -> dict:
        """Probe a grid of fingertip positions (plan-only): returns a text map ("o" =
        reachable, rows y / columns x) and the reachable points. One probe costs ~0.3 s;
        keep grids under ~30 points.
        """
        return self._prims.reach_map(arm, xs, ys, z, quaternion_wxyz)

    @tool
    def probe_surface(
        self, arm: str, x: float, y: float, z_from: float = 0.95, z_min: float = 0.77
    ) -> dict:
        """Lower the CLOSED gripper tip at (x, y) until it touches something: returns
        {"contact", "z"} — the height of whatever is there (table ~0.77, object tops...).
        Cheap way to find a plate/lid surface the bounding box misrepresents.
        """
        return self._prims.probe_surface(arm, x, y, z_from=z_from, z_min=z_min)

    @tool
    def place_gently(self, arm: str, object_name: str) -> dict:
        """Lower the HELD object in 1.5 cm steps until it rests on whatever is below (it
        stops following the hand), then release and retreat 6 cm. Returns {"rested",
        "descent_m", "object_z"}. Use it instead of guessing the support height.
        """
        return self._prims.place_gently(arm, object_name)

    @tool
    def pick_flat(
        self,
        arm: str,
        object_name: str,
        jaw: str | list = "auto",
        tilt_deg: float = 20.0,
        along: float | str = "com",
        lift: float = 0.08,
        clearance: float = 0.002,
        hover: float = 0.10,
    ) -> dict:
        """Pick a FLAT object lying on the table (tools, tiles, bars) and lift it, with
        an honest hold check. jaw = world axis the fingers close along ("x"/"y", a
        horizontal [x, y, 0] vector, or "auto" = across the object's OWN shortest
        horizontal bounding-box side, whatever its yaw on the table); tilt_deg leans
        the tool axis about the jaw axis toward the arm (the X5 keeps a wrist
        orientation far from its base only with a 20-30 deg lean); along = pinch offset
        from the object's origin along cross(z, jaw) (returned as `along_axis`): "com"
        (default) pinches on the centre-of-mass line - the ONLY pinch that holds a flat
        tool level (measured: wrench 0.2 deg at the centre of mass vs 14-26 deg at the
        origin 2 cm away; a pendulum wants its mass under the pinch line) - or a signed
        metre offset. Parts with diverging or rounded sides (pliers handles, a round
        joint) roll in the pads at ANY offset: read get_object_profile first and prefer
        parallel flat sides. The tip goes to table + clearance so the pads overlap the
        whole thickness; hover is the approach height above the object. The hold is
        checked at a 2 cm test lift AND at the full lift (rise and turn < 15 deg).
        Returns {"held", "reason", "width", "rise_m", "turn_deg", "moved_m", "jaw_open",
        "quat_wxyz", "pinch_xy", "along_axis", "jaw_axis", "object_side_m", "notes"};
        held=False means the jaw was opened where the object is - read moved_m/turn_deg
        before retrying (a pivoted tool can be left on its edge, no longer flat) and
        jaw_open (a jammed close is reported, not hidden). With jaw="auto" the sign of
        along_axis follows the object's OWN edge (read it before passing a numeric
        along), and an object whose narrowest side exceeds the 8.7 cm jaw is refused
        before any motion; "com" on a prop without mass properties pinches at the origin
        and says so in notes.
        """
        return self._prims.pick_flat(
            arm,
            object_name,
            jaw=jaw,
            tilt_deg=tilt_deg,
            along=along,
            lift=lift,
            clearance=clearance,
            hover=hover,
        )

    @tool
    def carry_held(self, arm: str, object_name: str, target_xy: list, step: float = 0.03) -> dict:
        """Translate the HELD object so its origin reaches target_xy, in short steps at
        the current height. Use it for thin or off-centre-pinched parts (one 25 cm goto
        let a 1 cm wrench bar pivot 30 deg in the pads; ten 3 cm steps kept it within
        1 deg); a well-centred pinch (hammer handle, tape measure) survives one
        goto_pose and costs a tenth of the steps. Stops as soon as the object stops
        following the hand. Do NOT re-close the gripper mid-carry: a second
        close_gripper on a held object drops it. Returns {"arrived", "dropped",
        "reason": arrived | dropped | not held | unreachable | max_steps, "steps",
        "error_m", "turn_deg"}.
        """
        return self._prims.carry_held(arm, object_name, target_xy, step=step)

    @tool
    def align_held(
        self, arm: str, object_name: str, quaternion_wxyz: list, point: str | None = None
    ) -> dict:
        """Rotate the wrist about the fingertip so the HELD object's orientation
        matches the given world quaternion - computed from the object's ground-truth
        quaternion, not searched. Without `point` the ROOT frame is aligned; with
        point=<active functional point name> (e.g. "center", "handle", "center_flip")
        that point's frame is aligned - the frame the scorer's is_align_qpos compares
        (root ∘ the point's local rotation; identical to the root only for points with
        an identity local rotation, such as the toolbox slots' "center", not a mug's
        120-deg "handle" or the pliers' "center_flip"). The object's origin swings
        about the fingertip (lift it clear first). Returns {"residual_deg" (of the
        aligned frame), "orientation_dev_deg" (None when no motion was needed), "held",
        "flipped"}: a large residual means the wrist cannot take that orientation here
        (move closer to the arm or change the grasp tilt); flipped=True means goto_pose
        took the 180-deg wrist branch - do not iterate, re-plan the grasp. Raises
        RuntimeError when no IK exists at this position, KeyError for an unknown point.
        """
        return self._prims.align_held(arm, object_name, quaternion_wxyz, point=point)

    @tool
    def align_held_current(
        self,
        arm: str,
        object_name: str,
        quaternion_wxyz: list,
        tolerance_deg: float = 3.0,
        workspace_center: list | None = None,
        point: str | None = None,
    ) -> dict:
        """Align a HELD object through bounded current-posture wrist steps.

        Every plan is seeded by current joints, uses the solver's true FK wrist, refuses
        wound/configuration-jump candidates before motion, executes exactly the probed
        joint candidate once, and verifies the fresh object-hand SE(3) anchor. The point
        argument uses legacy root-frame composition. workspace_center only produces a
        staging suggestion; this tool never executes it.

        The result has exactly ten sanitized fields and never exposes joints, solver
        history, raw actions, scorer state, or hidden simulator state.
        """
        raw = self._prims.align_held_current(
            arm,
            object_name,
            quaternion_wxyz,
            tolerance_deg=tolerance_deg,
            workspace_center=workspace_center,
            point=point,
        )
        if not isinstance(raw, dict):
            raw = {
                "status": "execution_failed",
                "success": False,
                "held": False,
                "reason": "held-current primitive returned a non-mapping result",
                "steps": 0,
                "residual_deg": None,
                "anchor_position_error_m": None,
                "anchor_orientation_error_deg": None,
                "used_returned_wrist": False,
                "staging": None,
            }
        stage = raw.get("staging")
        staging = None
        if isinstance(stage, dict):
            staging = {
                "hand_pose": stage.get("hand_pose"),
                "predicted_object_pose": stage.get("predicted_object_pose"),
                "translation_m": stage.get("translation_m"),
                "reason": stage.get("reason"),
            }
        return {
            "status": raw.get("status", "execution_failed"),
            "success": bool(raw.get("success", False)),
            "held": bool(raw.get("held", False)),
            "reason": str(raw.get("reason", "missing typed reason")),
            "steps": int(raw.get("steps", 0)),
            "residual_deg": raw.get("residual_deg"),
            "anchor_position_error_m": raw.get("anchor_position_error_m"),
            "anchor_orientation_error_deg": raw.get("anchor_orientation_error_deg"),
            "used_returned_wrist": bool(raw.get("used_returned_wrist", False)),
            "staging": staging,
        }

    @tool
    def seat_object(
        self,
        arm: str,
        object_name: str,
        target_xyz: list,
        clear_z: float | None = None,
        container: str | None = None,
    ) -> dict:
        """Lower the HELD object into a slot/recess so its origin lands on target_xyz
        (e.g. a functional point): servo over the target, fast-descend to just above the
        rim, re-servo, descend in 5 mm steps until the object stops following the hand
        (two consecutive steps) or reaches the target height, then release in two stages
        and retreat. The rim height is clear_z (world z of the recess top) or, with
        container=<name>, the top of that object's world bounding box; with neither the
        slow descent starts from the current height (about 50 sim steps per 5 mm). If the
        object never rests it stays HELD ("held": True) so you can retry. Returns
        {"seated" (rested within 1 cm of target z), "rested", "held", "servo_error_m",
        "descent_refused", "error_m", "dz_m", "turn_deg"} - error/dz measured after the
        release; a thick object can rest 1-2 cm above its seat while held and drop in on
        release.
        """
        return self._prims.seat_object(
            arm, object_name, target_xyz, clear_z=clear_z, container=container
        )

    @tool
    def push_joint(
        self,
        object_name: str,
        joint: str,
        target_ratio: float,
        arm: str,
        link: str,
        contact_local: list,
    ) -> dict:
        """EXPERIMENTAL: push a hinged part (lid/door) toward a joint ratio with the closed
        gripper, along the hinge arc. contact_local = a point on the moving link, in that
        link's frame (see get_functional_points / get_joint_states for names). Returns
        {"pushed", "ratio", "strokes"} or an "error" with the unreachable point.
        """
        return self._prims.push_joint(object_name, joint, target_ratio, arm, link, contact_local)

    @tool
    def get_step_budget(self) -> dict:
        """The episode's simulation-step budget: {"used", "max"}. A typical
        single pick-and-place costs roughly 1500-2500 steps; reserve ~1500
        for the endgame (release + both arms home).
        """
        return self._env.get_step_budget()
