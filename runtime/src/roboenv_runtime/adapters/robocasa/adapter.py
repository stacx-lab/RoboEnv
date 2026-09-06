"""RoboCasa365 (robosuite / MuJoCo; Franka Panda on an Omron mobile base) adapter.

The first embodiment with ``Capability.BASE_MOTION``: tools include
``base_move`` and the situation text tells the agent to drive within reach
before manipulating.  Task ids are ``robocasa.<KitchenEnvName>``
(``roboenv tasks sync robocasa`` writes one file per registered environment).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.tools import tool
from roboenv_runtime.core.types import Budget, Capability, ScoreReport
from roboenv_runtime.embodiments import load_embodiment

__all__ = ["RoboCasaAdapter"]


@register_adapter
class RoboCasaAdapter(AdapterBase):
    """RoboCasa task ``task_name`` on the ``panda_omron`` embodiment."""

    name: ClassVar[str] = "robocasa"
    embodiment: ClassVar[str] = "panda_omron"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {Capability.PRIVILEGED, Capability.VIDEO, Capability.BASE_MOTION}
    )

    def __init__(self, task_name: str = "PickPlaceCounterToCabinet", **env_kwargs: Any) -> None:
        from roboenv_runtime.adapters.robocasa.backend import (
            RoboCasaEnv,  # heavy: builds the kitchen
        )

        self.task_name = task_name
        self._env = RoboCasaEnv(task_name=task_name, **env_kwargs)

    # ------------------------------------------------------------ context

    @property
    def profile(self):
        """The ``panda_omron`` embodiment profile."""
        return load_embodiment(self.embodiment)

    def situation(self) -> str:
        """Profile situation plus the episode's language instruction."""
        base = self.profile.render_situation().rstrip()
        goal = self.goal()
        return base + (f"\nBenchmark instruction: {goal}" if goal else "") + "\n"

    def facts(self) -> str:
        """Profile facts plus this episode's task fixtures and kitchen inventory."""
        refs = ", ".join(f"{k} = {v}" for k, v in self._env.task_fixtures().items())
        fixtures = ", ".join(self._env.list_fixtures()[:40])
        return (
            self.profile.render_facts()
            + (f"\nFixtures this task refers to: {refs}\n" if refs else "")
            + f"\nFixtures in this kitchen: {fixtures}\n"
        )

    def goal(self) -> str:
        """The task's language instruction."""
        return self._env.goal_text()

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int) -> None:
        """Start a fresh episode with video capture on.

        RoboCasa's placement samplers draw from Python's global ``random`` (and some
        utilities from NumPy's global state), which the benchmark's own ``reset(seed)``
        does not seed — so the same seed produced different layouts in different
        processes (ButterOnPan's butter spawned on either side of the stove).  Seeding
        both here makes "seed 0" mean one scene, whichever process builds it.
        """
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        self._env.reset(seed=seed)
        self._env.enable_video_capture(True, clear=True)

    def close(self) -> None:
        """Close the simulation."""
        self._env.close()

    def score(self) -> ScoreReport:
        """The benchmark's sparse success."""
        completed = self._env.task_completed()
        return ScoreReport(reward=float(completed), raw_score=float(completed), completed=completed)

    def budget(self) -> Budget:
        """Environment steps vs the task horizon."""
        raw = self._env.get_step_budget()
        return Budget(used=int(raw["used"]), max=int(raw["max"]), unit="action_frames")

    def frames(self) -> list[Any]:
        """Captured camera frames."""
        return self._env.get_video_frames()

    # ------------------------------------------------------------ tools

    @tool
    def list_objects(self) -> list[str]:
        """Names of the movable objects the task mentions (use get_object_pose on them)."""
        return self._env.list_objects()

    @tool
    def list_fixtures(self) -> list[str]:
        """Names of the kitchen fixtures (counters, cabinets, sink, stove, microwave...).
        Fixtures do not move; their poses come from get_object_pose too.
        """
        return self._env.list_fixtures()

    @tool
    def get_object_pose(self, name: str):
        """Ground-truth world pose of an object or fixture: (position xyz, quaternion wxyz)."""
        pos, quat = self._env.get_object_pose(name)
        return pos.tolist(), quat.tolist()

    @tool
    def get_object_extent(self, name: str) -> dict:
        """Object size along its own axes plus those axes in world coordinates, and its
        bottom/top z: pick the horizontal axis under 0.08 m to close the fingers across.
        """
        return self._env.object_extent(name)

    @tool
    def top_down_grasp_quat(self, opening_yaw: float, nearest: bool = True) -> list[float]:
        """Orientation (wxyz) for goto_pose: tool pointing straight down, fingers opening along
        world yaw `opening_yaw` (radians). The finger-opening axis is the grip site's x; the
        gripper is symmetric, so with `nearest` (default) the axis is flipped to yaw+pi when
        that side is closer to the wrist's CURRENT finger axis. That is a heuristic against
        the joint-7 stall (0.81 rad residual measured for the far side from the reset pose),
        not a guarantee: compute it right before the goto_pose that uses it (a base turn or
        an earlier goto changes the answer) and read the goto_pose orientation residual.
        """
        return self._env.top_down_grasp_quat(float(opening_yaw), nearest=bool(nearest)).tolist()

    @tool
    def get_fixture_joints(self, name: str) -> dict:
        """Every joint of a fixture with what you need to actuate it: type (hinge about a
        vertical axis = cabinet/microwave/fridge door, about a horizontal axis = oven/
        dishwasher/toaster-oven drop-down door; slide = drawer), world axis and anchor,
        range, qpos, `fraction` (the benchmark's own normalised value, 0 closed .. 1 open),
        `open` / `closed` (that fraction against the scorer's thresholds: doors > 0.9 /
        < 0.005, drawers > 0.95 / < 0.05; the fridge drawer task wants > 0.7),
        `open_direction` (where the handle travels when opening = the outer face normal;
        None when no handle was found) and `handle` (None, or bar centre/axis/half_length/
        thickness; kind "handle" for a real handle geom, "lip" for the protruding collision
        geom of a handle-less SLIDING part - a drawer front, a rack edge, or a button cap
        when half_length < 1 cm; hinged knobs/lids/heads have no handle entry: use the axis
        and anchor). `door_joints` are the joints the fixture calls its door(s); `is_open` /
        `is_closed` are fixture.is_open()/is_closed() with default arguments (for a fridge:
        its fridge-door compartment only) - prefer the per-joint `open`/`closed`.
        """
        return self._env.fixture_joints(name)

    @tool
    def find_standing_pose(self, target: list, approach_dir: list | None = None) -> list:
        """Candidate base poses [x, y, yaw] for reaching the world point `target`, best
        first: standing 0.5-0.8 m from it along directions fanned around `approach_dir`
        (the way from the target toward free floor, e.g. a door's open_direction; default:
        toward where the base is now), facing it, with the base footprint (inflated 3 cm)
        clear of every fixture, wall, door and drawer collision box at base height and the
        target within 0.86 m of the shoulder. It does NOT check the drive path, line of
        sight or an IK solution: after drive_to(*pose) read its `reached` flag, and read the
        goto_pose residual. Returns [] when no candidate clears. Each entry also reports
        standoff, shoulder_dist and drive_m.
        """
        return self._env.find_standing_pose(target, approach_dir)

    @tool
    def grasp_reach(
        self,
        object_name: str,
        base_pose: list | None = None,
        torso: float | None = None,
        below_m: float = 0.02,
    ) -> dict:
        """PREDICTED site_above_bottom_m for a top-down pinch on an object from a standing
        pose the robot has NOT driven to -- kinematics only, ZERO frames. pick_object
        reports that number only after spending the descent, and it is what separates a
        grasp from a miss (measured over 22 shape-D tasks: <= 0.039 grasped 12 of 12,
        >= 0.07 failed 9 of 9). IK is solved for the grip site at bottom_z - below_m over
        the object from a hypothetical base_pose [x, y, yaw] and torso height, walking
        upward until it converges. base_pose/torso None mean "as we stand now". Returns
        {"site_above_bottom_m", "reachable" (< 0.04), "hover_ok", "width_m", "fits_jaw"}.
        """
        return self._env.grasp_reach(
            object_name, base_pose=base_pose, torso=torso, below_m=below_m
        )

    @tool
    def find_pick_pose(self, object_name: str, max_probe: int = 14) -> dict:
        """Search standing poses for one the arm can actually pick `object_name` from, at a
        cost of ZERO frames: a fan of footprint-clear poses 0.45-0.75 m around the object,
        each crossed with torso heights 0.0/0.17/0.34 (the torso is a free variable and on a
        shelf or in a cabinet it is worth more than any base position), priced with
        grasp_reach and ranked by the predicted site_above_bottom_m, then by driving
        distance. Returns {"best" (None when nothing reaches), "here" (the same prediction
        for standing still), "candidates", "probed", "pool"}.
        """
        return self._env.find_pick_pose(object_name, max_probe=max_probe)

    @tool
    def stand_to_pick(self, object_name: str, max_drive_m: float = 2.6) -> dict:
        """Stand where find_pick_pose predicts the pinch will reach, and lift the torso to
        match; costs nothing when standing still already predicts a grasp, and refuses to
        drive when no candidate is materially better (the limit is then the arm's vertical
        envelope, not the base position). Call it BEFORE pick_object. Returns {"moved",
        "here_site_above_bottom_m", "predicted_site_above_bottom_m", "reachable", "drive",
        "torso", "steps", "reason"}.
        """
        return self._env.stand_to_pick(object_name, max_drive_m=max_drive_m)

    @tool
    def base_pose_clear(self, x: float, y: float, yaw: float) -> bool:
        """True when the base footprint at world (x, y, yaw) overlaps no fixture, wall, open
        door or drawer (collision geometry as it stands now).
        """
        return self._env.base_pose_clear(x, y, yaw)

    @tool
    def get_fixture_regions(self, name: str) -> dict:
        """Interior placement regions of a fixture as world boxes (tray, shelf, rack, ...):
        the volumes RoboCasa checks objects in. Aim the object's centre at region "center".
        """
        return self._env.fixture_regions(name)

    @tool
    def horizontal_grasp_quat(
        self, point_dir: list, close_dir: list = (0.0, 0.0, 1.0), nearest: bool = True
    ) -> list:
        """Orientation (wxyz) for goto_pose with the tool axis along the horizontal `point_dir`
        (e.g. [-1, 0, 0] to face a door in -x) and the fingers closing along `close_dir`
        (default vertical: pinch a vertical handle bar or reach into a cavity sideways).
        `nearest` (default) may flip close_dir to -close_dir (the same grasp) when that side
        is closer to the wrist's current finger axis - a heuristic against the joint-7
        stall, not a guarantee (see top_down_grasp_quat). Raises when close_dir is parallel
        to point_dir.
        """
        return self._env.horizontal_grasp_quat(point_dir, close_dir, nearest=bool(nearest)).tolist()

    @tool
    def open_door(self, fixture: str, target_fraction: float = 0.97) -> dict:
        """Open every door joint of `fixture` (cabinet/microwave/fridge doors, drop-down oven/
        dishwasher/toaster-oven doors, drawers) to `target_fraction` of the benchmark's own
        open fraction: finds the handle on the moving part, drives the base to a clear spot
        the arm reaches from, pinches the handle with the finger pads seated BEHIND the bar
        (a hook: a plain friction pinch slides off a door that rubs its frame) and moves it
        along the joint's arc/axis in strokes clamped to the travel left, re-grasping,
        re-positioning or towing with the base when the arm stalls; drop-down doors are
        released to fall open under gravity. Verified in sim (2026-08-27) for drawers,
        vertical-hinge doors (microwave; each wall-cabinet door alone) and drop-down doors
        (dishwasher, oven - the oven needs base repositioning and runs past the advisory
        horizon). Not yet solved: a two-door cabinet (the second door's base tow knocks the
        first shut - the report now measures every joint AFTER the last pass and says so
        with a `reason`, instead of reporting the stale per-door pass), the 54 kg fridge
        door (the tow slips at ~0.5), the toaster oven (its bar needs a wrist-forward
        posture the servo does not find). Returns {"done", "fraction" (fresh), "strokes",
        "regrasps", "mode", "budget_exhausted", "joints": per-joint reports with
        "fraction_final", "seat" (how deep the pads went behind the bar; "seated" False
        means a friction pinch), "rebounds", "history", and "redone" when a door pair
        needed another pass}; mode "no-handle" with a `reason` means nothing was
        attempted, and a `reason` with strokes 0 means every approach was refused. Use
        0.95 for doors (the scorer wants > 0.9; a greedier 0.97 costs extra tows) and the
        default 0.97 for drawers (> 0.95). The handle/lip travels with the part: anything
        you aim yourself must re-read get_fixture_joints after every stroke.
        """
        return self._env.actuate_door(fixture, target_fraction)

    @tool
    def close_door(self, fixture: str, target_fraction: float = 0.0) -> dict:
        """Close every door joint of `fixture` to `target_fraction` (0.0 = against the stop,
        what the benchmark's is_closed wants): drawers AND vertical-hinge doors are pushed
        shut at the handle bar with the closed fingertips, along the slide axis or the
        hinge arc with the wrist turning (verified in sim 2026-08-27: drawer; one
        wall-cabinet door in 4 strokes - the other's bar faces away from every reachable
        spot); when the arm runs out of reach along the arc the door is finished by a
        PANEL push at its free edge (the certified CloseFridge seed 0: handle strokes to
        0.55, then mode "push-panel" 6 strokes to 0.0; seed 1's fridge is unreachable
        from every standing spot and fails). `history` in the per-joint report lists
        both paths; `mode` names the one that moved the part last. Drop-down
        doors are pinched at their free edge and swung up - measured NOT closable by the
        arm for the dishwasher (40 Nm at the hinge, 22 kg door) and oven (21 Nm): the OSC
        pulls a few tens of N, the base would be needed; the toaster oven's edge (2.7 Nm)
        is closable but sits 9 cm above the counter where the servo finds no pinch pose.
        Same report as open_door.
        """
        return self._env.actuate_door(fixture, target_fraction)

    @tool
    def actuate_joint(self, fixture: str, joint: str, target_fraction: float) -> dict:
        """open_door/close_door for one named joint of a fixture that is not (only) its door:
        e.g. a fridge's drawer joint or a freezer door listed by get_fixture_joints.
        """
        return self._env.actuate_door(fixture, target_fraction, joints=[joint])

    @tool
    def get_task_fixtures(self) -> dict:
        """Role -> fixture name for the fixtures this task is about: the benchmark's
        registered references first (counter, cab, fxtr ...), then every fixture the task
        class keeps as a plain attribute of the environment under its attribute name
        (blender, toaster_oven, cab2 ...) - a task can register some and keep others -
        and auxiliary parts as <attribute>_<part> (blender_lid) or <attribute>_auxiliary.
        Empty only when the task has neither.
        """
        return self._env.task_fixtures()

    @tool
    def get_fixture_parts(self, name: str) -> dict:
        """Every collision geom and joint of a fixture from the live model: {"geoms": [{name,
        type, center, half_size, axes (world rows), top_z, bottom_z, body}], "joints": {name:
        {type, axis, anchor, range, qpos, fraction, body}}} - buttons, knobs, levers, lids and
        bars included, whether or not get_fixture_joints lists them (a stove's knob joints, a
        microwave's *start_button geom, a blender lid's *handle_main knob). Use it to aim a
        press ALONG the button joint's axis or the face normal from `axes` (top_z only for
        an upward-facing cap: microwave buttons are on a vertical panel, the blender's
        button axis is diagonal), a top-down pinch (a knob's half_size across the fingers)
        or a turn (a knob joint's axis/anchor). `fraction` is the scorer's own (the same
        number get_fixture_joints reports) and can be slightly NEGATIVE at rest - MuJoCo
        joint limits are soft and a button settles ~0.6 mm under its lower limit (the
        blender reads -0.40); hinge/slide joints only (a free-jointed lid lists none: use
        get_object_pose for its pose); a visual-only fixture lists no geoms. Geoms of an
        auxiliary fixture that shares the name prefix (a blender's lid) are listed under
        that fixture's own name.
        """
        return self._env.fixture_parts(name)

    @tool
    def get_task_fixture(self, hint: str | None = None) -> dict:
        """The fixture the benchmark spawns the robot in front of (the task's own
        `init_robot_base_ref`). For the atomic door/drawer/appliance tasks (Open*/Close*/
        TurnOn*/TurnOff*/Press*/Adjust*) that IS the fixture the scorer reads. It is only ONE
        side of a PickPlace task — the SOURCE appliance for *ToCounter tasks (microwave, stove,
        sink, fridge, toaster, drawer, cabinet), the target for CounterTo* tasks — and for
        NavigateKitchen it is the START fixture, not the scored target (measured 2026-08-27):
        read the other roles from get_task_fixtures(). Else the registered "fxtr" reference,
        else the only / first registered reference, else the first Fixture attribute.
        Returns {"name", "role", "class", "source"}; use ["name"] with get_fixture_joints /
        open_door / press_part. Raises only when the environment has no fixture at all.
        `hint` resolves another fixture the way programs used to by hand (a task_fixtures
        role "blender", a class "StandMixer"/"stand_mixer", or a name fragment "sink":
        role, then class, then name substring, auxiliary parts last; KeyError lists what
        exists when nothing matches) - one call instead of the .get()/next(iter)/
        list_fixtures() idiom.
        """
        return self._env.task_fixture(hint)

    @tool
    def get_joint_state(self, fixture: str, joint: str | None = None) -> dict:
        """A fresh read of one joint the way the scorer sees it (default: the fixture's
        first door joint): {"joint", "type", "fraction" (the benchmark's own normalised
        value), "qpos", "open", "closed" (THIS joint against "thresholds": doors 0.9 / 0.005,
        Drawer fixtures 0.95 / 0.05, a fridge's drawer joints 0.70 / 0.05, other non-door
        slide joints such as racks 0.95 / 0.05; None for a knob or a lever, which the door
        thresholds do not describe - steer those on qpos/range), "is_door", "fixture_open",
        "fixture_closed" (the scorer's verdict for the fixture: EVERY default door joint at
        once - a two-door cabinet with one door open is neither), "handle_center",
        "open_direction" (fresh - a handle travels with its part), "axis", "anchor", "range",
        "door_fractions" (the door joints the fixture's default is_open/is_closed check)}.
        Cheaper than get_fixture_joints; read it before and after every actuation instead of
        trusting a primitive's claim.
        """
        return self._env.joint_state(fixture, joint)

    @tool
    def set_torso_height(self, height: float) -> dict:
        """Raise or lower the arm's mounting torso to `height` metres above its lowest
        position (0 .. 0.34; RoboCasa resets it to 0). A stiff position servo: the full
        lift takes ~30 steps and moves the shoulder from z 1.03 to 1.37, so wall-cabinet
        handles (z 1.57) and microwave doors on a counter come within the arm's comfortable
        envelope, and tuck_arm / drive_to_fixture tuck relative to the lifted torso. The arm
        rides with the torso (release a held handle first); a request outside 0..0.34 is
        clamped and reported ("clamped", reached False); a torso that stops moving with the
        error still open reports reason "stalled". Returns {"height", "requested", "clamped",
        "shoulder_z", "eef_z", "steps", "reached", "reason", "budget_exhausted"}.
        """
        return self._env.set_torso_height(float(height))

    @tool
    def tuck_arm(self) -> dict:
        """Bring the arm back to its reset pose relative to the base (and the current torso
        height) before driving: backs off 6 cm along the tool axis first, keeps the gripper
        as it is (a carried object stays carried). Returns the goto_pose report (+ "reason").
        """
        return self._env.tuck_arm()

    @tool
    def stand_at(self, target: list, approach_dir: list | None = None) -> dict:
        """Tuck the arm and drive the base to a spot the arm can work the world point
        `target` from: the find_standing_pose candidates (0.5-0.8 m away along directions
        fanned around `approach_dir`, e.g. a handle's open_direction; footprint clear, target
        within reach, judged from the LIVE shoulder height so a torso lift counts) are driven
        to in order, arm held, until drive_to reports the pose reached; without approach_dir
        (or with a vertical one) the candidates fan around the direction toward the base.
        `target` must be a 3-D world point. Worst case ~1060 frames. Returns {"reached",
        "pose", "candidate", "tried", "drive", "tuck", "shoulder_dist", "steps",
        "budget_exhausted"} or reached False with a "reason" (no clear spot / every drive
        obstructed or out of steps). Read the goto_pose residual afterwards: reach is not
        guaranteed by the geometry check.
        """
        return self._env.stand_at(target, approach_dir)

    @tool
    def press_part(
        self,
        fixture: str,
        part: str,
        increase: bool = True,
        depth_m: float = 0.03,
        retreat_m: float = 0.2,
    ) -> dict:
        """Press a button, lever or cap with the closed fist and come away: `part` is a
        SLIDE joint of the fixture (pressed along its own axis in the sense that raises the
        scorer's fraction - a kettle lid button, a toaster lever, a blender power button; set
        increase=False to push a drawer's lip shut) or a collision GEOM name from
        get_fixture_parts (a microwave's *start_button, pressed along its inward normal).
        A sideways press points the fingertips at the part (oriented: the finger's side face
        pushed the panel 1.9 cm short of a flush microwave button position-only), a downward
        press keeps the wrist free; from 6 cm before the part to `depth_m` past it in 1 cm
        steps, stopping when the joint fraction rose by 0.15 or the gripper touched the geom,
        then a retreat of `retreat_m` (verified, default 0.2 m: the microwave scorer wants the
        grip site > 0.15 m from the button; a blender toggles on the RELEASE of the contact).
        Drives to a standing spot when the ORIENTED pre-approach missed and the shoulder is
        > 0.6 m away, then retries position-only. Certified on flush button geoms of a vertical
        panel (microwave start/stop); the joint path is verified on the kettle lid button only
        - a long-travel lever (toaster) and a diagonal button (blender) are not solved by it
        (2026-08-27). `part` must belong to `fixture`. Returns {"kind", "center", "press_dir",
        "normal_source", "oriented", "fraction_before", "fraction_after" (before the retreat),
        "fraction_peak", "fraction_final" (after the retreat), "disturbed_by_approach",
        "touched", "depth_reached", "retreat_m", "reached_pre", "reached_contact", "ok",
        "reason", "steps", "budget_exhausted"} - a joint's own fraction (peak, then final), or
        a geom's contact, is the verdict, not the goto residual.
        """
        n = max(1, round(float(depth_m) / 0.01))
        depths = tuple(0.01 * k for k in range(n + 1))
        return self._env.press_part(
            fixture, part, increase=bool(increase), depths=depths, retreat_m=float(retreat_m)
        )

    @tool
    def close_door_with_base(self, fixture: str, joint: str | None = None) -> dict:
        """Close a heavy drop-down door (dishwasher: 22 kg, 40 Nm at the hinge - beyond the
        arm's OSC) by driving the BASE into its panel: lines the base up with the panel's
        centre line heading along the closing direction, drives forward in 10 cm pushes with
        the arm tucked and held while reading the scorer's fraction, stops when the door reads
        closed or the base stops moving (or slides past the panel), then backs off 0.25 m and
        re-reads. Refuses BEFORE moving, with a "reason" and 0 frames: a vertical-hinge door
        (use push_door_shut), a panel whose bottom stands above the pedestal top 0.38 m (an
        oven door open at 66 deg, hinge z 0.40), a door open past ~72 deg, an already-closed
        door. Measured 2026-08-27: CloseDishwasher 1.00 -> 0.00 in 5 pushes, 165 frames.
        Returns {"done", "fraction", "pushes", "history", "base_travel_m", "panel_bottom_z",
        "steps", "budget_exhausted"} (+ "reason").
        """
        return self._env.close_door_with_base(fixture, joint)

    @tool
    def push_door_shut(
        self,
        fixture: str,
        joint: str | None = None,
        torso: float | None = None,
        stroke_rad: float = 0.4,
    ) -> dict:
        """Close a vertical-hinge door (cabinet, microwave, fridge) by pushing its handle bar
        along the hinge arc with the closed fist, POSITION-ONLY (the wrist is left free, so
        the arm does not stall on orientation the way close_door's push-handle path does):
        every stroke re-reads the bar, aims 2.5 cm outside it along the panel's current outer
        normal and rotates that point about the hinge by up to `stroke_rad` (0.4; clamped
        to the travel left); when the next target is beyond the arm or a stroke moved nothing the
        hand retreats and the base is re-stood for the bar's current position. `torso`
        (metres, default auto: lift so the bar sits ~0.25 m above the shoulder, max 0.34)
        raises the arm for wall cabinets. For a door pair call it once per joint (the joints
        from get_fixture_joints door_joints). Measured 2026-08-27: microwave 0.99 -> 0.0 in
        4 strokes, wall-cabinet door beside the shoulder 0.97 -> 0.0025 in 5 strokes, the
        fridge door 0.908 -> 0.0 in 3 strokes / 677 frames. The lifted torso stays lifted;
        strokes stop at the horizon. Returns {"done", "fraction", "strokes", "repositions",
        "torso", "contact" ("handle"|"panel"), "history", "steps", "budget_exhausted"} on
        every path (+ "reason" on a refusal or when it stopped short).
        """
        return self._env.push_door_shut(fixture, joint, torso=torso, stroke_rad=float(stroke_rad))

    @tool
    def get_instruction(self) -> str:
        """The episode's language instruction - the ONLY place the benchmark names a direction
        (TurnSinkSpout "left"/"right"), a knob ("front left burner"), a rack ("top"/"bottom")
        or a pull/push sense; the scene itself is symmetric. Read it with get_episode_meta
        (task_refs / should_pull / rack_level carry the same choice as data).
        """
        return self._env.instruction()

    @tool
    def get_episode_meta(self) -> dict:
        """RoboCasa's episode metadata without the scene dumps: lang, layout_id, style_id,
        fixture_refs (role -> fixture name), task_refs (a stove task's `knob` and
        `cookware_burner`, a sink task's `behavior`), refs, should_pull, rack_level,
        initial_temp ... whatever the task class stored. Costs no steps.
        """
        return self._env.episode_meta_public()

    @tool
    def get_goal(self) -> dict:
        """What this episode is scored on, decoded from the task class: {"instruction",
        "task", "behavior", "satisfied" (the benchmark's _check_success, LIVE),
        "predicates": [{kind, fixture, joints/joint/part, test, thresholds/target, value}],
        "fixtures", "objects", "meta"}. Predicate kinds: joint_fraction (door/drawer
        fractions with the scorer's own thresholds), knob_qpos (a stove burner: raw qpos,
        on = 0.35 <= |q mod 2pi| <= 2pi - 0.35; LowerHeat 0.35 <= |q| < 1.396),
        fixture_state (get_state() keys: turned_on, lid, head, time, temperature ...),
        gripper_far (a button/lid press wants the grip site > 0.15 m away afterwards), or
        unknown (undecoded: the task class and its scoring attributes are listed - read
        `satisfied`). Costs no steps; re-read it after acting.
        """
        return self._env.goal()

    @tool
    def get_placement_goal(self) -> dict:
        """Which object has to move and what it has to end up in, READ OUT OF THE SCORER —
        the episode's own _check_success is run with the benchmark's placement predicates
        wrapped, so both referents are the ones it really tests, not a guess from the
        object's name or the sentence. Returns {"placements": [{kind, object, destination,
        destination_is_fixture, satisfied, support}], "pending" (the placements that are
        FALSE now: the work), "move" (the first pending one), "single" (exactly one pending:
        the referent is decided), "gripper_far" (the objects the hand must end 0.25 m from),
        "constraints", "satisfied", "source"}. `support` is the fixture the destination
        object is resting on — where the base has to stand. A destination that is already
        satisfied at reset is staging, not work (PrepForTenderizing scores the meat on its
        own board AND the rolling pin onto it; only the pin moves). Costs no steps.
        """
        return self._env.placement_goal()

    @tool
    def get_supporting_fixture(self, object_name: str) -> str | None:
        """The fixture this object is resting on right now, from MuJoCo's own contacts —
        which is the fixture to drive to when you have to reach it. None when it touches no
        fixture (it is in the hand, or on another object). Costs no steps.
        """
        return self._env.supporting_fixture(object_name)

    @tool
    def plan_base_route(self, x: float, y: float, yaw: float | None = None) -> dict:
        """Waypoints [[x, y, yaw], ...] from the base's current pose to world (x, y[, yaw])
        found by a uniform-cost search over base_pose_clear on a 25 cm grid (8-connected,
        greedily shortcut): drive_to is a straight line and a kitchen island or counter run
        blocks most cross-kitchen lines. Every EDGE is checked, not every cell, so a route
        it returns is drivable. It plans at the current heading first and only then with
        the heading in the state (the base is holonomic). A goal whose own footprint is not
        clear - which is most of RoboCasa's own get_base_placement poses, they park the base
        against the counter - is moved to the nearest pose within 0.35 m that is, reported
        as "goal" and "goal_moved_m". No simulation steps. Returns {"ok", "waypoints",
        "length_m", "turns", "goal", "goal_moved_m", "cells_expanded", "reason"}; ok False
        when nothing within 0.35 m of the goal is clear or nothing connects. drive_route
        drives it.
        """
        return self._env.plan_base_route(x, y, yaw)

    @tool
    def drive_route(self, x: float, y: float, yaw: float | None = None) -> dict:
        """plan_base_route + drive_to per waypoint with the arm held at its base-relative
        pose, after a tuck_arm (an extended arm carried along the route rams the furniture
        the base drives past). A leg whose heading differs from the base's is turned into on
        the spot first (drive_to turns WHILE it translates, which sweeps the footprint
        through orientations the plan never checked). Stops at the first leg not reached.
        Measured 2026-08-28: 4.7 m across a kitchen, four legs, 200 frames. Returns
        {"reached", "legs", "plan", "pose", "tuck", "steps", "budget_exhausted"}.
        """
        return self._env.drive_route(x, y, yaw)

    @tool
    def solve_arm_ik(self, position: list, quaternion_wxyz: list | None = None) -> dict:
        """Joint angles (joint1..7) that put the grip site at a world pose from where the
        base stands now - kinematics only, no steps: damped-least-squares IK from several
        seeds inside the joint limits, both finger sides (the gripper is symmetric). Returns
        {"ok", "q", "position_error_m", "orientation_error_rad", "limit_margin_rad",
        "joint_distance_rad", "finger_flipped", "candidates"}. ok False = outside the arm's
        kinematic envelope from here (move the base); ok True does NOT promise the OSC servo
        gets there in a straight line - use goto_pose_posture for that.
        """
        return self._env.solve_arm_ik(position, quaternion_wxyz)

    @tool
    def goto_pose_posture(
        self,
        position: list,
        quaternion_wxyz: list | None = None,
        waypoints: int = 16,
        tolerance_m: float = 0.01,
        tolerance_rad: float = 0.08,
    ) -> dict:
        """Reach a grip-site pose through a JOINT-SPACE path: the IK posture is solved
        (solve_arm_ik), the OSC is servoed through `waypoints` poses interpolated in joint
        space (forward kinematics of each), and the requested pose is finished. This is the
        posture option for wrists the plain goto_pose never finds: a tool-horizontal-forward
        pinch in front of a counter (stove knobs, the toaster-oven bar), side-on fingers at a
        low switch - the straight Cartesian servo ends in an elbow/joint-limit minimum with
        0.5-0.9 rad of orientation error while the posture exists. The controller's
        null-space spring is pointed at the IK posture until tuck_arm. Returns {"reached",
        "error_m", "error_rad", "ik", "joint_residual_rad", "waypoints_done", "steps",
        "budget_exhausted"} (+ "reason": "no IK solution ..." costs 0 steps). ~waypoints x 30
        steps.
        """
        return self._env.goto_pose_posture(
            position,
            quaternion_wxyz,
            waypoints=int(waypoints),
            tolerance_m=float(tolerance_m),
            tolerance_rad=float(tolerance_rad),
        )

    @tool
    def turn_knob(
        self,
        fixture: str,
        joint: str,
        target_rad: float | None = None,
        delta_rad: float | None = None,
        stroke_rad: float = 0.4,
    ) -> dict:
        """Turn a hinged knob (stove burner, oven / toaster-oven dial: a `knob_*_joint` from
        get_fixture_parts) to `target_rad` of RAW joint position (or by `delta_rad`): pinch
        the tab standing proud of the dial (tool along the knob axis into the dial, fingers
        across the tab's thinnest side, pads on the proud part, fingertips 3 mm off the dial
        face) through goto_pose_posture, then roll the wrist about the axis in strokes of
        `stroke_rad` clamped to the travel left, re-reading the joint after each; a stroke the
        knob does not follow, or a wrist at its roll limit, releases and re-grips (3 times).
        Knobs are scored on raw qpos (get_goal: a burner is on for 0.35 <= |q mod 2pi| <=
        2pi - 0.35 - ask for 0.6 to turn on, 0.0 to turn off, ~0.9 for low heat). Returns
        {"done", "qpos", "qpos_before", "target_rad", "strokes", "regrasps", "opening_m",
        "tab", "history", "steps", "budget_exhausted"} (+ "reason").
        """
        return self._env.turn_knob(
            fixture, joint, target_rad, delta_rad=delta_rad, stroke_rad=float(stroke_rad)
        )

    @tool
    def get_gripper_pose(self) -> list[float]:
        """Grip-site pose [x, y, z, qw, qx, qy, qz] in the world frame; the site's z axis
        points out of the fingers (down when the hand faces a counter).
        """
        return self._env.get_eef_pose().tolist()

    @tool
    def get_base_pose(self) -> list[float]:
        """Mobile-base pose [x, y, yaw] in the world frame."""
        return self._env.get_base_pose().tolist()

    @tool
    def get_gripper_opening(self) -> float:
        """Distance between the finger pads in metres (0 = closed)."""
        return self._env.get_gripper_opening()

    @tool
    def get_step_budget(self) -> dict:
        """Environment steps used vs the horizon: {"used", "max"}."""
        return self._env.get_step_budget()

    @tool
    def goto_pose(self, position, quaternion_wxyz=None, tolerance_m: float = 0.01) -> dict:
        """Servo the grip site to a world position (and optional wxyz orientation, e.g. from
        top_down_grasp_quat). No planner: plan clearance yourself (rise, translate, descend).
        Raises when the target is not reached — usually out of reach: drive_to_fixture() or
        drive_to() closer first.
        """
        return self._env.goto_pose(position, quaternion_wxyz, tolerance_m=tolerance_m)

    @tool
    def move_by_displacement(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
        """Move the end-effector by a world-frame offset, keeping its orientation."""
        pose = self._env.get_eef_pose()
        return self._env.goto_pose(np.asarray(pose[:3]) + np.array([x, y, z]), pose[3:7])

    @tool
    def base_move(
        self, forward: float = 0.0, left: float = 0.0, turn: float = 0.0, steps: int = 20
    ) -> list[float]:
        """Open-loop base velocities in [-1, 1] (forward/left in the current heading frame,
        turn = yaw rate; ~2 cm and ~0.065 rad per step at 1.0) for `steps` control steps.
        Returns the new base pose [x, y, yaw].
        """
        return self._env.move_base(forward, left, turn, steps=steps).tolist()

    @tool
    def get_fixture_extent(self, name: str) -> dict:
        """Footprint of a fixture: world pos, yaw, width/depth/height and its 8 bounding-box
        corners — use it to choose a standing spot beside a counter or appliance.
        """
        return self._env.fixture_extent(name)

    @tool
    def get_base_placement(self, fixture: str, ref_object: str | None = None) -> list[float]:
        """The standing pose [x, y, yaw] RoboCasa itself uses in front of `fixture` (optionally
        centred on `ref_object` placed on it): a known-good, collision-free spot.
        """
        return self._env.base_placement_for(fixture, ref_object).tolist()

    @tool
    def drive_to_fixture(self, fixture: str, ref_object: str | None = None) -> dict:
        """Drive the base to RoboCasa's standing pose in front of `fixture` (see
        get_base_placement). The arm is first retracted to its reset pose relative to the
        base (an extended arm rams furniture); a held object stays held. Returns the
        drive_to report plus the target pose and the retraction report under "tuck".
        """
        return self._env.drive_to_fixture(fixture, ref_object)

    @tool
    def drive_to(self, x: float, y: float, yaw: float | None = None) -> dict:
        """Closed-loop drive of the base to a world (x, y) and optional heading. Stops and
        reports reached=False when blocked by furniture; stand ~0.5 m from a fixture's
        front face, facing it, before manipulating.
        """
        return self._env.drive_to(x, y, yaw)

    @tool
    def place_gently(self, object_name: str, target_xy: list | None = None) -> dict:
        """Lower the HELD object in 1.5 cm steps until it rests (it stops following the hand),
        release and retreat 6 cm. With target_xy the descent also slides the hand (<= 8 mm a
        step) toward putting the OBJECT on that spot -- it hangs 1-2 cm off the grip site.
        Empty jaws return at once with held=False instead of a fake descent. Returns
        {"rested", "held", "descent_m", "object_z", "xy_error_m"}.
        """
        return self._env.place_gently(object_name, target_xy=target_xy)

    @tool
    def get_grasp_axis(self, object_name: str) -> dict:
        """Which of the object's own axes the fingers must close across, and how wide it is:
        {"yaw" (pass to top_down_grasp_quat), "width_m", "height_m", "bottom_z", "top_z",
        "fits_jaw"}. fits_jaw False means no top-down pinch exists -- grasp it side-on.
        """
        return self._env.grasp_axis(object_name)

    @tool
    def pick_object(
        self,
        object_name: str,
        grip_frac: float = 0.0,
        below_m: float = 0.02,
        approach_m: float = 0.12,
        lift_m: float = 0.15,
        attempts: int = 2,
        oriented: bool = True,
    ) -> dict:
        """Top-down pinch on an object and lift it, VERIFIED (pads on material AND the object
        rose); a failed attempt restages and descends below_m deeper. The descent is a
        floor-seeking request: it is aimed at bottom_z - below_m because a goto stalls
        1.4-1.6 cm high and bottoms out 2-3 cm above the support whatever it was asked for,
        so aiming at the object's upper third closes the jaws on air. Approaches
        position-only and rotates in place (an oriented goto into a cavity stalls and
        poisons the next call). Returns {"grasped", "opening_m", "rise_m", "attempts",
        "width_m", "fits_jaw", "site_above_bottom_m", "oriented"}.
        """
        return self._env.pick_object(
            object_name,
            grip_frac=grip_frac,
            below_m=below_m,
            approach_m=approach_m,
            lift_m=lift_m,
            attempts=attempts,
            oriented=oriented,
        )

    @tool
    def place_object(
        self,
        object_name: str,
        x: float | None = None,
        y: float | None = None,
        surface_z: float | None = None,
        clearance_m: float = 0.05,
        gentle: bool = True,
        force: bool = False,
    ) -> dict:
        """Put the HELD object down with its OWN centre at (x, y) -- it hangs 1-2 cm off the
        grip site, so the tool is offset by the difference measured now. With surface_z the
        release is the cheap one (open from surface_z + height + clearance_m and wait, ~27
        frames); with gentle the descent then searches for contact instead of dropping.
        Empty jaws are reported, not placed, and a carry that stopped more than 6 cm from
        the target KEEPS HOLDING (a release there is a drop) unless force=True -- lift the
        torso, drive closer or re-aim and call again. Returns {"placed", "held",
        "carry_error_m", "xy_error_m", "object_z", "descent_m"}.
        """
        return self._env.place_object(
            object_name,
            x=x,
            y=y,
            surface_z=surface_z,
            clearance_m=clearance_m,
            gentle=gentle,
            force=force,
        )

    @tool
    def retreat_from(self, object_name: str, distance_m: float = 0.28) -> dict:
        """Back the hand away from the object until the grip site clears distance_m, moving
        HORIZONTALLY toward the base first (a vertical goto the arm cannot make oscillates
        the hand and sweeps the object off what it was just put on). Every pick-and-place
        scorer ends in gripper_obj_far (0.25 m; 0.15 m for the fridge tasks). Returns
        {"clear", "gap_m", "hops"}.
        """
        return self._env.retreat_from(object_name, distance_m=distance_m)

    @tool
    def open_gripper(self) -> float:
        """Open the gripper fully; returns the finger-pad opening in metres."""
        return self._env.set_gripper(closed=False)

    @tool
    def close_gripper(self) -> float:
        """Close the gripper (grasp); returns the finger-pad opening in metres (~0.009 when
        it closed on nothing, larger when something is held).
        """
        return self._env.set_gripper(closed=True)

    @tool
    def wait(self, steps: int = 10) -> None:
        """Let the simulation settle for N control steps."""
        self._env.hold(steps)
