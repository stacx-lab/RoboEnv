"""Route A: planner-level control API for RoboTwin dual Piper arms.

All motion goes through RoboTwin's own motion layer (mplib/curobo planned,
collision-checked screw/RRT trajectories). Object poses are privileged
(read directly from the simulator).
"""

# Moved from third_party/cap-x/capx/integrations/piper/robotwin_planner.py on 2026-08-26 (refactor P2); cap-x keeps a shim.

from __future__ import annotations

from typing import Any

import numpy as np
from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class PiperPlannerPrivilegedApi(ApiBase):
    """Planner-level control of a dual-arm Piper robot with privileged object poses.

    Conventions:
    - All positions/orientations are in the WORLD frame.
    - Quaternions are WXYZ order: [qw, qx, qy, qz].
    - Gripper fraction: 0.0 = fully closed, 1.0 = fully open.
    - The left arm is at negative X, the right arm at positive X.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "list_objects": self.list_objects,
            "get_object_pose": self.get_object_pose,
            "get_gripper_pose_left": self.get_gripper_pose_left,
            "get_gripper_pose_right": self.get_gripper_pose_right,
            "goto_pose_left": self.goto_pose_left,
            "goto_pose_right": self.goto_pose_right,
            "goto_pose_both": self.goto_pose_both,
            "move_by_displacement_left": self.move_by_displacement_left,
            "move_by_displacement_right": self.move_by_displacement_right,
            "open_gripper_left": self.open_gripper_left,
            "close_gripper_left": self.close_gripper_left,
            "open_gripper_right": self.open_gripper_right,
            "close_gripper_right": self.close_gripper_right,
            "grasp_object": self.grasp_object,
            "find_grasp": self.find_grasp,
            "place_object": self.place_object,
            "get_object_functional_pose": self.get_object_functional_pose,
            "back_to_origin_left": self.back_to_origin_left,
            "back_to_origin_right": self.back_to_origin_right,
            "checkpoint": self.checkpoint,
            "servo_delta_left": self.servo_delta_left,
            "servo_delta_right": self.servo_delta_right,
            "restore": self.restore,
            "drag_to_point": self.drag_to_point,
            "goto_pose_flex_left": self.goto_pose_flex_left,
            "goto_pose_flex_right": self.goto_pose_flex_right,
            "glide_to_pose_left": self.glide_to_pose_left,
            "glide_to_pose_right": self.glide_to_pose_right,
            "check_reachable_left": self.check_reachable_left,
            "check_reachable_right": self.check_reachable_right,
            "probe_reach": self.probe_reach,
            "carry_in_planner": self.carry_in_planner,
            "avoid_in_planner": self.avoid_in_planner,
            "get_planner_world": self.get_planner_world,
            "open_articulated": self.open_articulated,
            "place_gently": self.place_gently,
            "place_held_object": self.place_held_object,
            "transfer_object": self.transfer_object,
        }

    # ------------------- servo (closed-loop micro-motion) -------------------

    def servo_delta_left(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list:
        """Move the LEFT end-effector by a SMALL straight-line delta (<= 6 cm
        total; larger raises) — fast, deterministic, no path planning. This is
        the closed-loop building block: read poses or contacts, servo a small
        step, read again, repeat. Ideal for final approach, precise alignment,
        gentle pressing, and synchronized dual-arm adjustments.

        It does NOT avoid obstacles and does not even see them (measured
        2026-08-27: the screw planner skips its own collision check and the
        planning world holds only table/wall/ground). A blocked step therefore
        does NOT raise — it reports success and the hand stops wherever the
        physics stops it, short of the target. That is why this returns the
        pose it actually reached: compare it with what you asked for, or watch
        the thing you are pushing. It raises only when the straight line hits a
        joint limit or starts from a singular posture (the reset/home pose is
        one — goto_pose somewhere first).

        For a step that also reorients the hand, use glide_to_pose_left.

        Args:
            x, y, z: World-frame displacement in meters.

        Returns:
            The end-effector pose actually reached [x, y, z, qw, qx, qy, qz].
        """
        return self._env.plan_servo_delta("left", x, y, z)

    def servo_delta_right(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list:
        """Move the RIGHT end-effector by a SMALL straight-line delta (<= 6 cm
        total; larger raises) — fast, deterministic, no path planning. This is
        the closed-loop building block: read poses or contacts, servo a small
        step, read again, repeat. Ideal for final approach, precise alignment,
        gentle pressing, and synchronized dual-arm adjustments.

        It does NOT avoid obstacles and does not even see them (measured
        2026-08-27: the screw planner skips its own collision check and the
        planning world holds only table/wall/ground). A blocked step therefore
        does NOT raise — it reports success and the hand stops wherever the
        physics stops it, short of the target. That is why this returns the
        pose it actually reached: compare it with what you asked for, or watch
        the thing you are pushing. It raises only when the straight line hits a
        joint limit or starts from a singular posture (the reset/home pose is
        one — goto_pose somewhere first).

        For a step that also reorients the hand, use glide_to_pose_right.

        Args:
            x, y, z: World-frame displacement in meters.

        Returns:
            The end-effector pose actually reached [x, y, z, qw, qx, qy, qz].
        """
        return self._env.plan_servo_delta("right", x, y, z)

    # ------------------- checkpoint / rollback -------------------

    def checkpoint(self) -> str:
        """Save a snapshot of the ENTIRE simulation state (every object and
        robot pose, velocity, joint state). Returns a checkpoint id.

        Use before any risky, hard-to-undo maneuver (a tricky grasp, a
        transport over the table edge, a release onto a small target): if it
        goes wrong, restore(id) rewinds the world and you try a different
        approach. Checkpoint liberally — snapshots are cheap. The snapshot
        includes the PD drive targets, so an object held at checkpoint time is
        still held after restore.

        Returns:
            Checkpoint id string for restore().
        """
        return self._env.checkpoint()

    def restore(self, checkpoint_id: str) -> None:
        """Rewind the simulation to a previously saved checkpoint. Objects
        return to where they were, dropped/flung objects are recovered, the
        arms return to their saved configuration, and a gripper that was
        holding something is still holding it. The step budget is NOT
        refunded — steps spent since the checkpoint stay spent, so rehearse
        efficiently.

        Args:
            checkpoint_id: Id returned by checkpoint().
        """
        self._env.restore(checkpoint_id)

    # ------------------- higher-level motion primitives -------------------

    def drag_to_point(
        self, object_name: str, arm: str, contact_point_id: int = 1, max_iters: int = 15
    ) -> int:
        """Drag an ARTICULATED part (laptop lid, door, drawer, handle) along
        its hinge/slide arc: after touching or holding the part, this
        repeatedly pulls the gripper onto the part's MOVING annotated contact
        point, opening it a little more each iteration. Check joint progress
        with get_joint_states() between calls.

        Prefer open_articulated() for drawers and doors: it measures joint
        progress, searches where on the handle to grip, re-aims on every stall
        and gives up on a deterministic budget.
        Use drag_to_point when the part has no usable joint reading, or for a
        single nudge to expose a contact point. If the part gets pushed fully
        closed/flat first, its contact points may become ungraspable and no
        motion can hook it anymore — avoid bumping the part shut.

        Args:
            object_name: The articulated object (see list_objects()).
            arm: 'left' or 'right'.
            contact_point_id: Annotated contact point to chase (default 1).
            max_iters: Max pull iterations.

        Returns:
            Number of successful pull iterations (0 = no progress made).
        """
        return self._env.plan_drag_articulated(
            object_name, arm, contact_point_id=contact_point_id, max_iters=max_iters
        )

    def goto_pose_flex_left(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray | None = None
    ) -> list:
        """Move the LEFT end-effector to a position, automatically sweeping
        wrist YAW (about world Z, 0/+-30/+-60/+-90/135/180 deg from a top-down
        tool) until one has an IK solution. Use when goto_pose keeps failing
        with IK errors at a position you believe is reachable. Returns the
        quaternion actually used. It does NOT sweep tool PITCH, which is the
        degree of freedom that decides reachability near the envelope edge, so
        'no wrist orientation reaches' here does not prove the point is
        unreachable — try an explicit pitched quaternion."""
        return self._env.plan_move_to_pose_flex("left", position, quaternion_wxyz)

    def open_articulated(
        self,
        object_name: str,
        arm: str | None = None,
        target_fraction: float = 0.6,
        max_planner_calls: int = 300,
        time_budget_s: float | None = None,
    ) -> dict:
        """Open an articulated part (drawer / door / lid) to target_fraction
        of its joint limit in ONE call. arm=None picks the arm automatically.

        Internally: hooks the handle on its annotated contact points,
        discovers the pull direction, pulls stroke-by-stroke with the joint
        position as the progress meter in decreasing step sizes,
        RE-DISCOVERS the direction whenever a stroke stops paying (a hinged
        part's tangent rotates as it swings, so a fixed pull becomes an
        off-axis shove after a few centimetres), and — when strokes stop
        paying at every heading — SLIDES THE GRIP ALONG THE HANDLE and pulls
        again, keeping whichever grip point got furthest.

        Where on the handle you hold is usually what decides how far a drawer
        opens: the arm folds up long before the joint runs out, and a
        cross-body grip reaches far further than one in front of the arm's own
        shoulder (measured on the cabinet drawer: 34 % vs 69 % of travel).

        Verified 2026-08-27 under --planner-budget deterministic (seed 0):
        cabinet drawer asked 0.95 -> 0.77 of travel in 155 planner calls
        (asked 0.70 -> 0.72 in 96, still gripping the drawer), microwave door
        asked 0.70 -> 0.31 in 160 calls; the laptop lid's 72 % predates the
        stall re-discovery and was not re-measured. Assumes a handle pulled
        horizontally (the grip slides along cross(pull, up)); progress
        thresholds are in raw joint units (m for a slide, rad for a hinge).
        The joint worked is the one whose link carries the handle's annotated
        contact points (a cabinet has three identical drawer joints), else the
        largest-range one; 'holding' is measured against THAT link, and
        'gripper_on_links' lists every link the fingers touch (the neighbouring
        drawer front counts there, not as holding).

        IMPORTANT: a part held open against gravity snaps shut when released,
        so this does NOT let go — but it also does not pretend. Whether the
        jaws are still on the part is MEASURED and reported as 'holding' (the
        arm, or None) and 'gripper_on_part': a pull that ends at the arm's
        reach limit often ends with the handle slipped out of the fingers
        (measured: the microwave door stops at 31 % with the gripper touching
        no part of it). When it IS holding, do the task with the OTHER arm and
        open that gripper last. When it is not, nothing is holding the part
        open: a drawer on horizontal rails stays where it is, a gravity-loaded
        door may already be closing, so re-read get_joint_states() before you
        rely on the opening.

        Args:
            object_name: the articulated object (see list_objects()).
            arm: 'left' | 'right' | None (try right, then left).
            target_fraction: fraction of the joint's travel to reach.
            max_planner_calls: how much motion this call may spend before
                giving up. The COUNT is deterministic; the RRT plans inside
                (engage, park, the first-pull fallback) sample identically only
                under `roboenv replay --planner-budget deterministic` in a fresh
                process (155 calls / 0.7713 fresh vs 142 / 0.7717 after an
                earlier grasp in the same process, measured), which is what the
                certification path provides. Raise it for a stubborn part,
                lower it to fail fast on one you suspect is unreachable.
            time_budget_s: optional extra wall-clock valve (None = off).
                It makes the call non-deterministic; use it only while
                exploring.

        Returns {'opened': bool, 'qpos', 'target', 'fraction', 'strokes',
        'arm', 'holding', 'gripper_on_part', 'gripper_on_links',
        'grip_offset_m', 'planner_calls', 'reason', 'note'};
        'reason' is 'opened' | 'arm reach limit' | 'stroke cap' |
        'planner budget' | 'time budget' | 'already open'. 'arm reach limit'
        means the arm ran out of travel with budget to spare — more budget
        will not help, a different arm or a re-grasp might; 'stroke cap' means
        the per-phase stroke count ran out first. Raises only when no handle
        on the object could be hooked at all."""
        return self._env.plan_open_articulated(
            object_name,
            arm,
            target_fraction=target_fraction,
            max_planner_calls=max_planner_calls,
            time_budget_s=time_budget_s,
        )

    def place_held_object(
        self,
        object_name: str,
        arm: str,
        target_position: np.ndarray,
        hover: float = 0.10,
        release: bool = True,
    ) -> dict:
        """Put the object you are ALREADY HOLDING down so that ITS OWN ORIGIN
        lands on target_position, searching wrist orientations until one is
        reachable and then descending vertically onto the spot.

        Use this instead of place_object when the target is deep inside a
        container or near the edge of the arm's envelope. Four differences that
        matter:
        (1) it aims the OBJECT, not the hand, and re-measures the hand-to-object
            offset at every waypoint — the gripper does not hold things at their
            centre (a box pinched near its top hangs 19 cm from the
            end-effector) and the grip slips while you carry;
        (2) the search sweeps wrist TILT as well as yaw — a top-down tool puts
            the end-effector 12 cm straight above the target, often outside the
            envelope, while the same target with the tool pitched 20-35 deg puts
            it 5-6 cm nearer the robot and 1 cm lower;
        (3) it lifts the load clear before travelling and glides in straight
            lines — the motion planner does not model what is in the gripper,
            so a planned transit drags the object through whatever is in the way;
        (4) it steps down until the object stops following the hand before
            opening the jaws, so the object is set down rather than dropped.

        Args:
            object_name: The object currently in this arm's gripper.
            arm: 'left' or 'right'.
            target_position: (3,) XYZ where the object's ORIGIN should end up.
                Note that many assets put their origin at the bottom face.
            hover: Height above the release pose to approach from (vertical
                descent, so the object enters a container's mouth from above).
            release: Open the gripper and retreat upward after arriving.

        It also judges each orientation by WHERE THE OBJECT ENDED and by
        WHAT ELSE MOVED, not by whether the arm could plan to the pose: a
        candidate that leaves the object more than ~2.5 cm off, or that shoves
        an articulated fixture (the drawer you are reaching into) by more than
        2 cm, is rewound and the next one tried. You get the best of up to
        5 attempts.

        Returns {'placed': bool, 'quat_wxyz', 'tilt_deg', 'yaw_deg',
        'object_position', 'error_m', 'xy_error_m', 'released', 'attempts',
        'articulation_disturbed', 'accepted'}. 'accepted' False means this was
        the best of the candidates but still off — check 'xy_error_m' and
        'articulation_disturbed' before believing it.
        When no orientation reaches the target at all nothing is left
        disturbed and 'placed' is False."""
        return self._env.plan_place_held_object(
            object_name, arm, target_position, hover=hover, release=release
        )

    def glide_to_pose_left(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> list:
        """Move the LEFT hand to a full pose (position AND orientation) in ONE
        straight, monotone screw motion — the general form of
        servo_delta_left, which is its translation-only face.

        Use it instead of goto_pose whenever you are CARRYING something: the
        motion planner behind goto_pose does not model what is in the gripper,
        so its RRT path dips and swings the load through tables and shelves,
        while a glide never leaves the direct line. Use it also for a
        deterministic, quick (~0.2 s) reorientation in place.

        Measured 2026-08-27 on this arm (holding the scene object):
        - a pure translation is straight to a few mm (3 cm steps end 1.3-2.7 mm
          off, an 8 cm move 3.6 mm off with 2.9 mm of chord deviation);
        - a combined translate-and-rotate BOWS off the chord: 0.3 / 0.7 / 1.2 cm
          over 8 cm at 0 / 30 / 60 deg of yaw (the end link is 12 cm off the
          screw axis) — split a large reorientation from the travel when
          clearance is tight;
        - a reorientation "in place" pivots about the END-EFFECTOR frame, 12 cm
          behind the fingertips, so a carried object sweeps ~0.12*sin(angle)
          plus its own offset (a 30 deg yaw moved the held box 1.2 cm): re-read
          its pose afterwards;
        - a request within 0.2 mm / 0.2 mrad of the current pose returns the
          current pose without moving (mplib would otherwise refuse it);
        - it does NOT avoid obstacles, and it does not even see them: mplib's
          screw planner skips its own collision check and the planning world
          holds only table/wall/ground. A target through solid geometry ends
          one of two ways and NEITHER is arrival — it reports Success and the
          physics stops the hand short (measured: aimed 10 cm below the
          tabletop, reported Success, ended 0.24 m short), or the screw plan is
          rejected for the joint limits ALONG the way, which reads like an
          obstacle error but is not one. You cannot tell which from the return
          value, so check the pose.

        It raises on a joint limit along the way, a singular posture (the
        reset/home pose is one — goto_pose somewhere first), a ~180-degree
        reorientation, a path longer than 2500 steps, or a non-unit /
        non-wxyz quaternion (ValueError); the message names which. Arrival is
        NOT checked: this returns the pose the hand actually reached, so
        compare it against what you asked for, and read the pose of whatever
        you carry rather than assuming it came along.

        Args:
            position: (3,) world XYZ for the end-effector frame (which sits
                0.12 m behind the fingertips along the approach axis — see
                get_tcp_pose).
            quaternion_wxyz: (4,) target orientation.

        Returns:
            The end-effector pose actually reached [x, y, z, qw, qx, qy, qz].
        """
        return self._env.plan_glide_to_pose("left", position, quaternion_wxyz)

    def glide_to_pose_right(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> list:
        """Move the RIGHT hand to a full pose in ONE straight screw motion —
        see glide_to_pose_left for the full contract. In short: straight and
        deterministic, the right tool for moving a CARRIED object (the RRT
        behind goto_pose drags it through obstacles), blind to collisions (it
        stops where the physics stops it and still reports success), raises
        only on joint limits / a singular home pose / a 180-degree flip, and
        returns the pose actually reached — check it.

        Args:
            position: (3,) world XYZ for the end-effector frame.
            quaternion_wxyz: (4,) target orientation.

        Returns:
            The end-effector pose actually reached [x, y, z, qw, qx, qy, qz].
        """
        return self._env.plan_glide_to_pose("right", position, quaternion_wxyz)

    def check_reachable_left(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray | None = None
    ) -> dict:
        """Planning-only probe: can the LEFT arm reach this gripper position?
        Rehearses the real motion and rewinds, so the robot and the world end
        exactly as they were (anything held stays held) — only the step budget
        is spent. Returns {'reachable': bool, 'yaw', 'quat_wxyz'}.

        CAVEAT: it sweeps wrist YAW about the world Z axis only. Reachability
        near the envelope edge is governed far more by tool PITCH — a target
        that fails every yaw here is often reachable with the tool tipped
        20-35 deg. probe_reach() sweeps PITCH, with IK instead of rehearsed
        motions (milliseconds, not seconds); place_held_object searches it too."""
        return self._env.plan_check_reachable("left", position, quaternion_wxyz)

    def check_reachable_right(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray | None = None
    ) -> dict:
        """Planning-only probe: can the RIGHT arm reach this gripper position?
        Rehearses the real motion and rewinds, so nothing is disturbed and
        anything held stays held. Returns {'reachable', 'yaw', 'quat_wxyz'}.
        Sweeps wrist YAW only — see check_reachable_left on why pitch matters
        more at the envelope edge, and probe_reach() for the pitch sweep."""
        return self._env.plan_check_reachable("right", position, quaternion_wxyz)

    def goto_pose_flex_right(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray | None = None
    ) -> list:
        """Move the RIGHT end-effector to a position, automatically sweeping
        wrist YAW about world Z until one has an IK solution. Returns the
        quaternion actually used. Like its left twin it does NOT sweep tool
        pitch, so a failure here is not proof the point is unreachable."""
        return self._env.plan_move_to_pose_flex("right", position, quaternion_wxyz)

    def probe_reach(
        self,
        arm: str,
        position: np.ndarray,
        at: str = "fingertips",
        elevations: Any = (90, 75, 60, 45, 30, 15),
        azimuths: Any = None,
        rolls: Any = (0,),
        max_probes: int = 120,
        confirm: int = 1,
    ) -> dict:
        """Which tool ORIENTATIONS put this hand on this point? — a PITCH sweep.

        "Out of reach" usually means "you only tried pointing straight down". The
        end-effector sits a hand's length back along the tool axis, so tipping the
        tool moves the wrist several centimetres for the SAME fingertip point:
        measured on the RoboTwin table, fingertips at (-0.05, -0.05, 0.90) have no
        top-down solution for either arm and are solvable tipped 15-30 degrees off
        vertical. check_reachable_* / goto_pose_flex_* sweep wrist YAW only, and pay
        a rehearsed motion per probe; this is IK-only (~1-30 ms each), moves
        nothing, and touches nothing.

        Args:
            arm: 'left' or 'right'.
            position: (3,) world point for the FINGERTIPS (at='fingertips', the
                default — the fingertip-to-ee offset is read from the robot at call
                time, following RoboTwin's tcp convention rather than a written-down
                number) or for the end-effector frame (at='ee'). The wrist roll left
                free by (elevation, azimuth, roll) is inherited from the hand's current
                orientation, so the same request can answer differently from a
                different pose.
            elevations: dive angle(s) of the approach in degrees: 90 = straight
                down, 0 = horizontal, negative = pointing up. Tried in order.
            azimuths: compass heading(s) of the approach in the world xy plane
                (degrees). None = straight out from THIS arm's base.
            rolls: extra rotation(s) about the tool axis, degrees.
            max_probes: bound on IK probes (a COUNT, not wall clock).
            confirm: how many IK successes to re-check with a real path plan
                (seconds each). IK success means the pose is in the envelope, not
                that a collision-free path exists.

        Returns:
            {'reachable', 'best', 'tried', 'probes_used', 'plans_used',
             'budget_exhausted', 'tool_offset_m', 'method'}; feed best's
            'ee_position' + 'quat_wxyz' straight to goto_pose_* / glide_to_pose_*.
        """
        return self._env.plan_reach_probe(
            arm,
            position,
            at=at,
            elevations=elevations,
            azimuths=azimuths,
            rolls=rolls,
            max_probes=max_probes,
            confirm=confirm,
        )

    def carry_in_planner(self, object_name: str, arm: str, carrying: bool = True) -> dict:
        """Tell the motion planner that this hand is CARRYING this object.

        The planner's world is built before the props are spawned and is never
        refreshed, so it holds only the table, the wall, the ground and the two
        arms: every transit is planned as if the jaws were EMPTY, which is how a
        carried object gets dragged along the table edge or stripped out of the
        fingers. This attaches the object's collision geometry to the gripper link,
        so from now on plans for this arm carry it — and the other arm stops
        planning through it.

        Call it after a successful grasp; call it with carrying=False before a
        deliberate contact (a hand-over, seating a lid, a place) and after the
        release, because an attached payload that MUST touch something makes every
        approach unplannable.

        CAVEAT, measured 2026-08-27: on the mplib this checkout ships (0.2.1) the
        call RAISES instead, and its message carries the measurement. That build
        places an attached body from the arm's base-frame link pose while the arm's
        collision links and all obstacles are in the world frame, so the payload
        lands about a metre away and every plan for that arm then fails with "IK
        Failed". The check is live, so the tool starts working if that is ever
        fixed. Meanwhile: avoid_in_planner() for the obstacles, LIFT the payload
        clear before travelling, and check_grasp() to see if it is still held.

        Returns {'carrying', 'object', 'arm', 'link', 'geometry', 'start_colliding',
        'colliding_with', 'world_objects'}. start_colliding=True means the payload
        is already intersecting something (usually the table it was lifted from) and
        every plan for this arm will fail until you lift clear.
        """
        return self._env.plan_carry_in_planner(object_name, arm, carrying=carrying)

    def avoid_in_planner(self, object_name: str, avoid: bool = True) -> dict:
        """Make the motion planner SEE a task object (it was never told about any).

        The planning world holds only the table, the wall, the ground and the arms,
        so a transit plans straight through a standing bottle. This adds the object
        to both arms' worlds with its pose kept live; avoid=False removes it again.
        Adding an object standing where you must reach can make a pose unplannable
        that used to plan — that is the point, but it means adding the whole scene
        is rarely wise. Registering ANYTHING also starts refreshing both arms' live
        configurations into the planning world before every plan, so plans become
        stricter (and more honest) for motions that have nothing to do with this
        object. Returns {'object', 'avoided', 'arms', 'world_objects'}.
        """
        return self._env.plan_avoid_in_planner(object_name, avoid=avoid)

    def get_planner_world(self, arm: str = "left") -> dict:
        """What the motion planner knows about the scene (observation only):
        {'objects' (the only PROPS a plan avoids - everything in 'articulations',
        i.e. both arms, is collision-checked too), 'attached' (what it believes the
        hand carries), 'articulations', 'tracked_live', 'scene_objects',
        'colliding_now'}. The gap between 'objects' and 'scene_objects' is the
        planner's blind spot; 'colliding_now' is what it currently considers
        intersecting at the arm's REAL configuration (an object you added with
        avoid_in_planner and are holding shows up here as the fingers touching it)."""
        return self._env.plan_planner_world(arm)

    def place_gently(self, object_name: str, arm: str, target_name: str | None = None) -> dict:
        """Lower the HELD object in small vertical steps until it physically
        contacts the target object (or the table), only then open the gripper,
        and retreat straight up. Use for precision placement on small or
        raised surfaces where dropping from height bounces the object off.

        The release is REHEARSED: after opening the jaws the sim is stepped
        briefly and, if the object did not stay put, the world is rewound to
        the moment before the release and you are still holding it — so a
        failed placement costs steps but leaves nothing on the floor. Look for
        'world_restored_to_pre_release' in the result to tell that case apart.

        Args:
            object_name: The held object.
            arm: 'left' or 'right' (the arm holding it).
            target_name: Object to rest it on; None = the table.

        Returns:
            {'released_on', 'contact_before_release', 'settled' (did the
            object STAY on the support after physics settled — if False it
            slid off; regrasp and try a flatter release),
            'drift_after_release_m', and 'world_restored_to_pre_release': True
            only when it rewound a failed release}
        """
        return self._env.plan_place_gently(object_name, arm, target_name)

    def transfer_object(
        self,
        object_name: str,
        from_arm: str,
        to_arm: str,
        via_xy: list | None = None,
        via_z: float | None = None,
    ) -> None:
        """Relay an object from one arm to the other when the first arm
        cannot reach the goal region: parks it at a mid-table spot both arms
        reach (default x=0, y=-0.12), releases, and regrasps with the other
        arm. Raises if any stage fails to plan. The handoff spot is a plain
        table drop, so it is unsuitable for objects that will not stand up on
        their own.

        Args:
            object_name: Object currently held or graspable by from_arm.
            from_arm / to_arm: 'left' / 'right'.
            via_xy: Optional [x, y] handoff spot override.
            via_z: Optional release height. By default the height is MEASURED
                from the top of whatever surface lies under the handoff point
                (+1 cm), so it follows the scene's own table rather than a
                baked-in number.
        """
        self._env.plan_transfer_object(object_name, from_arm, to_arm, via_xy, via_z)

    # ------------------------- perception (privileged) -------------------------

    def list_objects(self) -> list[str]:
        """List every name the scene's objects answer to — including fixtures
        (table, wall) and the articulated furniture, not only what you can pick
        up.

        THE SAME OBJECT USUALLY APPEARS TWICE, under two kinds of name:
        - its ASSET name, e.g. '036_cabinet', '113_coffee-box';
        - its ROLE in this task, e.g. 'cabinet', 'object', 'target_box'.
        Both address the same body. Prefer the role name: assets are drawn at
        random per episode, so the role name is the only stable handle on what
        a thing is FOR (it is what the benchmark's own instruction refers to).
        Where a task holds several instances of one asset the names are
        suffixed by their slot (object1, object2, ... / bottles0, bottles1).

        Returns:
            List of name strings usable with get_object_pose/grasp_object.

        Example:
            objects = list_objects()
            # put_object_cabinet -> ['036_cabinet', '113_coffee-box',
            #                        'cabinet', 'object', 'table', 'wall']
        """
        return self._env.list_objects()

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get the world-frame pose of an object.

        Args:
            object_name: Name of the object (see list_objects()).

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion.

        Example:
            pos, quat = get_object_pose('box')
        """
        return self._env.get_object_pose(object_name)

    def get_gripper_pose_left(self) -> np.ndarray:
        """Get the left end-effector pose in the world frame.

        Returns:
            (7,) array [x, y, z, qw, qx, qy, qz].
        """
        return self._env.get_arm_pose("left")

    def get_gripper_pose_right(self) -> np.ndarray:
        """Get the right end-effector pose in the world frame.

        Returns:
            (7,) array [x, y, z, qw, qx, qy, qz].
        """
        return self._env.get_arm_pose("right")

    # ------------------------- motion (planned) -------------------------

    def goto_pose_left(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Move the LEFT arm end-effector to a world-frame pose (collision-checked plan).

        Raises an error if planning fails (arm does not move); catch it or pick
        a reachable pose.

        Args:
            position: (3,) XYZ target in meters.
            quaternion_wxyz: (4,) WXYZ target orientation.

        Example:
            goto_pose_left(np.array([-0.1, -0.1, 0.9]), np.array([0, 1, 0, 0]))


        ORIENTATIONS (wxyz) that this arm's IK reliably accepts — guessed
        quaternions almost always fail with "IK Failed". Approach direction is
        the tool's local +x, and the fingertips sit 0.12 m along it:
        top-down (approach 0,0,-1)          [-0.5,   0.5,   -0.5,   -0.5]
        top-down, jaws rolled +30 deg       [-0.354, 0.612, -0.354, -0.612]
        top-down, jaws rolled -30 deg       [-0.612, 0.354, -0.612, -0.354]
        facing front (approach 0,+1,0)      [-0.707, 0,      0,     -0.707]
        facing left (approach -1,0,0)       [ 0,     0,      0,     -1]
        facing right (approach +1,0,0)      [-1,     0,      0,      0]
        PITCHED top-down — down AND forward, the ones that buy reach at the
        far edge of the envelope because they pull the end-effector back
        toward the robot for the same fingertip point:
        pitch 20 deg (approach 0,+0.34,-0.94) [-0.579, 0.406, -0.406, -0.579]
        pitch 30 deg (approach 0,+0.50,-0.87) [-0.612, 0.354, -0.354, -0.612]
        pitch 35 deg (approach 0,+0.57,-0.82) [-0.627, 0.327, -0.327, -0.627]
        (The two "rolled" quaternions above look almost identical to the
        pitched ones but their approach is exactly straight down — they are a
        wrist roll, not a tilt. Check the sign pattern.)
        To press/push from above, use top-down a few cm up, then
        move_by_displacement down.
        """
        self._env.plan_move_to_pose("left", position, quaternion_wxyz)

    def goto_pose_right(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Move the RIGHT arm end-effector to a world-frame pose (collision-checked plan).

        Args:
            position: (3,) XYZ target in meters.
            quaternion_wxyz: (4,) WXYZ target orientation.


        ORIENTATIONS: same table as goto_pose_left — top-down
        [-0.5, 0.5, -0.5, -0.5], facing front [-0.707, 0, 0, -0.707], facing
        left [0, 0, 0, -1], facing right [-1, 0, 0, 0], and the PITCHED
        top-downs [-0.579, 0.406, -0.406, -0.579] (20 deg),
        [-0.612, 0.354, -0.354, -0.612] (30 deg) and
        [-0.627, 0.327, -0.327, -0.627] (35 deg) that reach targets a
        straight-down tool cannot. To press/push from above, use top-down a
        few cm up, then move_by_displacement down.
        """
        self._env.plan_move_to_pose("right", position, quaternion_wxyz)

    def goto_pose_both(
        self,
        left_position: np.ndarray,
        left_quaternion_wxyz: np.ndarray,
        right_position: np.ndarray,
        right_quaternion_wxyz: np.ndarray,
    ) -> None:
        """Move BOTH arms simultaneously to world-frame poses.

        Args:
            left_position: (3,) XYZ target for the left arm.
            left_quaternion_wxyz: (4,) WXYZ target for the left arm.
            right_position: (3,) XYZ target for the right arm.
            right_quaternion_wxyz: (4,) WXYZ target for the right arm.
        """
        self._env.plan_move_both(
            left_position, left_quaternion_wxyz, right_position, right_quaternion_wxyz
        )

    def move_by_displacement_left(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quaternion_wxyz: list | None = None,
        move_axis: str = "world",
    ) -> None:
        """Translate the LEFT arm end-effector by a displacement, optionally
        rotating it to a new orientation during the move.

        Args:
            x: Displacement along X in meters (world frame only).
            y: Displacement along Y in meters (world frame only).
            z: Displacement along Z in meters: with move_axis='world' positive
                is up; with move_axis='arm' see below.
            quaternion_wxyz: Optional target end-effector orientation [w,x,y,z]
                reached at the END of the same planned motion (e.g. turning a
                held object to present it to the other arm); omit to keep the
                current one.  Must be 4 finite numbers; it is normalised.
            move_axis: 'world' (default) reads x/y/z in the world frame.
                'arm' IGNORES x and y (they raise ValueError when non-zero) and
                moves the hand by z along its CURRENT approach (finger) axis,
                sign-flipped: z > 0 backs the hand away from whatever the
                fingers point at — UP after a top-down grasp, backward and
                level at the home pose where the tool points forward — and
                z < 0 advances.  The direction is taken from the orientation
                BEFORE the turn; read get_tcp_pose(arm) if you need to know
                where the approach axis points.  Measured 2026-08-27 (round 3
                review): x=0.05 / y=0.05 with 'arm' moved the hand 0.000 m.

        Example:
            move_by_displacement_left(z=0.1)  # lift 10 cm (world)
            move_by_displacement_left(z=0.08, move_axis='arm')  # retreat along the approach
        """
        self._env.plan_move_by_displacement(
            "left", x=x, y=y, z=z, quat_wxyz=quaternion_wxyz, move_axis=move_axis
        )

    def move_by_displacement_right(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quaternion_wxyz: list | None = None,
        move_axis: str = "world",
    ) -> None:
        """Translate the RIGHT arm end-effector by a displacement, optionally
        rotating it to a new orientation during the move.

        Args:
            x: Displacement along X in meters (world frame only).
            y: Displacement along Y in meters (world frame only).
            z: Displacement along Z in meters (world: positive = up).
            quaternion_wxyz: Optional target end-effector orientation [w,x,y,z]
                reached at the end of the move (e.g. for tilting or shaking a
                held object); omit to keep the current orientation.
            move_axis: 'world' (default) reads x/y/z in the world frame; 'arm'
                ignores x/y and moves by z along the hand's own approach axis,
                sign-flipped (z > 0 = retreat) — see the left twin.
        """
        self._env.plan_move_by_displacement(
            "right", x=x, y=y, z=z, quat_wxyz=quaternion_wxyz, move_axis=move_axis
        )

    # ------------------------- gripper -------------------------

    def open_gripper_left(self, fraction: float = 1.0) -> None:
        """Open the LEFT gripper.

        Args:
            fraction: Target opening, 0.0 (closed) to 1.0 (fully open).
        """
        self._env.set_gripper("left", fraction)

    def close_gripper_left(self, fraction: float = 0.0) -> None:
        """Close the LEFT gripper.

        Args:
            fraction: Target opening, 0.0 (fully closed) to 1.0 (open).
        """
        self._env.set_gripper("left", fraction)

    def open_gripper_right(self, fraction: float = 1.0) -> None:
        """Open the RIGHT gripper.

        Args:
            fraction: Target opening, 0.0 (closed) to 1.0 (fully open).
        """
        self._env.set_gripper("right", fraction)

    def close_gripper_right(self, fraction: float = 0.0) -> None:
        """Close the RIGHT gripper.

        Args:
            fraction: Target opening, 0.0 (fully closed) to 1.0 (open).
        """
        self._env.set_gripper("right", fraction)

    # ------------------------- high-level skills -------------------------

    def grasp_object(
        self,
        object_name: str,
        arm: str,
        pre_grasp_dis: float = 0.1,
        grasp_dis: float = 0.0,
        contact_point_ids: list[int] | int | None = None,
        verify: bool = True,
        search: bool = False,
    ) -> None:
        """Grasp an object with one arm using annotated grasp points
        (approach, then close gripper). Raises an error if planning fails,
        and also when THIS arm's fingers have no physical contact with the
        object afterwards ("closed on air"). The check matches contacts by
        entity identity, so it is not fooled by the other arm touching the
        object, and when you name contact_point_ids it requires contact on
        the very link those points ride on — grasping a cabinet's drawer is
        not satisfied by resting on the cabinet's frame. Verify at any later
        moment with check_grasp().
        A contact_point_id you name that has no plannable pose raises instead
        of falling back to a synthesised top-down pinch.

        Args:
            object_name: Name of the object (see list_objects()).
            arm: 'left' or 'right'.
            pre_grasp_dis: Approach offset distance in meters before the grasp.
            grasp_dis: Final offset distance in meters at the grasp.
            contact_point_ids: Optional annotated grasp point id(s) to restrict
                which part of the object is grasped (e.g. [0, 1, 2, 3] for one
                end of an elongated object, [4, 5, 6, 7] for the other end).
            verify: raise when the closed fingers are not touching the object.
                Turn it OFF only when you want the MOTION and not the grasp —
                nudging a part into place, hooking a handle you will pull by
                its joint, parking the hand on something before a servo
                approach. With verify=False you are responsible for checking
                what you achieved (check_grasp(), the object's pose, or the
                joint you are moving).
            search: when the grasp RoboTwin picked cannot be PLANNED, try the
                other annotated points (only the ones you named, when you named
                any) instead of giving up. Off by default because it costs one
                planned motion per point — and each failed candidate still
                EXECUTES its approach, so a failed search leaves the arm at the
                last approach pose and can nudge the object (the error says how
                many motions ran). A verification failure ("closed on air")
                never starts the search. On the Piper embodiment RoboTwin ranks
                the points by wrist ORIENTATION alone and never checks that the
                descend onto the one it picks has an IK solution (embodiments
                with strict_grasp_check, e.g. yam, do), so "unplannable" says
                nothing about the rest — find_grasp() answers that without
                moving.

        Example:
            grasp_object('box', arm='left', pre_grasp_dis=0.07)
        """
        self._env.plan_grasp_object(
            object_name,
            arm,
            pre_grasp_dis=pre_grasp_dis,
            grasp_dis=grasp_dis,
            contact_point_id=contact_point_ids,
            verify=verify,
            search=search,
        )

    def find_grasp(
        self,
        object_name: str,
        arm: str,
        contact_point_ids: list[int] | int | None = None,
        pre_grasp_dis: list | tuple = (0.10, 0.07, 0.05, 0.03),
        grasp_dis: list | tuple = (0.0, 0.02, 0.04),
        max_plans: int = 140,
    ) -> dict:
        """Which annotated grasp of this object can this arm actually REACH,
        and from how far back? Planning only — nothing moves, nothing in the
        simulation changes.

        Ask this BEFORE grasp_object when a grasp has failed or when the object
        sits near the edge of the arm's envelope. RoboTwin picks the contact
        point by wrist-ORIENTATION score alone and never verifies that the
        descend onto it is reachable, so grasp_object can raise "motion
        planning failed" while a different annotated point — or the same one
        approached from a different standoff — plans perfectly.

        Args:
            object_name: see list_objects().
            arm: 'left' or 'right'.
            contact_point_ids: id or list of ids to consider; None = every
                annotated point of the asset.
            pre_grasp_dis: approach standoffs to try, in meters (same meaning as
                grasp_object's argument).
            grasp_dis: how far SHORT of the contact point to stop, in meters.
                0.0 closes on the point itself. A non-zero value trades grip
                depth for reach — useful at the envelope edge, useless if it
                exceeds the object's own half-width (the fingers close on air).
            max_plans: bound on planner CALLS (not wall clock), so the same
                episode costs the same on a busy host; plans_used never exceeds
                it and budget_exhausted is True only when a call was skipped.
                One charged call is one get_grasp_pose (itself a sweep over
                RoboTwin's 10 yaw variants), one approach plan or one descend
                plan.  Cost per call: an IK failure ~0.1 s, a plan that has to
                fall back to RRT up to try_times x planning_time (9 x 5 s under
                the replay budget) — a blind sweep of a 16-point asset at four
                standoffs can be minutes: name the contact_point_ids you
                actually want (one grasp group, say) and one or two standoffs,
                and widen only if it comes back empty.  The sweep is breadth-
                first over standoffs, so a truncated budget still sees every
                point at the first standoff.  The answer is for the arm's
                CURRENT posture (planning starts from it): ask right before
                grasp_object, not at episode start.

        Returns:
            {'candidates': [{'contact_point_id', 'pre_grasp_dis', 'grasp_dis',
            'pre_pose', 'grasp_pose'}, ...] (every one plannable, best first),
            'best': the first of them or None, 'plans_used', 'budget_exhausted',
            'reasons': {'cid3@pre0.1': 'descend unplannable (IK Failed...)'}}.
            'best' prefers the smallest grasp_dis (the deepest, firmest grip)
            and then the largest approach standoff. Feed its three numbers
            to grasp_object — a candidate is a PLAN, not a guarantee (the
            executed approach can end in a different IK branch than the probe's
            seed), so iterate the candidates under try/except. An empty list
            with budget_exhausted False and no 'skipped:' reasons means no
            annotated grasp of this object is reachable for this arm — use the
            other arm, move the object, or fall back to a hand-built pinch
            (goto_pose + close).  Empty id lists and a grasp_dis that exceeds
            every pre_grasp_dis raise ValueError instead of returning [].
        """
        return self._env.grasp_candidates(
            object_name,
            arm,
            contact_point_ids=contact_point_ids,
            pre_grasp_dis=pre_grasp_dis,
            grasp_dis=grasp_dis,
            max_plans=max_plans,
        )

    def place_object(
        self,
        object_name: str,
        arm: str,
        target_position: np.ndarray,
        target_quaternion_wxyz: np.ndarray | None = None,
        pre_dis: float = 0.1,
        dis: float = 0.02,
        release: bool = True,
        functional_point_id: int | None = None,
        constrain: str = "auto",
        pre_dis_axis: str = "grasp",
    ) -> None:
        """Move the currently grasped object to a world-frame target pose,
        optionally releasing it. Raises an error if planning fails.

        Args:
            object_name: Name of the grasped object.
            arm: 'left' or 'right' (the arm holding the object).
            target_position: (3,) XYZ target in meters.
            target_quaternion_wxyz: Optional (4,) WXYZ target orientation.
            pre_dis: Approach offset before the target in meters.
            dis: Final offset at the target in meters.
            release: If True, open the gripper after reaching the target. Set
                False to hold the object there (e.g. mid-air handover).
            functional_point_id: Optional functional point on the grasped object
                to align with the target pose instead of the object center.
            constrain: 'free', 'align', or 'auto' placement orientation mode.
            pre_dis_axis: 'grasp' (approach along grasp axis) or 'fp'
                (approach along the target functional point axis).

        Example:
            place_object('box', arm='right', target_position=np.array([0.15, 0.17, 0.75]))
            place_object('box', arm='left', target_position=np.array([0, 0, 0.9]),
                         release=False, constrain='free')  # hold for handover


        VALID VALUES (anything else raises):
        - constrain: "auto" | "align" | "free"
        - pre_dis_axis: "grasp" | "fp" | a 3-vector like [0.0, 0.0, 1.0]
          ("grasp" = retreat along the grasp approach axis; "fp" = along the
          functional point's axis; a vector = along that world direction).
        """
        pos = np.asarray(target_position, dtype=np.float64).reshape(3)
        if target_quaternion_wxyz is None:
            target = pos.tolist()
        else:
            quat = np.asarray(target_quaternion_wxyz, dtype=np.float64).reshape(4)
            target = [*pos.tolist(), *quat.tolist()]
        self._env.plan_place_object(
            object_name,
            arm,
            target,
            functional_point_id=functional_point_id,
            pre_dis=pre_dis,
            dis=dis,
            is_open=release,
            constrain=constrain,
            pre_dis_axis=pre_dis_axis,
        )

    def get_object_functional_pose(self, object_name: str, point_id: int = 0) -> np.ndarray:
        """Get an annotated functional point pose on an object (world frame).

        Functional points mark task-relevant spots (e.g. the placement center
        of a target pad) — usually the very spot a task's scorer measures
        against, so read this before deciding where to put something.

        A functional point on an ARTICULATED object is anchored to one of its
        LINKS and therefore TRAVELS WITH THAT PART. Measured on the RoboTwin
        cabinet: the drawer's interior point sits at y = +0.11 with the drawer
        shut (outside both arms' envelope) and at y = -0.01 with it 67 % open.
        Opening a container does not merely give access, it moves the goal —
        so re-read this after every change of joint state, never cache it.

        Args:
            object_name: Name of the object.
            point_id: Index of the functional point.

        Returns:
            (7,) array [x, y, z, qw, qx, qy, qz].

        Example:
            fp = get_object_functional_pose('target_box', 1)
            place_object('box', arm='right', target_position=fp[:3],
                         target_quaternion_wxyz=fp[3:], functional_point_id=0,
                         pre_dis=0.05, dis=0.0, constrain='align', pre_dis_axis='fp')
        """
        return self._env.get_object_functional_pose(object_name, point_id)

    def back_to_origin_left(self) -> None:
        """Return the LEFT arm to its initial (home) configuration.

        This is a PLANNED motion and it fails often from a grasp pose — the
        home configuration is a singular posture and the path back from deep
        inside a container or a shelf frequently has no RRT solution (a failed
        RRT is also the single most expensive call here, 20-30 s). Retreat
        first with servo_delta/move_by_displacement, and treat a raise from
        this call as normal rather than as a lost episode."""
        self._env.plan_back_to_origin("left")

    def back_to_origin_right(self) -> None:
        """Return the RIGHT arm to its initial (home) configuration. Like its
        left twin it is a planned motion that often has no solution from a
        grasp pose — retreat with servo_delta first."""
        self._env.plan_back_to_origin("right")


__all__ = ["PiperPlannerPrivilegedApi"]


def register() -> None:
    """Register the API with cap-x's registry (idempotent)."""
    from capx.integrations.base_api import register_api

    register_api("PiperPlannerPrivilegedApi", PiperPlannerPrivilegedApi)


register()  # idempotent: make the API resolvable through capx.integrations.base_api.get_api
