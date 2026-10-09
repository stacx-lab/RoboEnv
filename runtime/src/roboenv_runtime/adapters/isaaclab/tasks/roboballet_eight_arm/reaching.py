"""RoboBallet's largest released evaluation problem, with agent-owned planning.

Transition rules follow Google LLC's Apache-2.0 RoboBallet release (2025),
base_env.py and planning_env.py. The original geometry/collision implementation
is loaded unchanged from the verified source depot. No learned policy is loaded.
"""

import copy
import json
import random
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from roboenv_runtime.adapters.native_record import digest

ROBOT_ORDER = ["panda1", "panda2", "panda5", "panda6", "panda3", "panda4", "panda7", "panda8"]
DT = 0.1
MAX_STEPS = 420
MAX_VELOCITY = 2.175
MAX_ACCELERATION = 8.0
POSITION_TOLERANCE = 0.025
ANGLE_TOLERANCE = np.pi / 12
DWELL_STEPS = 5
HOME_CAPTURE = 0.25
HOME_TOLERANCE = 1e-3
INSTRUCTION = """Complete all 40 pose-reaching operations in the shared RoboBallet workcell
using eight Panda arms, then return every arm to its initial joint configuration.
Four arms are table-mounted and four are inverted above the same workpiece.
Each target specifies a world-frame tool-tip position and full orientation.
An operation is accepted within 25 mm and 15 degrees, then its arm is held for
0.5 simulation seconds. A dwelling arm cannot complete another operation.
Any physically capable arm may perform any operation, in any order. Minimize
the total simulation makespan while respecting the workpiece, table, tool and
robot collision geometry. All operations, dwell intervals and return motions
must finish within the original 42-second simulation horizon.
The workcell device provides the public target poses, obstacle geometry, robot
states, progress and home configurations. Its check operation evaluates candidate
joint configurations without executing them; it does not compute motion plans.
Device profiles provide tool-tip kinematics and calibrated world-frame mounts.
Joint commands are nonblocking position samples on the shared 0.1-second clock;
the environment converts each sample to a requested velocity and applies the
original 2.175 rad/s velocity and 8 rad/s^2 acceleration limits. A command that
would collide is rolled back for the affected arm. Inspect measured feedback.
After all targets are accepted, the original environment finishes each arm's
return once all its joints are within 0.25 rad of home. You choose assignments,
operation order, trajectories, coordination and any planning code you need.
"""


def achieved(pose, target):
    return bool(np.linalg.norm(pose[:3, 3] - target["position"]) < POSITION_TOLERANCE
                and (Rotation.from_matrix(pose[:3, :3]).inv()
                     * Rotation.from_matrix(target["rotation"])).magnitude()
                < ANGLE_TOLERANCE)


class ReachingTask:
    """One coupled 56-DoF workcell; no task allocator or motion planner."""

    def __init__(self, depot, scene_path, *, target_set_index, seed):
        if type(target_set_index) is not int or not 0 <= target_set_index < 10:
            raise ValueError("target_set_index must select one of the ten original sets")
        sys.path.insert(0, str(depot))
        from roboballet.environments.mujoco_simulator import MujocoWorld, MujocoWorldConfig

        self.scene = json.loads(scene_path.read_text())
        self.scene_hash = digest(self.scene)
        self.target_set_index = target_set_index
        self.targets = self.scene["target_sets"][target_set_index]
        if len(self.targets) != 40 or len(self.scene["obstacles"]) != 30:
            raise ValueError("Expected the full eight-arm, forty-target evaluation scene")
        self.world = MujocoWorld(MujocoWorldConfig(
            xml_path=str(depot / "roboballet/data/mujoco_world/8_pandas_world.xml")))
        self.world.RemoveObject("workpiece")
        objects = [self.world.AddBox(np.asarray(o["spans"]), np.array([.46, .43, .68]))
                   for o in self.scene["obstacles"]]
        for name, box in zip(objects, self.scene["obstacles"], strict=True):
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = box["rotation"], box["centroid"]
            self.world.SetWorldTransform(name, pose)
        self.robots = list(ROBOT_ORDER)
        self.home = self.positions()
        self.lower = np.array([self.world.GetLowerJointPositionLimits(n) for n in self.robots])
        self.upper = np.array([self.world.GetUpperJointPositionLimits(n) for n in self.robots])
        self.velocity = np.zeros((8, 7))
        self.dwell = np.zeros(8, dtype=int)
        self.solved_by = np.full(40, -1, dtype=int)
        self.rng = random.Random(seed)
        self.events = []
        self.collision_robot_steps = 0
        self.unsafe_proposal_steps = 0
        self.unsafe_proposal_robot_steps = 0
        self.measured_velocity = np.zeros((8, 7))
        self.telemetry = []
        self.last_blocked = []
        self.motion_histogram = [0] * 9
        self.work_histogram = [0] * 9
        self.frame = 0
        if self.world.RobotsInCollision():
            raise ValueError("Original initial workcell is in collision")
        self.probe = copy.copy(self.world)

    def validate_reachability(self):
        """Reject a provably impossible task before rendering or agent execution."""
        from .feasibility import unreachable_targets

        excluded = unreachable_targets(
            self.world, self.robots, self.targets,
            position_tolerance=POSITION_TOLERANCE, angle_tolerance=ANGLE_TOLERANCE)
        if excluded:
            targets = ", ".join(str(item["target"]) for item in excluded)
            raise ValueError(
                f"Invalid RoboBallet target set {self.target_set_index}: targets {targets} "
                "are geometrically unreachable by every arm within the native pose "
                "tolerances. This is a task configuration error, not an agent failure.")

    def positions(self):
        return np.array([self.world.GetJointValues(n) for n in self.robots])

    def poses(self):
        return [self.world.GetRobotTcpTransform(n) for n in self.robots]

    @property
    def completed(self):
        return bool((self.solved_by >= 0).all() and not self.dwell.any()
                    and (np.abs(self.positions() - self.home) < HOME_TOLERANCE).all())

    def step(self, references):
        old = self.positions()
        requested = np.clip((np.asarray(references) - old) / DT, -MAX_VELOCITY, MAX_VELOCITY)
        dwelling = self.dwell > 0
        requested[dwelling] = 0
        self.dwell[dwelling] -= 1
        if (self.solved_by >= 0).all():
            near_home = (np.abs(self.home - old) < HOME_CAPTURE).all(axis=1)
            requested[near_home] = (self.home - old)[near_home] / DT
        # Match original acceleration clamping, including the braking interval
        # at the start of a dwell and the native final home capture.
        velocity = np.clip(requested, self.velocity - MAX_ACCELERATION * DT,
                           self.velocity + MAX_ACCELERATION * DT)
        proposed = np.clip(old + velocity * DT, self.lower, self.upper)
        for name, q in zip(self.robots, proposed, strict=True):
            self.world.SetJointValues(name, q)
        collisions = self.world.RobotsInCollision()
        unsafe_arms = [i + 1 for i, name in enumerate(self.robots) if name in collisions]
        unsafe_pairs = self.collision_pairs(self.world) if collisions else []
        self.unsafe_proposal_steps += bool(collisions)
        self.unsafe_proposal_robot_steps += len(collisions)
        reset = set()
        while collisions:
            candidates = sorted(collisions - reset)
            if not candidates:
                raise RuntimeError("Collision rollback could not restore a valid workcell")
            name = self.rng.choice(candidates)
            i = self.robots.index(name)
            self.world.SetJointValues(name, old[i])
            velocity[i] = 0
            reset.add(name)
            collisions = self.world.RobotsInCollision()
        self.velocity = velocity
        self.frame += 1
        self.last_blocked = [i for i, name in enumerate(self.robots) if name in reset]
        self.collision_robot_steps += len(reset)
        if reset:
            self.events.append({"frame": self.frame, "kind": "collision_rollback",
                                "arms": [i + 1 for i in self.last_blocked]})
        poses = self.poses()
        for t, target in enumerate(self.targets):
            if self.solved_by[t] >= 0:
                continue
            for i, pose in enumerate(poses):
                if self.dwell[i] == 0 and achieved(pose, target):
                    self.solved_by[t] = i
                    self.dwell[i] = DWELL_STEPS
                    self.events.append({"frame": self.frame, "kind": "target_accepted",
                                        "target": t, "arm": i + 1})
                    break
        positions = self.positions()
        moving = (np.abs(positions - old) > 1e-7).any(axis=1)
        self.motion_histogram[int(moving.sum())] += 1
        self.work_histogram[int((moving | dwelling).sum())] += 1
        measured_velocity = (positions - old) / DT
        # This describes executed motion, including abrupt stops introduced by
        # collision rollback. It never changes controls or the native verdict.
        self.telemetry.append({
            "frame": self.frame,
            "moving_arms": (np.flatnonzero(moving) + 1).tolist(),
            "dwelling_arms": (np.flatnonzero(dwelling) + 1).tolist(),
            "targets_completed": int((self.solved_by >= 0).sum()),
            "unsafe_proposal_arms": unsafe_arms,
            "unsafe_proposal_pairs": unsafe_pairs,
            "rollback_arms": [i + 1 for i in self.last_blocked],
            "max_measured_speed_rad_s": float(np.abs(measured_velocity).max()),
            "max_measured_acceleration_rad_s2": float(
                np.abs(measured_velocity - self.measured_velocity).max() / DT),
            "native_state_sha256": digest({
                "positions": positions.tolist(), "velocities": self.velocity.tolist(),
                "solved_by": self.solved_by.tolist(), "dwell_steps": self.dwell.tolist()}),
        })
        self.measured_velocity = measured_velocity

    @staticmethod
    def collision_pairs(world):
        """Read already-computed contacts without changing native world state."""
        pairs = set()
        for contact in world._data.contact[:world._data.ncon]:
            if contact.dist <= 0:
                geoms = [world._model.geom(int(g)) for g in contact.geom]
                if any(world._robot_from_body_id(int(g.bodyid[0])) for g in geoms):
                    pairs.add(tuple(sorted(g.name for g in geoms)))
        return [list(pair) for pair in sorted(pairs)]

    def check(self, configurations):
        values = np.asarray(configurations, dtype=float)
        if (values.ndim != 3 or values.shape[1:] != (8, 7) or not 1 <= len(values) <= 128
                or not np.isfinite(values).all()):
            raise ValueError("Supply 1..128 candidate configurations, each shaped [8, 7]")
        results = []
        for candidate in values:
            if ((candidate < self.lower) | (candidate > self.upper)).any():
                results.append({"valid": False, "reason": "joint_limits", "collision_pairs": []})
                continue
            for name, q in zip(self.robots, candidate, strict=True):
                self.probe.SetJointValues(name, q)
            blocked = self.probe.RobotsInCollision()
            results.append({
                "valid": not blocked, "collision_pairs": self.collision_pairs(self.probe),
                "blocked_arms": [i + 1 for i, n in enumerate(self.robots) if n in blocked]})
        return {"configurations": results, "timestamp": self.frame * DT}

    def public_state(self):
        return {"timestamp": self.frame * DT, "frame": "world",
                "targets": [{"id": i, **t, "done": bool(self.solved_by[i] >= 0),
                             "completed_by": None if self.solved_by[i] < 0 else f"arm{self.solved_by[i] + 1}"}
                            for i, t in enumerate(self.targets)],
                "obstacles": self.scene["obstacles"],
                "robots": {f"arm{i + 1}": {"source_name": n, "positions": self.positions()[i].tolist(),
                            "velocities": self.velocity[i].tolist(), "home": self.home[i].tolist(),
                            "tool_tip_world": self.poses()[i].tolist(),
                            "base_world": self.world.GetRobotBaseTransform(n).tolist(),
                            "dwell_remaining_s": int(self.dwell[i]) * DT,
                            "collision_blocked_last_step": i in self.last_blocked}
                           for i, n in enumerate(self.robots)}}
