"""Practice task definitions and owner-only state machines, independent of Isaac."""

import copy
import math

import numpy as np

CONTROL = "go2-d1-control"
REACH = "go2-d1-reach"
PICK_PLACE = "go2-d1-pick-place"
TASKS = (CONTROL, REACH, PICK_PLACE)
TCP = np.array([0.0, 0.0, 0.105])
REACH_POSITION = [1.4, 0.0, 0.65]
REACH_QUAT = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
OBJECT_POSITION = [1.25, 0.0, 0.03]
OBJECT_SIZE = [0.04, 0.04, 0.06]
DESTINATION = [2.5, 0.0, 0.0]
INSTRUCTIONS = {
    CONTROL: "Practice Go2 base velocity/posture control and independent D1 arm manipulation.",
    REACH: (
        "Move the base and use independent arm IK to reach the cyan gripper TCP target "
        "at world position [1.4, 0, 0.65] m, orientation xyzw [0, 0.70710678, 0, 0.70710678]. "
        "TCP is [0, 0, 0.105] m from the arm flange. Move the base at least 0.5 m from start, "
        "then hold within 0.025 m and 0.10 rad for 1 s with base speed below 0.08 m/s."
    ),
    PICK_PLACE: (
        "Walk to the orange 4 x 4 x 6 cm, 80 g block on the ground near [1.25, 0, 0] m. "
        "Grasp it with both fingers, lift its center above 0.12 m, and carry it while walking "
        "at least 0.6 m to the green ground destination centered at [2.5, 0, 0] m. "
        "Do not drop it during transport. Place the entire block inside the 0.30 x 0.30 m "
        "destination, open the gripper, and let the block settle for 1 s. "
        "Use the RGB-D cameras and robot feedback; all coordinates are in the world frame."
    ),
}


def definition(name):
    if name not in TASKS:
        raise ValueError("Unknown Go2 task")
    return {
        "name": name,
        "version": 1,
        "tcp_in_flange_m": TCP.tolist(),
        "reach_position": REACH_POSITION,
        "reach_quaternion_xyzw": REACH_QUAT,
        "object_position": OBJECT_POSITION,
        "object_size": OBJECT_SIZE,
        "object_mass_kg": 0.08,
        "destination": DESTINATION,
        "destination_half_size_m": 0.15,
        "reach_position_tolerance_m": 0.025,
        "reach_rotation_tolerance_rad": 0.10,
        "minimum_reach_base_displacement_m": 0.5,
        "minimum_carry_base_displacement_m": 0.6,
        "lift_center_height_m": 0.12,
        "hold_steps": 50,
        "contact_loss_grace_steps": 10,
    }


class TaskState:
    """Evaluates measured physics only; never writes a robot or object pose."""

    def __init__(self, name):
        self.spec = copy.deepcopy(definition(name))
        self.completed = False
        self.failure = None
        self.stage = "practice" if name == CONTROL else "reach" if name == REACH else "pickup"
        self.hold = self.contact_loss = self.lift_hold = 0
        self.pickup_base = None
        self.carried_distance = 0.0
        self.metrics = {}

    @property
    def done(self):
        return self.completed or self.failure is not None

    def update(self, sample):
        if self.done:
            return
        if sample["fallen"]:
            self.failure = "robot_fallen"
            return
        if self.spec["name"] == CONTROL:
            return
        base = np.asarray(sample["base_position"])
        if self.spec["name"] == REACH:
            distance = float(np.linalg.norm(np.array(sample["tcp_position"]) - REACH_POSITION))
            dot = abs(float(np.dot(sample["tcp_quaternion"], REACH_QUAT)))
            angle = 2 * math.acos(np.clip(dot, 0, 1))
            moved = float(np.linalg.norm(base[:2]))
            good = (
                distance <= 0.025 and angle <= 0.10 and moved >= 0.5 and sample["base_speed"] < 0.08
            )
            self.hold = self.hold + 1 if good else 0
            self.metrics = {
                "position_error_m": distance,
                "rotation_error_rad": angle,
                "base_displacement_m": moved,
            }
        else:
            obj = np.asarray(sample["object_position"])
            contact = np.asarray(sample["finger_forces"]) > 0.01
            grasped = bool(contact.all() and sample["object_tcp_distance"] < 0.07)
            # Corner projections account for rotated blocks, not just their center.
            corners = np.asarray(sample["object_corners"])
            in_zone = bool(np.max(np.abs(corners[:, :2] - np.array(DESTINATION[:2]))) <= 0.15)
            bottom = float(corners[:, 2].min())
            if self.stage == "pickup":
                lifted = grasped and obj[2] >= 0.12
                self.lift_hold = self.lift_hold + 1 if lifted else 0
                if self.lift_hold >= 10:
                    self.stage = "carry"
                    self.pickup_base = base[:2].tolist()
            if self.stage in ("carry", "place"):
                self.carried_distance = max(
                    self.carried_distance, float(np.linalg.norm(base[:2] - self.pickup_base))
                )
                release_zone = in_zone and obj[2] < 0.10 and self.carried_distance >= 0.6
                if release_zone:
                    self.stage = "place"
                if self.stage == "carry":
                    self.contact_loss = 0 if grasped else self.contact_loss + 1
                    if bottom < 0.008 or self.contact_loss > 10:
                        self.failure = "object_dropped"
                elif not in_zone:
                    self.failure = "object_left_destination"
                settled = (
                    self.stage == "place"
                    and in_zone
                    and -0.005 <= bottom <= 0.012
                    and sample["object_speed"] < 0.03
                    and sample["object_angular_speed"] < 0.15
                    and sample["gripper_opening"] > 0.8
                    and not contact.any()
                    and sample["base_speed"] < 0.08
                )
                self.hold = self.hold + 1 if settled else 0
            self.metrics = {
                "bilateral_contact": grasped,
                "object_height_m": float(obj[2]),
                "inside_destination": in_zone,
                "carried_distance_m": self.carried_distance,
            }
        if self.hold >= 50 and self.failure is None:
            self.completed = True
            self.stage = "success"

    def snapshot(self):
        return copy.deepcopy(vars(self))
