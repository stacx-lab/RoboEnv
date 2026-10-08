"""Pure measured-state scoring for go2 d1 reach."""

import math

import numpy as np

from ...robots.go2_d1.task import Go2TaskState


class State(Go2TaskState):
    def evaluate(self, sample):
        spec = self.spec
        base = np.asarray(sample["base_position"])
        distance = float(
            np.linalg.norm(np.array(sample["tcp_position"]) - spec["reach_position"])
        )
        dot = abs(float(np.dot(sample["tcp_quaternion"], spec["reach_quaternion_xyzw"])))
        angle = 2 * math.acos(np.clip(dot, 0, 1))
        moved = float(np.linalg.norm(base[:2]))
        good = (
            distance <= spec["reach_position_tolerance_m"]
            and angle <= spec["reach_rotation_tolerance_rad"]
            and moved >= spec["minimum_reach_base_displacement_m"]
            and sample["base_speed"] < spec["maximum_base_speed_m_s"]
        )
        self.hold = self.hold + 1 if good else 0
        self.metrics = {
            "position_error_m": distance,
            "rotation_error_rad": angle,
            "base_displacement_m": moved,
        }
        if self.hold >= spec["hold_steps"] and self.failure is None:
            self.completed = True
