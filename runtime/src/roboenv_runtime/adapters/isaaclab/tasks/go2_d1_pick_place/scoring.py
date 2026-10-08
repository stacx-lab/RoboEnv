"""Pure measured-state scoring for go2 d1 pick place."""

import numpy as np

from ...robots.go2_d1.task import Go2TaskState


class State(Go2TaskState):
    def evaluate(self, sample):
        spec = self.spec
        base = np.asarray(sample["base_position"])
        obj = np.asarray(sample["object_position"])
        contact = np.asarray(sample["finger_forces"]) > spec["minimum_contact_force_n"]
        grasped = bool(
            contact.all()
            and sample["object_tcp_distance"] < spec["maximum_object_tcp_distance_m"]
        )
        # Corner projections account for rotated blocks, not just their center.
        corners = np.asarray(sample["object_corners"])
        in_zone = bool(
            np.max(np.abs(corners[:, :2] - np.array(spec["destination"][:2])))
            <= spec["destination_half_size_m"]
        )
        bottom = float(corners[:, 2].min())
        if self.pickup_base is None:
            lifted = grasped and obj[2] >= spec["lift_center_height_m"]
            self.lift_hold = self.lift_hold + 1 if lifted else 0
            if self.lift_hold >= spec["lift_hold_steps"]:
                self.pickup_base = base[:2].tolist()
        if self.pickup_base is not None:
            self.carried_distance = max(
                self.carried_distance, float(np.linalg.norm(base[:2] - self.pickup_base))
            )
            release_zone = (
                in_zone
                and obj[2] < spec["release_center_height_m"]
                and self.carried_distance >= spec["minimum_carry_base_displacement_m"]
            )
            if release_zone:
                self.placement_started = True
            if not self.placement_started:
                self.contact_loss = 0 if grasped else self.contact_loss + 1
                if (
                    bottom < spec["drop_bottom_height_m"]
                    or self.contact_loss > spec["contact_loss_grace_steps"]
                ):
                    self.failure = "object_dropped"
            elif not in_zone:
                self.failure = "object_left_destination"
            settled = (
                self.placement_started
                and in_zone
                and spec["settled_bottom_min_m"] <= bottom <= spec["settled_bottom_max_m"]
                and sample["object_speed"] < spec["maximum_object_speed_m_s"]
                and sample["object_angular_speed"] < spec["maximum_object_angular_speed_rad_s"]
                and sample["gripper_opening"] > spec["minimum_gripper_opening"]
                and not contact.any()
                and sample["base_speed"] < spec["maximum_base_speed_m_s"]
            )
            self.hold = self.hold + 1 if settled else 0
        self.metrics = {
            "bilateral_contact": grasped,
            "object_height_m": float(obj[2]),
            "inside_destination": in_zone,
            "carried_distance_m": self.carried_distance,
        }
        if self.hold >= spec["hold_steps"] and self.failure is None:
            self.completed = True
