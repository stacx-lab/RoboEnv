"""Reach marker scene and Go2 task registration."""

import math

import numpy as np

from ...robots.go2_d1.controllers import quat_matrix
from ...robots.go2_d1.task import Go2Scene, Go2Task, validate_numeric_spec
from .scoring import State


class Scene(Go2Scene):
    def spectator_pose(self):
        from ...robots.go2_d1.spectator import workspace

        return workspace([2.7, -3.2, 2.1], [0.65, 0, 0.45],
                         [self.spec["reach_position"]])

    def build(self):
        import isaaclab.sim as sim

        spec = self.spec
        marker = sim.SphereCfg(
            radius=0.025, visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.9))
        )
        marker.func("/World/ReachTarget", marker, translation=tuple(spec["reach_position"]))
        # Red/green/blue axes make the target orientation visible without collision.
        r = quat_matrix(spec["reach_quaternion_xyzw"])
        for i, color in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
            size = [0.008] * 3
            size[i] = 0.12
            axis = sim.CuboidCfg(
                size=tuple(size), visual_material=sim.PreviewSurfaceCfg(diffuse_color=color)
            )
            axis.func(
                f"/World/ReachAxis{i}",
                axis,
                translation=tuple(np.array(spec["reach_position"]) + r[:, i] * 0.06),
                orientation=tuple(spec["reach_quaternion_xyzw"]),
            )


class Task(Go2Task):
    name = "go2-d1-reach"
    state_type = State
    scene_type = Scene
    vectors = {"reach_position": 3, "reach_quaternion_xyzw": 4}
    fields = {
        "reach_position",
        "reach_quaternion_xyzw",
        "reach_position_tolerance_m",
        "reach_rotation_tolerance_rad",
        "minimum_reach_base_displacement_m",
        "hold_steps",
        "maximum_base_speed_m_s",
    }

    @classmethod
    def validate(cls, spec):
        validate_numeric_spec(spec, cls.vectors)
        if not math.isclose(np.linalg.norm(spec["reach_quaternion_xyzw"]), 1, abs_tol=1e-6):
            raise ValueError("reach_quaternion_xyzw must be a unit quaternion")
