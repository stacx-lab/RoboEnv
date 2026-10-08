"""Physical payload, destination and contact sensors for pick/carry/place."""

import itertools

import numpy as np

from ...robots.go2_d1.controllers import quat_matrix
from ...robots.go2_d1.task import Go2Scene, Go2Task, validate_numeric_spec
from .scoring import State


class Scene(Go2Scene):
    def spectator_pose(self):
        from ...robots.go2_d1.spectator import workspace

        return workspace([3.6, -4, 2.6], [1.2, 0, 0.45],
                         [self.spec["object_position"], self.spec["destination"]])

    def build(self):
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        native, spec = self.native, self.spec
        self.contacts = []
        sim.activate_contact_sensors("/World/Robot")
        from pxr import UsdPhysics

        bodies = {
            prim.GetName(): str(prim.GetPath())
            for prim in native.sim.stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        self.object = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Payload",
                spawn=sim.CuboidCfg(
                    size=tuple(spec["object_size"]),
                    rigid_props=sim.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=8,
                        max_depenetration_velocity=1.0,
                    ),
                    collision_props=sim.CollisionPropertiesCfg(
                        contact_offset=0.001, rest_offset=0.0
                    ),
                    mass_props=sim.MassPropertiesCfg(mass=native.payload_mass_kg),
                    physics_material=sim.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                    ),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.03)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(spec["object_position"])),
            )
        )
        zone = sim.CuboidCfg(
            size=(
                2 * spec["destination_half_size_m"],
                2 * spec["destination_half_size_m"],
                0.002,
            ),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
        )
        zone.func(
            "/World/Destination",
            zone,
            translation=(spec["destination"][0], spec["destination"][1], 0.001),
        )
        for suffix in ("1", "2"):
            sim.activate_contact_sensors(bodies["d1_Link7_" + suffix])
            self.contacts.append(
                ContactSensor(
                    ContactSensorCfg(
                        prim_path=bodies["d1_Link7_" + suffix],
                        filter_prim_paths_expr=["/World/Payload"],
                        update_period=0.0,
                    )
                )
            )

    def reset(self):
        super().reset()
        self.forces = [0.0, 0.0]
        obj = self.object
        obj.write_root_pose_to_sim_index(root_pose=obj.data.default_root_pose.torch.clone())
        obj.write_root_velocity_to_sim_index(
            root_velocity=obj.data.default_root_vel.torch.clone()
        )
        obj.reset()
        obj.update(self.native.dt)
        for sensor in self.contacts:
            sensor.reset()

    def physics_update(self):
        self.object.update(self.native.dt)
        for sensor in self.contacts:
            sensor.update(self.native.dt, force_recompute=True)
        self.forces = [
            float(np.linalg.norm(self.native.array(s.data.force_matrix_w)))
            for s in self.contacts
        ]

    def sample(self, fallen):
        sample = super().sample(fallen)
        n, spec = self.native, self.spec
        tcp = sample["tcp_position"]
        obj = n.array(self.object.data.root_link_pose_w)
        corners = (
            np.array(list(itertools.product((-1, 1), repeat=3)))
            * np.array(spec["object_size"])
            / 2
        )
        corners = corners @ quat_matrix(obj[3:]).T + obj[:3]
        sample.update(
            object_position=obj[:3],
            object_corners=corners,
            object_speed=float(np.linalg.norm(n.array(self.object.data.root_lin_vel_w))),
            object_angular_speed=float(
                np.linalg.norm(n.array(self.object.data.root_ang_vel_w))
            ),
            finger_forces=self.forces,
            object_tcp_distance=float(np.linalg.norm(tcp - obj[:3])),
            gripper_opening=(n.joints()["d1_Joint7_1"] - n.joints()["d1_Joint7_2"]) / 0.06,
        )
        return sample

    def snapshot(self):
        result = super().snapshot()
        n = self.native
        result["object"] = {
            "pose": n.array(self.object.data.root_link_pose_w).tolist(),
            "linear_velocity": n.array(self.object.data.root_lin_vel_w).tolist(),
            "angular_velocity": n.array(self.object.data.root_ang_vel_w).tolist(),
            "finger_forces": self.forces,
        }
        return result


class Task(Go2Task):
    name = "go2-d1-pick-place"
    state_type = State
    scene_type = Scene
    contact_sensors = True
    payload_mass_field = "object_mass_kg"
    vectors = {"object_position": 3, "object_size": 3, "destination": 3}
    nonpositive_fields = {"settled_bottom_min_m"}
    fields = {
        "object_position",
        "object_size",
        "object_mass_kg",
        "destination",
        "destination_half_size_m",
        "minimum_carry_base_displacement_m",
        "lift_center_height_m",
        "lift_hold_steps",
        "contact_loss_grace_steps",
        "minimum_contact_force_n",
        "maximum_object_tcp_distance_m",
        "release_center_height_m",
        "drop_bottom_height_m",
        "settled_bottom_min_m",
        "settled_bottom_max_m",
        "maximum_object_speed_m_s",
        "maximum_object_angular_speed_rad_s",
        "minimum_gripper_opening",
        "hold_steps",
        "maximum_base_speed_m_s",
    }

    @classmethod
    def validate(cls, spec):
        validate_numeric_spec(spec, cls.vectors, cls.nonpositive_fields)
        if min(spec["object_size"]) <= 0 or spec["object_mass_kg"] > 5:
            raise ValueError("Object size must be positive and mass at most 5 kg")
        if spec["settled_bottom_min_m"] > spec["settled_bottom_max_m"]:
            raise ValueError("Settled bottom bounds are reversed")
        if spec["minimum_gripper_opening"] >= 1:
            raise ValueError("minimum_gripper_opening must be less than 1")
        if spec["destination"][2] != 0:
            raise ValueError("Destination must lie on the ground plane")
