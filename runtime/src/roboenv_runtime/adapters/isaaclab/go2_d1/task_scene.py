"""Task-only Isaac assets, physical contact sensing and private measurements."""

import itertools

import numpy as np

from .kinematics import matrix_quat, quat_matrix
from .tasks import (
    CONTROL,
    DESTINATION,
    OBJECT_POSITION,
    OBJECT_SIZE,
    PICK_PLACE,
    REACH,
    REACH_POSITION,
    REACH_QUAT,
    TCP,
    TaskState,
)


class TaskScene:
    def __init__(self, native, name):
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self.native, self.name = native, name
        self.object = None
        self.contacts = []
        if name == REACH:
            marker = sim.SphereCfg(
                radius=0.025, visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.9))
            )
            marker.func("/World/ReachTarget", marker, translation=tuple(REACH_POSITION))
            # Red/green/blue axes make the target orientation visible without collision.
            r = quat_matrix(REACH_QUAT)
            for i, color in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
                size = [0.008] * 3
                size[i] = 0.12
                axis = sim.CuboidCfg(
                    size=tuple(size), visual_material=sim.PreviewSurfaceCfg(diffuse_color=color)
                )
                axis.func(
                    f"/World/ReachAxis{i}",
                    axis,
                    translation=tuple(np.array(REACH_POSITION) + r[:, i] * 0.06),
                    orientation=tuple(REACH_QUAT),
                )
        elif name == PICK_PLACE:
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
                        size=tuple(OBJECT_SIZE),
                        rigid_props=sim.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=8,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim.CollisionPropertiesCfg(
                            contact_offset=0.001, rest_offset=0.0
                        ),
                        mass_props=sim.MassPropertiesCfg(mass=0.08),
                        physics_material=sim.RigidBodyMaterialCfg(
                            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                        ),
                        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.03)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(OBJECT_POSITION)),
                )
            )
            zone = sim.CuboidCfg(
                size=(0.30, 0.30, 0.002),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.2)),
            )
            zone.func(
                "/World/Destination", zone, translation=(DESTINATION[0], DESTINATION[1], 0.001)
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
        self.state = TaskState(self.name)
        self.forces = [0.0, 0.0]
        if self.object is not None:
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
        if self.object is not None:
            self.object.update(self.native.dt)
            for sensor in self.contacts:
                sensor.update(self.native.dt, force_recompute=True)
            self.forces = [
                float(np.linalg.norm(self.native.array(s.data.force_matrix_w)))
                for s in self.contacts
            ]

    def update(self, fallen):
        if self.name == CONTROL:
            self.state.update({"fallen": fallen})
            return
        n = self.native
        root = n.root_pose()
        transform = np.eye(4)
        transform[:3, :3] = quat_matrix(root[3:])
        transform[:3, 3] = root[:3]
        flange = transform @ n.model.forward(n.joints())["d1_Link6"]
        tcp = flange[:3, 3] + flange[:3, :3] @ TCP
        sample = {
            "fallen": fallen,
            "base_position": root[:3],
            "base_speed": float(np.linalg.norm(n.array(n.robot.data.root_lin_vel_b))),
            "tcp_position": tcp,
            "tcp_quaternion": matrix_quat(flange[:3, :3]),
        }
        if self.object is not None:
            obj = n.array(self.object.data.root_link_pose_w)
            corners = (
                np.array(list(itertools.product((-1, 1), repeat=3))) * np.array(OBJECT_SIZE) / 2
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
        self.state.update(sample)

    def snapshot(self):
        result = {"task": self.state.snapshot()}
        if self.object is not None:
            n = self.native
            result["object"] = {
                "pose": n.array(self.object.data.root_link_pose_w).tolist(),
                "linear_velocity": n.array(self.object.data.root_lin_vel_w).tolist(),
                "angular_velocity": n.array(self.object.data.root_ang_vel_w).tolist(),
                "finger_forces": self.forces,
            }
        return result
