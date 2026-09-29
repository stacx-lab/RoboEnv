"""Practice task rules and native scenes, with lazy Isaac imports."""

import copy
import itertools
import math

import numpy as np

from .adapter import matrix_quat, quat_matrix

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
        self.placement_started = False
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
            if self.pickup_base is None:
                lifted = grasped and obj[2] >= 0.12
                self.lift_hold = self.lift_hold + 1 if lifted else 0
                if self.lift_hold >= 10:
                    self.pickup_base = base[:2].tolist()
            if self.pickup_base is not None:
                self.carried_distance = max(
                    self.carried_distance, float(np.linalg.norm(base[:2] - self.pickup_base))
                )
                release_zone = in_zone and obj[2] < 0.10 and self.carried_distance >= 0.6
                if release_zone:
                    self.placement_started = True
                if not self.placement_started:
                    self.contact_loss = 0 if grasped else self.contact_loss + 1
                    if bottom < 0.008 or self.contact_loss > 10:
                        self.failure = "object_dropped"
                elif not in_zone:
                    self.failure = "object_left_destination"
                settled = (
                    self.placement_started
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

    def snapshot(self):
        return copy.deepcopy(vars(self))


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
                        mass_props=sim.MassPropertiesCfg(mass=native.payload_mass_kg),
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
        self.state.spec["object_mass_kg"] = self.native.payload_mass_kg
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
