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


def catalog_definition(name):
    """Defaults for direct Python callers; launch/export pass the resolved YAML spec."""
    from roboenv_runtime.paths import repo_root
    import yaml

    if name not in TASKS:
        raise ValueError("Unknown Go2 task")
    path = repo_root() / "catalog/tasks/isaaclab" / name / "task.yaml"
    return yaml.safe_load(path.read_text())["backend_kwargs"]["task_spec"]


def catalog_tcp():
    from roboenv_runtime.paths import repo_root
    import yaml

    path = repo_root() / "catalog/robots/isaaclab/isaaclab-go2-d1/devices/gripper/profile.yaml"
    return gripper_tcp(yaml.safe_load(path.read_text()))


def gripper_tcp(profile):
    from roboenv_runtime.adapters.urdf_geometry import matrix

    mount = profile["extrinsics"]["mount"]
    tcp = profile["intrinsics"]["frames"]["tcp"]
    return (matrix(mount["translation_m"] + mount["rotation_xyzw"])
            @ matrix(tcp["translation_m"] + tcp["rotation_xyzw"]))[:3, 3].tolist()


REACH_FIELDS = {
    "reach_position",
    "reach_quaternion_xyzw",
    "reach_position_tolerance_m",
    "reach_rotation_tolerance_rad",
    "minimum_reach_base_displacement_m",
    "hold_steps",
    "maximum_base_speed_m_s",
}
PICK_FIELDS = {
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


def definition(name, task_spec=None, tcp_in_flange_m=None):
    """Validate all configuration before native startup; retain plain values for replay."""
    if name not in TASKS:
        raise ValueError("Unknown Go2 task")
    spec = copy.deepcopy(catalog_definition(name) if task_spec is None else task_spec)
    fields = REACH_FIELDS if name == REACH else PICK_FIELDS if name == PICK_PLACE else set()
    if not isinstance(spec, dict) or set(spec) != fields:
        raise ValueError(f"Invalid {name} task_spec fields; expected {sorted(fields)}")
    spec["tcp_in_flange_m"] = copy.deepcopy(
        catalog_tcp() if tcp_in_flange_m is None else tcp_in_flange_m
    )
    vectors = {
        "tcp_in_flange_m": 3,
        "reach_position": 3,
        "reach_quaternion_xyzw": 4,
        "object_position": 3,
        "object_size": 3,
        "destination": 3,
    }
    for key, value in spec.items():
        values = value if key in vectors else [value]
        if key in vectors and (not isinstance(value, list) or len(value) != vectors[key]):
            raise ValueError(f"{key} requires {vectors[key]} finite numbers")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numbers")
        if key.endswith("_steps") and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key not in vectors and key != "settled_bottom_min_m" and value <= 0:
            raise ValueError(f"{key} must be positive")
    if name == REACH and not math.isclose(
        np.linalg.norm(spec["reach_quaternion_xyzw"]), 1, abs_tol=1e-6
    ):
        raise ValueError("reach_quaternion_xyzw must be a unit quaternion")
    if name == PICK_PLACE:
        if min(spec["object_size"]) <= 0 or spec["object_mass_kg"] > 5:
            raise ValueError("Object size must be positive and mass at most 5 kg")
        if spec["settled_bottom_min_m"] > spec["settled_bottom_max_m"]:
            raise ValueError("Settled bottom bounds are reversed")
        if spec["minimum_gripper_opening"] >= 1:
            raise ValueError("minimum_gripper_opening must be less than 1")
        if spec["destination"][2] != 0:
            raise ValueError("Destination must lie on the ground plane")
    return {"name": name, "version": 1, **spec}


class TaskState:
    """Evaluates measured physics only; never writes a robot or object pose."""

    def __init__(self, name, task_spec=None, tcp_in_flange_m=None):
        self.spec = definition(name, task_spec, tcp_in_flange_m)
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
        spec = self.spec
        base = np.asarray(sample["base_position"])
        if self.spec["name"] == REACH:
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
        else:
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

    def snapshot(self):
        return copy.deepcopy(vars(self))


class TaskScene:
    def __init__(self, native, name):
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self.native, self.name = native, name
        self.spec = copy.deepcopy(native.task_spec)
        spec = self.spec
        self.object = None
        self.contacts = []
        if name == REACH:
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
        self.state = TaskState(
            self.name,
            {k: v for k, v in self.spec.items() if k not in {"name", "version", "tcp_in_flange_m"}},
            self.spec["tcp_in_flange_m"],
        )
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
        spec = self.spec
        root = n.root_pose()
        transform = np.eye(4)
        transform[:3, :3] = quat_matrix(root[3:])
        transform[:3, 3] = root[:3]
        flange = transform @ n.model.forward(n.joints())["d1_Link6"]
        tcp = flange[:3, 3] + flange[:3, :3] @ np.asarray(spec["tcp_in_flange_m"])
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
