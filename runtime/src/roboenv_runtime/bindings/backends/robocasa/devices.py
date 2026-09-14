"""RoboCasa devices bind directly to native controller parts, never task macros."""

from dataclasses import replace
from pathlib import Path
import uuid
import time

import numpy as np

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.adapters.robocasa.geometry import chain, pose, vector
from roboenv_runtime.adapters.robocasa.depth import write_depth

from .base import intrinsics as base_intrinsics, reading as base_reading
from .camera import intrinsics as camera_intrinsics
from .gripper import intrinsics as gripper_intrinsics

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
PARTS = {"arm": "right", "base": "base", "torso": "torso", "gripper": "right_gripper"}
TYPES = {
    "arm": "manipulator",
    "base": "base",
    "torso": "manipulator",
    "gripper": "end_effector",
    "camera": "rgb",
    "world": "world",
}


def array(size):
    return {
        "type": "array",
        "minItems": size,
        "maxItems": size,
        "items": {"type": "number"},
    }


def arguments(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def bind(spec, root, context):
    world = context.native._env
    robot, sim = world._robot, world.sim
    model, data = sim.model._model, sim.data._data
    arm_joints = robot.part_controllers["right"].joint_index
    arm_base = int(model.body_parentid[model.jnt_bodyid[arm_joints[0]]])
    entity = spec.entity
    if entity not in TYPES:
        raise ValueError(f"Unknown RoboCasa device entity: {entity}")
    definition = spec.definition(root)
    valid_types = {"rgb", "rgbd"} if entity == "camera" else {TYPES[entity]}
    if definition.type not in valid_types:
        raise ValueError(f"{entity} requires type {TYPES[entity]}")
    if set(spec.config) - ({"camera"} if entity == "camera" else set()):
        raise ValueError("Unknown device configuration")
    actions = {}
    intrinsics = dict(definition.intrinsics)
    extrinsics = dict(definition.extrinsics)

    def register(name, function, schema=EMPTY, *, mutation=False):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "binding." + name, args, lambda: function(**args), mutating=mutation
            ),
            schema,
            output_schema={
                "type": "object",
                "required": {
                    "load": ["loaded"],
                    "unload": ["unloaded"],
                    "status": ["loaded"],
                    "write": ["accepted"],
                    "stop": (["stopped", "behavior", "affected_devices"]
                             if entity in PARTS else ["stopped"]),
                    "read": (
                        ["file", "width", "height", "sample_time_s"]
                        + (["depth_file", "depth_width", "depth_height", "depth_sample_time_s"] if definition.type == "rgbd" else [])
                        if entity == "camera"
                        else (["pose" if entity == "base" else "positions", "velocities", "faults", "sample_time_s"]
                              + (["efforts", "flange_pose"] if entity in {"arm", "torso"} else []))
                    ),
                }[name],
            },
        )

    if entity != "world":
        register("load", lambda: {"loaded": True})
        register("unload", lambda: {"unloaded": True})
        register("status", lambda: {"loaded": True})

    if entity in PARTS:
        part = PARTS[entity]
        controller = robot.part_controllers[part]
        joints = np.asarray(controller.joint_index, dtype=int)
        # A gripper command has one value but two measured finger joints.
        split = robot.composite_controller._action_split_indexes[part]
        size = split[1] - split[0]
        if entity == "base":
            intrinsics = base_intrinsics(world, (context.devices or {}).get("world", "world"))
            intrinsics["stop"]["affected_devices"] = [definition.name]
        if entity == "gripper":
            intrinsics = gripper_intrinsics(robot, sim, joints)
            intrinsics["stop"]["affected_devices"] = [definition.name]
        definition = replace(
            definition,
            control_mode={
                "arm": "joint_position",
                "base": "normalized_velocity",
                "torso": "joint_position",
                "gripper": "gripper_velocity",
            }[entity],
        )
        if entity == "arm":
            base = arm_base
            body = sim.model.body_name2id(robot.robot_model.eef_name["right"])
        elif entity == "torso":
            base = sim.model.body_name2id(robot.robot_model.base.root_body)
            body = arm_base
        else:
            base, body = 0, sim.model.body_name2id(robot.robot_model.base.root_body)
        if entity in {"arm", "torso"}:
            from .manipulator import intrinsics as manipulator_intrinsics

            home = robot.init_qpos if entity == "arm" else robot.init_torso_qpos
            if home is None:
                home = robot.robot_model.init_torso_qpos
            intrinsics = manipulator_intrinsics(sim, joints, chain(model, joints, body, base)["chain"], home, definition.name)

        if entity in {"arm", "torso"}:
            limits = [joint["limits"] for joint in intrinsics["kinematics"]["model"]["joints"]]
        elif entity == "base":
            limits = [component["bounds"] for component in intrinsics["control"]["modes"]["drive"]["velocity"]["components"]]
        else:
            limits = [{"min": -1, "max": 1}]
        if size != len(limits):
            raise ValueError("Unexpected native controller action space")
        lower = [limit["min"] if limit else -np.inf for limit in limits]
        upper = [limit["max"] if limit else np.inf for limit in limits]

        def read():
            if entity == "base":
                return base_reading(model, data, body)
            result = {
                "positions": data.qpos[model.jnt_qposadr[joints]].tolist(),
                "velocities": data.qvel[model.jnt_dofadr[joints]].tolist(),
                "faults": [],
                "sample_time_s": float(data.time),
            }
            if entity == "gripper":
                result.update(aperture_m=None, force_n=None, object_detected=None, pressure_pa=None)
            if entity in {"arm", "torso"}:
                result.update(efforts=[], flange_pose=pose(model, data, body, base))
            return result

        def write(target):
            command = vector(target, size, lower=lower, upper=upper)
            world.step({part: command})
            return {"accepted": True}

        def write_base(**args):
            from roboenv.devices.base import command
            value = command(intrinsics["control"], args)
            return write([value["velocity"][axis] for axis in ("vx", "vy", "wz")])

        def stop():
            world.stop_part(part)
            return {"stopped": True, **intrinsics["stop"]}

        register("read", read)
        from roboenv.devices.base import command_schema
        register(
            "write",
            write_base if entity == "base" else write,
            command_schema(intrinsics["control"]) if entity == "base" else arguments({"target": array(size)}, ["target"]),
            mutation=True,
        )
        register("stop", stop, mutation=True)
    elif entity == "camera":
        camera = spec.config.get("camera")
        if camera is None:
            if not world._camera_names:
                raise ValueError("Camera must be enabled in the task's simulator configuration")
            camera = world._camera_names[0]
        if not world._render or camera not in world._camera_names:
            raise ValueError("Camera must be enabled in the task's simulator configuration")
        camera_id = sim.model.camera_name2id(camera)
        height, width = world._camera_size
        intrinsics = camera_intrinsics(width, height, model.cam_fovy[camera_id],
                                       definition.type == "rgbd")

        def read():
            from roboenv_runtime.runtime.worker import write_png

            directory = Path(context.native.run_dir) / "captures"
            directory.mkdir(parents=True, exist_ok=True)
            file = directory / f"{uuid.uuid4().hex}.png"
            extra = {}
            # On-demand paired acquisition uses the Kinex camera Unix clock.
            acquired = time.time()
            if definition.type == "rgbd":
                rgb, depth = world.render_rgbd(camera)
                if np.asarray(depth).shape != np.asarray(rgb).shape[:2]:
                    raise ValueError("Depth and RGB image dimensions must match")
                depth_file = file.with_suffix(".pgm")
                write_depth(depth_file, depth)
                extra = {"depth_file": str(depth_file), "depth_width": width,
                         "depth_height": height, "depth_sample_time_s": acquired}
            else:
                rgb = world.render(camera)
            h, w, _ = write_png(file, rgb)
            return {"file": str(file), "height": h, "width": w,
                    "sample_time_s": acquired, **extra}

        register("read", read)
    def relative_mount(position, rotation, base):
        import mujoco

        base_rotation = data.xmat[base].reshape(3, 3).T
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (base_rotation @ rotation).ravel())
        return {
            "translation": (base_rotation @ (position - data.xpos[base])).tolist(),
            "rotation": [*quat[1:].tolist(), float(quat[0])],
        }

    if entity in {"arm", "torso"}:
        parent_entity = "torso" if entity == "arm" else "base"
        parent = (context.devices or {}).get(parent_entity, parent_entity)
        extrinsics = {
            "source": "introspected", "method": "introspection", "calibrated_at": None,
            "mount": {"parent": parent, "parent_frame": "flange" if entity == "arm" else parent,
                      "translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
        }
    elif entity == "gripper":
        flange = sim.model.body_name2id(robot.robot_model.eef_name["right"])
        site = sim.model.site_name2id(robot.gripper["right"].important_sites["grip_site"])
        transform = relative_mount(data.site_xpos[site], data.site_xmat[site].reshape(3, 3), flange)
        extrinsics = {
            "source": "introspected", "method": "introspection", "calibrated_at": None,
            "mount": {
                "parent": (context.devices or {}).get("arm", "arm"), "parent_frame": "flange",
                "translation_m": transform["translation"], "rotation_xyzw": transform["rotation"],
            },
        }
    elif entity == "camera":
        names = context.devices or {}
        frames = {0: (names.get("world", "world"), None)}
        if "base" in names:
            frames[sim.model.body_name2id(robot.robot_model.base.root_body)] = (names["base"], None)
        if "arm" in names:
            frames[arm_base] = (names["arm"], "base")
            frames[sim.model.body_name2id(robot.robot_model.eef_name["right"])] = (
                names["arm"],
                "flange",
            )
        if "torso" in names:
            frames[arm_base] = (names["torso"], "flange")
        def rigid_root(body):
            while body and not model.body_jntnum[body]:
                body = int(model.body_parentid[body])
            return body

        # The camera can sit on the lift support ABOVE the named torso flange.
        # Resolve all bodies in the same rigid assembly, including descendants.
        rigid_frames = {rigid_root(body): body for body in frames}
        ancestor = rigid_frames.get(rigid_root(int(model.cam_bodyid[camera_id])))
        if ancestor is not None:
            parent, frame = frames[ancestor]
            # MuJoCo cameras look along -Z with +Y up; optical frames use +Z forward, +Y down.
            rotation = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            transform = relative_mount(data.cam_xpos[camera_id], rotation, ancestor)
            extrinsics = {
                "source": "introspected", "method": "introspection", "calibrated_at": None,
                "mount": {
                    "parent": parent, "parent_frame": frame or parent,
                    "translation_m": transform["translation"],
                    "rotation_xyzw": transform["rotation"],
                },
            }
    elif entity == "base":
        extrinsics = {"source": "introspected"}
    if entity == "world":
        intrinsics = {"source": "default", "bounds": None, "surfaces": [], "method": "unknown", "calibrated_at": None}
    definition = replace(definition, intrinsics=intrinsics, extrinsics=extrinsics)
    return BoundDevice(
        definition,
        actions,
        implementation="robocasa-native-device",
        validation="live model and native controller resolved; replay unverified",
        binding={"id": spec.binding, "entity": entity},
    )
