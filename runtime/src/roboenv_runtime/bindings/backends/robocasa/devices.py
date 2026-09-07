"""RoboCasa devices bind directly to native controller parts, never task macros."""

from dataclasses import replace
from pathlib import Path
import uuid

import numpy as np

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.adapters.robocasa.geometry import chain, pose, vector
from roboenv_runtime.adapters.robocasa.depth import DEPTH_SCALE, write_depth

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
PARTS = {"arm": "right", "base": "base", "torso": "torso", "gripper": "right_gripper"}
TYPES = {
    "arm": "manipulator",
    "base": "body",
    "torso": "manipulator",
    "gripper": "end_effector",
    "camera": "rgb",
    "world": "world",
    "robot": "other",
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


def part_actions(robot, sim, part):
    controller = robot.part_controllers[part]
    if part in {"right", "torso"}:
        model = sim.model._model
        return [
            {"joint": sim.model.joint_id2name(int(j)),
             "units": "m" if model.jnt_type[j] == 2 else "rad",
             **({"bounds": {"min": float(model.jnt_range[j, 0]),
                            "max": float(model.jnt_range[j, 1])}} if model.jnt_limited[j] else {})}
            for j in controller.joint_index
        ]
    components = {"base": ["forward", "left", "yaw_rate"], "right_gripper": ["close"]}[part]
    return [{"component": name, "units": "normalized", "bounds": {"min": -1, "max": 1}}
            for name in components]


def command_vector(target, actions):
    return vector(target, len(actions),
                  lower=[a.get("bounds", {}).get("min", -np.inf) for a in actions],
                  upper=[a.get("bounds", {}).get("max", np.inf) for a in actions])


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
    if "robot" in (context.devices or {}) and set(PARTS) & set(context.devices):
        raise ValueError(
            "Select the composite robot or its actuators, not overlapping command channels"
        )
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
                    "stop": ["stopped"],
                    "read": (
                        ["file", "width", "height", "channels"]
                        + (["depth_file", "depth_scale"] if definition.type == "rgbd" else [])
                        if entity == "camera"
                        else ["positions", "velocities", "faults", "sample_time_s"]
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
        command_actions = part_actions(robot, sim, part)
        if size != len(command_actions):
            raise ValueError("Unexpected native controller action space")
        intrinsics.update(
            source="introspected",
            dof=len(joints),
            joints=[
                {
                    "name": sim.model.joint_id2name(int(j)),
                    "type": "prismatic" if model.jnt_type[j] == 2 else "revolute",
                    "source": "introspected",
                    **(
                        {
                            "limits": {
                                "min": float(model.jnt_range[j, 0]),
                                "max": float(model.jnt_range[j, 1]),
                            }
                        }
                        if model.jnt_limited[j]
                        else {"unbounded": True}
                    ),
                }
                for j in joints
            ],
            actions=command_actions,
        )
        # Diagnostic metadata; command units and bounds are explicit above.
        intrinsics["controller"] = {
            "name": type(controller).__name__,
            "input_type": getattr(controller, "input_type", None),
            "input_ref_frame": getattr(controller, "input_ref_frame", None),
        }
        for field in ("output_min", "output_max"):
            if hasattr(controller, field):
                intrinsics["controller"][field] = np.asarray(getattr(controller, field)).tolist()
        # Homes are model defaults, not the task's current (possibly noisy) pose.
        if entity in {"arm", "torso"}:
            home = robot.init_qpos if entity == "arm" else robot.init_torso_qpos
            if home is None:
                home = robot.robot_model.init_torso_qpos
            intrinsics["home_joints"] = vector(home, len(joints)).tolist()
        # Motor ctrlrange is torque; a velocity servo's ctrlrange is a velocity
        # target. Neither is a hard bound on measured joint velocity.
        actuators = np.flatnonzero(
            np.isin(model.actuator_trnid[:, 0], joints) & (model.actuator_trntype == 0)
        )
        intrinsics["actuators"] = [
            {
                "name": sim.model.actuator_id2name(int(a)),
                "joint": sim.model.joint_id2name(int(model.actuator_trnid[a, 0])),
                "control_limited": bool(model.actuator_ctrllimited[a]),
                "control_range": model.actuator_ctrlrange[a].tolist(),
                "force_limited": bool(model.actuator_forcelimited[a]),
                "force_range": model.actuator_forcerange[a].tolist(),
            }
            for a in actuators
        ]
        intrinsics["velocity_limits"] = {
            "source": "introspected",
            "measured_joint_velocity_cap": None,
            "notes": "Native model/controller has no hard measured joint velocity cap; "
            "controller deltas and actuator control ranges are not such a cap.",
        }
        if entity in {"arm", "torso"}:
            intrinsics["joint_limit_semantics"] = {
                "source": "MuJoCo compiled model jnt_limited/jnt_range",
                "units": "rad" if entity == "arm" else "m",
                "joint_order": "intrinsics.joints; same order as read.positions and fk/ik joints",
                "enforcement": "MuJoCo soft joint-limit constraints; measured positions may overshoot",
                "activation_margin": model.jnt_margin[joints].tolist(),
                "solver_reference": model.jnt_solref[joints].tolist(),
                "solver_impedance": model.jnt_solimp[joints].tolist(),
                "fk_ik": "Explicit FK joints and IK seeds must be within limits; IK results "
                "are bounded. These checks do not guarantee collision-free or reachable motion.",
                "read": "Reports actual joints and flange pose even outside nominal limits",
            }
            intrinsics["joint_limit_semantics"]["controller"] = (
                "Absolute joint targets are validated against joint limits before acceptance. "
                "stop holds measured joints; target acceptance does not imply arrival."
            )
        elif entity == "base":
            intrinsics["velocity_limits"]["actuator_target_ranges"] = [
                {"joint": a["joint"], "range": a["control_range"],
                 "units": "rad/s" if model.jnt_type[j] == 3 else "m/s"}
                for a, j in zip(intrinsics["actuators"], joints, strict=True)
            ]
        elif entity == "gripper":
            intrinsics["controller"]["normalized_increment"] = float(robot.gripper["right"].speed)
            intrinsics["actions"][0]["meaning"] = (
                "sign only: positive closes, negative opens, zero holds; native gripper "
                f"normalized increment {robot.gripper['right'].speed:g} per interval"
            )
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
            intrinsics["kinematics"] = chain(model, joints, body, base)

        def read():
            result = {
                "positions": data.qpos[model.jnt_qposadr[joints]].tolist(),
                "velocities": data.qvel[model.jnt_dofadr[joints]].tolist(),
                "faults": [],
                "sample_time_s": float(data.time),
            }
            if entity in {"arm", "torso", "base"}:
                key = "pose" if entity == "base" else "flange_pose"
                result[key] = pose(model, data, body, base)
            return result

        def write(target):
            command = command_vector(target, command_actions)
            world.step({part: command})
            return {"accepted": True}

        def stop():
            world.stop_part(part)
            return {"stopped": True}

        register("read", read)
        register(
            "write",
            write,
            arguments({"target": array(size)}, ["target"]),
            mutation=True,
        )
        register("stop", stop, mutation=True)
    elif entity == "robot":
        size = sum(end - start for start, end in robot.composite_controller._action_split_indexes.values())
        joints = np.unique(np.concatenate([c.joint_index for c in robot.part_controllers.values()]))
        intrinsics.update(
            actions=[a for part in robot.composite_controller._action_split_indexes
                     for a in part_actions(robot, sim, part)],
            parts={
                name: list(span)
                for name, span in robot.composite_controller._action_split_indexes.items()
            },
        )
        definition = replace(definition, control_mode="native_composite")

        def read():
            return {
                "positions": data.qpos[model.jnt_qposadr[joints]].tolist(),
                "velocities": data.qvel[model.jnt_dofadr[joints]].tolist(),
                "faults": [],
                "sample_time_s": float(data.time),
            }

        def write(target):
            world.step_vector(command_vector(target, intrinsics["actions"]))
            return {"accepted": True}

        def stop():
            for part in robot.part_controllers:
                world.stop_part(part)
            return {"stopped": True}

        register("read", read)
        register(
            "write",
            write,
            arguments({"target": array(size)}, ["target"]),
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
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
        intrinsics["source"] = "introspected"
        intrinsics["camera_name"] = camera
        intrinsics["value"] = dict(
            width=width,
            height=height,
            fx=float(focal),
            fy=float(focal),
            cx=width / 2,
            cy=height / 2,
            distortion=[],
            fovy_degrees=float(model.cam_fovy[camera_id]),
        )
        if definition.type == "rgbd":
            intrinsics["depth"] = {
                "scale": DEPTH_SCALE,
                "aligned": True,
                "format": "PGM P5, unsigned 16-bit big-endian; 0 invalid",
                "units": "optical-axis metres = raw * scale",
                "encoding": "Native depth buffer converted using live near/far clipping planes, "
                "then rounded to millimetres. Background, nonfinite and depths outside "
                "[0.001,65.535] metres are encoded as 0.",
            }

        def read():
            from roboenv_runtime.runtime.worker import write_png

            directory = Path(context.native.run_dir) / "captures"
            directory.mkdir(parents=True, exist_ok=True)
            file = directory / f"{uuid.uuid4().hex}.png"
            extra = {}
            if definition.type == "rgbd":
                rgb, depth = world.render_rgbd(camera)
                if np.asarray(depth).shape != np.asarray(rgb).shape[:2]:
                    raise ValueError("Depth and RGB image dimensions must match")
                depth_file = file.with_suffix(".pgm")
                write_depth(depth_file, depth)
                extra = {"depth_file": str(depth_file), "depth_scale": DEPTH_SCALE}
            else:
                rgb = world.render(camera)
            h, w, channels = write_png(file, rgb)
            return {"file": str(file), "height": h, "width": w, "channels": channels, **extra}

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

    def mounting(values):
        return {"translation": values[:3], "rotation": values[3:]}

    if entity in {"arm", "torso"}:
        extrinsics["source"] = "introspected"
        extrinsics["mount"] = {
            **extrinsics.get("mount", {}),
            "pose": mounting([0, 0, 0, 0, 0, 0, 1]),
        }
        extrinsics.pop("needs_calibration", None)
        if entity == "arm":
            extrinsics["mount"]["parent_frame"] = "flange"
    elif entity == "gripper":
        extrinsics["source"] = "introspected"
        flange = sim.model.body_name2id(robot.robot_model.eef_name["right"])
        site = sim.model.site_name2id(robot.gripper["right"].important_sites["grip_site"])
        extrinsics["mount"] = {
            **extrinsics.get("mount", {}),
            "pose": relative_mount(
                data.site_xpos[site], data.site_xmat[site].reshape(3, 3), flange
            ),
        }
        extrinsics.pop("needs_calibration", None)
    elif entity == "camera":
        names = context.devices or {}
        frames = {0: ("world", None)}
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
            extrinsics["source"] = "introspected"
            parent, frame = frames[ancestor]
            # MuJoCo cameras look along -Z with +Y up; image optical frames
            # use +Z forward and +Y down.
            rotation = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
            extrinsics["mount"] = {
                "parent": parent,
                "pose": relative_mount(data.cam_xpos[camera_id], rotation, ancestor),
            }
            if frame:
                extrinsics["mount"]["parent_frame"] = frame
            extrinsics.pop("needs_calibration", None)
    elif entity == "base":
        # A mobile chassis may be its own frame root. Only mark an explicitly
        # configured mount dynamic; availability of a pose read adds no edge.
        if isinstance(extrinsics.get("mount"), dict):
            extrinsics["mount"] = {**extrinsics["mount"], "dynamic": True}
    intrinsics["interface"] = f"roboenv/{definition.type}/1"
    definition = replace(definition, intrinsics=intrinsics, extrinsics=extrinsics)
    return BoundDevice(
        definition,
        actions,
        implementation="robocasa-native-device",
        validation="live model and native controller resolved; replay unverified",
        binding={"id": spec.binding, "entity": entity},
    )
