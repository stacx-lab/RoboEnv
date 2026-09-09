"""Typed Panda devices bound to one official LIBERO world."""

import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
from roboenv.devices.model import Action, BoundDevice

from roboenv_runtime.adapters.robocasa.geometry import chain, pose, vector
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics as camera_intrinsics
from roboenv_runtime.bindings.backends.robocasa.manipulator import intrinsics as arm_intrinsics

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
TYPES = {"arm": "manipulator", "gripper": "end_effector", "camera": "rgb", "world": "world"}


def mount(position, rotation, data, base, parent, frame):
    import mujoco

    inverse = data.xmat[base].reshape(3, 3).T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (inverse @ rotation).ravel())
    return {
        "source": "introspected", "method": "introspection", "calibrated_at": None,
        "mount": {
            "parent": parent, "parent_frame": frame,
            "translation_m": (inverse @ (position - data.xpos[base])).tolist(),
            "rotation_xyzw": [*quat[1:].tolist(), float(quat[0])],
        },
    }


def same_facts(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and expected.keys() == actual.keys() and all(
            same_facts(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            same_facts(a, b) for a, b in zip(expected, actual)
        )
    if type(expected) in (int, float):
        return type(actual) in (int, float) and bool(np.isclose(expected, actual, atol=1e-10, rtol=0))
    return expected == actual


def bind(spec, root, context):
    if context.backend != "libero" or context.embodiment != "panda":
        raise ValueError("LIBERO binding requires the native Panda world")
    world = context.native._env
    robot, sim = world._robot, world.sim
    model, data = sim.model._model, sim.data._data
    arm_joints = np.asarray(robot.controller.joint_index, dtype=int)
    base = int(model.body_parentid[model.jnt_bodyid[arm_joints[0]]])
    flange = sim.model.body_name2id(robot.robot_model.eef_name)
    definition = spec.definition(root)
    entity = spec.entity
    if entity not in TYPES or definition.type != TYPES[entity]:
        raise ValueError("Unsupported LIBERO entity or device type")
    if set(spec.config) - ({"camera"} if entity == "camera" else set()):
        raise ValueError("Unknown LIBERO binding configuration")
    names = context.devices or {}
    intrinsics, extrinsics, actions = {}, {}, {}

    def register(name, function, schema=EMPTY, *, mutation=False, required=()):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "binding." + name, args, lambda: function(**args), mutating=mutation
            ),
            schema, output_schema={"type": "object", "required": list(required)},
        )

    if entity != "world":
        register("load", lambda: {"loaded": True}, required=("loaded",))
        register("unload", lambda: {"unloaded": True}, required=("unloaded",))
        register("status", lambda: {"loaded": True}, required=("loaded",))
    if entity in {"arm", "gripper"}:
        joints = arm_joints if entity == "arm" else np.asarray([
            sim.model.joint_name2id(name) for name in robot.gripper_joints
        ])
        size = 7 if entity == "arm" else 1
        if len(arm_joints) != 7 or robot.action_dim != 8:
            raise ValueError("Expected the native seven-joint Panda position controller")
        if entity == "arm":
            intrinsics = arm_intrinsics(
                sim, joints, chain(model, joints, flange, base)["chain"],
                robot.robot_model.init_qpos, definition.name,
            )
            extrinsics = mount(data.xpos[base], data.xmat[base].reshape(3, 3), data, 0,
                               names.get("world", "world"), names.get("world", "world"))
            limits = [j["limits"] for j in intrinsics["kinematics"]["model"]["joints"]]
        else:
            intrinsics = {
                "source": "introspected", "method": "introspection", "calibrated_at": None,
                "tool_type": "parallel_gripper", "aperture_m": None,
                "control": {"mode": "direction", "open_target": -1, "close_target": 1,
                            "increment_per_interval": float(robot.gripper.speed)},
                "feedback": {
                    "positions": [{"name": sim.model.joint_id2name(int(j)), "units": "m"}
                                  for j in joints],
                    "velocities": True, "sensors": [], "clock": "simulation",
                    "aperture_model": None,
                },
                "stop": {"behavior": "hold", "affected_devices": [definition.name]},
            }
            site = sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
            extrinsics = mount(data.site_xpos[site], data.site_xmat[site].reshape(3, 3),
                               data, flange, names.get("arm", "arm"), "flange")
            limits = [{"min": -1, "max": 1}]
        definition = replace(definition, control_mode=(
            "joint_position" if entity == "arm" else "gripper_velocity"
        ))

        def read():
            result = {
                "positions": data.qpos[model.jnt_qposadr[joints]].tolist(),
                "velocities": data.qvel[model.jnt_dofadr[joints]].tolist(),
                "faults": [], "sample_time_s": float(data.time),
            }
            if entity == "arm":
                result.update(efforts=[], flange_pose=pose(model, data, flange, base))
            else:
                result.update(aperture_m=None, force_n=None, object_detected=None, pressure_pa=None)
            return result

        def write(target):
            if not isinstance(target, (list, tuple)) or any(type(v) not in (int, float) for v in target):
                raise ValueError("Target must contain numbers, not booleans or strings")
            command = vector(target, size, lower=[a["min"] for a in limits],
                             upper=[a["max"] for a in limits])
            world.step({entity: command})
            return {"accepted": True}

        def stop():
            world.stop_part(entity)
            return {"stopped": True, **intrinsics["stop"]}

        register("read", read, required=("positions", "velocities", "faults", "sample_time_s"))
        register("write", write, {
            "type": "object", "additionalProperties": False, "required": ["target"],
            "properties": {"target": {"type": "array", "minItems": size, "maxItems": size,
                                      "items": {"type": "number"}}},
        }, mutation=True, required=("accepted",))
        register("stop", stop, mutation=True, required=("stopped", "behavior", "affected_devices"))
    elif entity == "camera":
        camera = spec.config.get("camera", "agentview")
        if not world._render or camera not in world._camera_names:
            raise ValueError("Selected camera must be enabled in the native world")
        camera_id = sim.model.camera_name2id(camera)
        height, width = world._camera_size
        intrinsics = camera_intrinsics(width, height, model.cam_fovy[camera_id], False)
        wrist = camera == "robot0_eye_in_hand"
        parent = names.get("arm", "arm") if wrist else names.get("world", "world")
        rotation = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        extrinsics = mount(data.cam_xpos[camera_id], rotation, data, flange if wrist else 0,
                           parent, "flange" if wrist else parent)

        def read():
            from roboenv_runtime.runtime.worker import write_png

            directory = Path(context.native.run_dir) / "captures"
            directory.mkdir(parents=True, exist_ok=True)
            file = directory / f"{uuid.uuid4().hex}.png"
            acquired = time.time()
            h, w, _ = write_png(file, world.render(camera))
            return {"file": str(file), "height": h, "width": w, "sample_time_s": acquired}

        register("read", read, required=("file", "height", "width", "sample_time_s"))
    else:
        intrinsics = {"source": "default", "bounds": None, "surfaces": [],
                      "method": "unknown", "calibrated_at": None}
    # Fail at connection time if a static rig or sensor selection has drifted.
    for label, expected, actual in (("intrinsics", definition.intrinsics, intrinsics),
                                    ("mounts", definition.extrinsics, extrinsics)):
        if expected.get("source") == "introspected" and not same_facts(expected, actual):
            raise ValueError(f"Catalog {label} do not match the native LIBERO rig")
    return BoundDevice(
        replace(definition, intrinsics=intrinsics, extrinsics=extrinsics), actions,
        implementation="libero-native-device",
        validation="Native model resolved; task success requires separate evidence",
        binding={"id": spec.binding, "entity": entity},
    )
