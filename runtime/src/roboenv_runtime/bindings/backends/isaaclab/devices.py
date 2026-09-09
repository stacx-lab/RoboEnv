"""Bindings select devices in the existing Go2 world; they never own physics."""

from pathlib import Path

import numpy as np

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.adapters.isaaclab.profiles import device_profiles
from roboenv_runtime.adapters.isaaclab.rig import FEET, JOINTS
from .profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "go2":
        raise ValueError("Isaac Lab binding requires the Go2 world")
    world = context.native._env
    entity = spec.entity
    definition = spec.definition(root)
    profiles = device_profiles(world.model, context.devices)
    if entity not in profiles or definition.type != profiles[entity]["type"] or spec.config:
        raise ValueError("Unsupported Isaac Lab device selection or configuration")
    if context.devices.get(entity) != spec.name:
        raise ValueError("A native device can have only one logical binding")
    if entity not in {"world", "body"} and "body" not in context.devices:
        raise ValueError("Sensor mount requires the selected body device")
    if entity != "world":
        for section in ("intrinsics", "extrinsics"):
            validate(getattr(definition, section), profiles[entity][section], f"{spec.name}.{section}")
    actions = {}

    def register(name, fn, required, schema=EMPTY, *, mutation=False):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "isaaclab." + name,
                args,
                lambda: fn(**args),
                mutating=mutation,
            ),
            schema,
            output_schema={"type": "object", "required": required},
        )

    if entity != "world":
        register("load", lambda: {"loaded": True}, ["loaded"])
        register("unload", lambda: {"unloaded": True}, ["unloaded"])
        register("status", lambda: {"loaded": True}, ["loaded"])
    if entity == "body":
        register(
            "read", world.read_body,
            ["positions", "velocities", "efforts", "pose", "body_velocities",
             "frame_poses", "faults", "sample_time_s"]
        )

        register(
            "write",
            lambda target: world.write(np.asarray(target, dtype=float)),
            ["accepted"],
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["target"],
                "properties": {
                    "target": {
                        "type": "array",
                        "minItems": len(JOINTS), "maxItems": len(JOINTS),
                        "prefixItems": [
                            {"type": "number", "minimum": low, "maximum": high}
                            for low, high in world.model["limits"]
                        ],
                    }
                },
            },
            mutation=True,
        )
        register("stop", world.stop, ["stopped"], mutation=True)
    if entity == "camera_front":
        register(
            "read",
            lambda: world.read_camera(Path(context.native.run_dir) / "captures"),
            ["file", "width", "height", "sample_time_s",
             "depth_file", "depth_width", "depth_height", "depth_sample_time_s"],
        )
    if entity == "imu" or entity in FEET:
        register(
            "read",
            lambda: world.read_sensor(entity),
            [
                "sample_time_s",
                *(["angular_velocity", "linear_acceleration", "orientation", "magnetic_field"] if entity == "imu" else ["force", "torque"]),
            ],
        )
    return BoundDevice(
        definition, actions, implementation="Isaac Lab native device", binding=spec.as_dict()
    )
