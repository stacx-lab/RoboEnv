"""Bindings select devices in the existing Go2 world; they never own physics."""

from pathlib import Path

import numpy as np

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.adapters.isaaclab.rig import FEET, JOINTS

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
TYPES = {
    "body": "body",
    "camera_front": "rgbd",
    "imu": "imu",
    "world": "world",
    **{foot: "other" for foot in FEET},
}


def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "go2":
        raise ValueError("Isaac Lab binding requires the Go2 world")
    world = context.native._env
    entity = spec.entity
    definition = spec.definition(root)
    if entity not in TYPES or definition.type != TYPES[entity] or spec.config:
        raise ValueError("Unsupported Isaac Lab device selection or configuration")
    if context.devices[entity] != spec.name:
        raise ValueError("A native device can have only one logical binding")
    if entity != "world" and entity != "body":
        mount = definition.extrinsics.get("mount", {})
        if mount.get("parent") != context.devices.get("body"):
            raise ValueError("Sensor mount must name the selected body device")
        if entity in FEET and mount.get("parent_frame") != entity:
            raise ValueError("Contact sensor mount must name its foot frame")
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
        if tuple(j["name"] for j in definition.intrinsics["joints"]) != JOINTS:
            raise ValueError("Body profile joint order differs from the rig")
        register(
            "read", world.read_body, ["positions", "velocities", "frame_poses", "sample_time_s"]
        )

        def write(target):
            command = np.asarray(target, dtype=float)
            limits = np.asarray(world.model["limits"])
            if command.shape != (len(JOINTS),) or not np.all(np.isfinite(command)):
                raise ValueError("Expected twelve finite joint targets")
            if np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]):
                raise ValueError("Joint target outside native limits")
            return world.write(command)

        register(
            "write",
            write,
            ["accepted"],
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["target"],
                "properties": {
                    "target": {
                        "type": "array",
                        "minItems": 12,
                        "maxItems": 12,
                        "items": {"type": "number"},
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
            ["file", "depth_file", "depth_scale", "width", "height", "channels", "sample_time_s"],
        )
    if entity == "imu" or entity in FEET:
        register(
            "read",
            lambda: world.read_sensor(entity),
            [
                "sample_time_s",
                *(["angular_velocity", "linear_acceleration"] if entity == "imu" else ["force"]),
            ],
        )
    return BoundDevice(
        definition, actions, implementation="Isaac Lab native device", binding=spec.as_dict()
    )
