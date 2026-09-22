"""Basic device operations on source-backed native worlds; no primitive dispatch."""

import copy
import time
import uuid
from dataclasses import replace
from pathlib import Path

from roboenv.devices.velocity import command as velocity_command
from roboenv.devices.velocity import command_schema
from roboenv.devices.model import Action, BoundDevice

from roboenv_runtime.adapters.robocasa.depth import write_depth
from roboenv_runtime.bindings.backends.libero.devices import same_facts

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def bind(spec, root, context):
    if context.backend not in {"robotwin", "robodojo", "behavior"}:
        raise ValueError("Source-backed device binding requires its native backend")
    world = context.native._env
    if context.embodiment != world.embodiment or spec.config:
        raise ValueError("Native embodiment mismatch or unsupported device configuration")
    definition = spec.definition(root)
    if spec.entity not in world.device_facts:
        raise ValueError("Device does not exist in the selected native rig")
    facts = copy.deepcopy(world.device_facts[spec.entity])
    if facts["type"] != definition.type:
        raise ValueError("Device type differs from native entity")
    for key in ("intrinsics", "extrinsics"):
        if getattr(definition, key).get("source") == "introspected" and (
            not same_facts(getattr(definition, key), facts.get(key, {}))
        ):
            raise ValueError(f"Catalog {key} differ from the native rig")
    names = context.devices or {}
    if "stop" in facts["intrinsics"]:
        facts["intrinsics"]["stop"]["affected_devices"] = [definition.name]
    mount = facts.get("extrinsics", {}).get("mount")
    if mount:
        parent = mount["parent"]
        mount["parent"] = names.get(parent, parent)
        if mount["parent_frame"] == parent:
            mount["parent_frame"] = mount["parent"]
    definition = replace(
        definition, intrinsics=facts["intrinsics"], extrinsics=facts.get("extrinsics", {})
    )
    actions = {}

    def register(name, function, schema=EMPTY, *, mutation=False, required=()):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "binding." + name, args, lambda: function(**args), mutating=mutation
            ),
            schema,
            output_schema={"type": "object", "required": list(required)},
        )

    if definition.type == "world":
        return BoundDevice(definition, {}, implementation=f"{context.backend}-native-device")
    register("load", lambda: {"loaded": True}, required=("loaded",))
    register("unload", lambda: {"unloaded": True}, required=("unloaded",))
    register("status", lambda: {"loaded": True}, required=("loaded",))
    if definition.type in {"rgb", "rgbd"}:

        def read():
            from roboenv_runtime.runtime.worker import write_png

            acquired = time.time()
            rgb, depth = world.device_image(spec.entity)
            directory = Path(context.native.run_dir) / "captures"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{uuid.uuid4().hex}.png"
            height, width, _ = write_png(path, rgb)
            result = {
                "file": str(path),
                "height": height,
                "width": width,
                "sample_time_s": acquired,
            }
            if definition.type == "rgbd":
                depth_path = path.with_suffix(".pgm")
                if depth.shape != rgb.shape[:2]:
                    raise ValueError("Native RGB and depth dimensions disagree")
                write_depth(depth_path, depth)
                result.update(
                    depth_file=str(depth_path),
                    depth_height=height,
                    depth_width=width,
                    depth_sample_time_s=acquired,
                )
            return result

        register("read", read, required=("file", "height", "width", "sample_time_s"))
    else:
        register(
            "read",
            lambda: world.device_read(spec.entity),
            required=("velocities",)
            if definition.type == "velocity"
            else ("positions", "velocities"),
        )
        size = len(world.control.limits[spec.entity])

        def write(**args):
            """Set this device's goal and execute it; report the time spent.

            Arrival is judged here, against the device's own measured state; the
            world only owns the clock. A velocity device has no goal to arrive
            at, so it never reports reached.
            """
            for_s, arrive = args.pop("for_s", None), args.pop("arrive", None)
            target = args.get("target")
            if definition.type == "velocity":
                value = velocity_command(facts["intrinsics"]["control"], args)
                target = [value["velocity"][axis] for axis in ("vx", "vy", "wz")]

            def arrived():
                tolerance = (arrive or {}).get("joint_tolerance")
                if not tolerance or definition.type == "velocity":
                    return False
                state = world.device_read(spec.entity)
                measured = state.get("positions") or []
                if len(measured) != len(target) or len(tolerance) != len(target):
                    return False
                if any(abs(m - t) > limit for m, t, limit in zip(measured, target, tolerance)):
                    return False
                quiet = (arrive or {}).get("quiet_velocity")
                if quiet is None:
                    return True
                speeds = state.get("velocities") or []
                return len(speeds) == len(quiet) and all(
                    abs(v) <= limit for v, limit in zip(speeds, quiet)
                )

            outcome = world.device_write(spec.entity, target, for_s=for_s, arrived=arrived)
            return {**outcome, "reached": arrived(), "state": world.device_read(spec.entity)}

        def stop():
            result = world.device_stop(spec.entity)
            return {
                **result,
                **facts["intrinsics"].get("stop", {}),
                "affected_devices": [definition.name],
            }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["target"],
            "properties": {
                "target": {
                    "type": "array",
                    "minItems": size,
                    "maxItems": size,
                    "items": {"type": "number"},
                },
                "for_s": {"type": "number", "exclusiveMinimum": 0},
                "arrive": {"type": "object"},
            },
        }
        if definition.type == "velocity":
            schema = command_schema(facts["intrinsics"]["control"])
        register("write", write, schema, mutation=True,
                 required=("accepted", "elapsed_s", "reached", "state"))
        register("stop", stop, mutation=True, required=("stopped", "behavior", "affected_devices"))
    return BoundDevice(
        definition,
        actions,
        implementation=f"{context.backend}-native-device",
        validation="Native inventory checked; task acceptance is separate",
        binding={"id": spec.binding, "entity": spec.entity},
    )
