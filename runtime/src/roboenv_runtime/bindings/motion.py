"""Device buffers executed on the world's native control cadence."""

import math
import uuid
from functools import wraps

from roboenv.devices.execution import motion_scheduler, request_identity
from roboenv.devices.model import Action, DeviceError
from roboenv.validation import action_validator


def object_schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def control_binding(factory):
    @wraps(factory)
    def bind(spec, root, context, **kwargs):
        device = factory(spec, root, context, **kwargs)
        if device.definition.type == "world":
            from roboenv_runtime.bindings.world import clock_actions

            device.actions = clock_actions(context)
            return device
        if "write" not in device.actions:
            return device
        period = context.native.control_interval_s
        if not math.isfinite(period) or period <= 0:
            raise ValueError("native control interval must be positive")
        kind = device.definition.type
        joint = kind in {"joint", "dexterous_hand"}
        scalar = kind in {"gripper", "suction"}
        joints = device.definition.intrinsics["model"]["joints"] if joint else []
        setter, reader, stopper = (device.actions[name] for name in ("write", "read", "stop"))
        validate_target = action_validator(setter.input_schema)

        def scalar_target(value):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("actions must contain finite numbers")
            control = device.definition.intrinsics["control"]
            if control["mode"] == "binary":
                if value not in (control["off_target"], control["on_target"]):
                    raise ValueError("target must equal off_target or on_target")
            elif control["mode"] == "direction":
                if not -1 <= value <= 1:
                    raise ValueError("direction target must be in [-1, 1]")
                value = 0 if value == 0 else math.copysign(1, value)
            elif control["mode"] == "position":
                bounds = control["bounds"]
                if not bounds["min"] <= value <= bounds["max"]:
                    raise ValueError("target exceeds command bounds")
                if control["resolution"] is not None:
                    value = round(value / control["resolution"]) * control["resolution"]
                if not bounds["min"] <= value <= bounds["max"]:
                    raise ValueError("quantized target exceeds command bounds")
            else:
                raise ValueError("unsupported end-effector control mode")
            return [value]

        def targets(args):
            if "actions" in args:
                return [
                    {"target": scalar_target(target) if scalar else target}
                    for target in args["actions"]
                ]
            return [{k: v for k, v in args.items() if k != "duration_s"}]

        def preflight(args):
            for target in targets(args):
                validate_target(target)
                if setter.validate is not None:
                    setter.validate(target)
                if joint:
                    for q, model in zip(target["target"], joints, strict=True):
                        bounds = model["limits"]
                        if (
                            not math.isfinite(q)
                            or bounds
                            and not bounds["min"] <= q <= bounds["max"]
                        ):
                            raise ValueError("target exceeds native joint limits")

        def execute(args, execution):
            schedule = motion_scheduler()
            if schedule is None:
                raise RuntimeError("device motion requires the world scheduler")
            values, duration = targets(args), args["duration_s"]
            identity = request_identity.get() or uuid.uuid4().hex
            value = reader.handler({}, execution)
            if value.get("faults"):
                raise DeviceError("device_fault", "native device reported faults")

            def motion():
                index = None
                ticks = math.ceil(duration / period - 1e-9)
                try:
                    for tick in range(ticks):
                        next_index = min(
                            int(tick * period * len(values) / duration), len(values) - 1
                        )
                        if next_index != index:
                            setter.handler(values[next_index], execution)
                            index = next_index
                        yield
                        if reader.handler({}, execution).get("faults"):
                            raise DeviceError("device_fault", "native device reported faults")
                    if kind == "velocity":
                        stopper.handler({}, execution)
                    elif index != len(values) - 1:
                        setter.handler(values[-1], execution)
                except Exception as error:
                    if getattr(error, "code", None) != "operation_interrupted":
                        try:
                            stopper.handler({}, execution)
                        except Exception as cleanup:
                            raise DeviceError("stop_failed", str(cleanup)) from error
                    raise

            future = schedule(motion())
            if future.done():
                future.result()
            return {"execution_id": identity, "elapsed_s": 0.0, "state": value}

        positive = {"type": "number", "exclusiveMinimum": 0}
        vector = {
            "type": "array",
            "minItems": len(joints) if joint else 1,
            "maxItems": len(joints) if joint else 1,
            "items": {"type": "number"},
        }
        sequence = object_schema(
            {
                "actions": {
                    "type": "array", "minItems": 1,
                    "items": {"type": "number"} if scalar else vector,
                },
                "duration_s": positive,
            },
            ["actions", "duration_s"],
        )

        def timed(schema):
            if "oneOf" in schema:
                return {"oneOf": [timed(item) for item in schema["oneOf"]]}
            return {
                **schema,
                "properties": {**schema["properties"], "duration_s": positive},
                "required": [*schema.get("required", []), "duration_s"],
            }

        schema = sequence if joint or scalar else timed(setter.input_schema)
        device.actions["write"] = Action(
            execute,
            schema,
            output_schema={"type": "object", "required": ["execution_id", "elapsed_s", "state"]},
            validate=preflight,
        )
        return device

    return bind
