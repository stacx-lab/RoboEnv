"""Buffered references on the world's single native clock.

The harness supplies references; this layer owns timing and terminal references.
"""

import math
from bisect import bisect_right
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
        if "write" not in device.actions:
            return device
        period = context.native.control_interval_s
        if not math.isfinite(period) or period <= 0:
            raise ValueError("native control interval must be positive")
        joint = device.definition.type in {"joint", "dexterous_hand"}
        facts = device.definition.intrinsics
        joints = facts["model"]["joints"] if joint else []
        setter, reader, stopper = (device.actions[name] for name in ("write", "read", "stop"))

        def state(execution):
            value = reader.handler({}, execution)
            if value.get("faults"):
                raise DeviceError(
                    "device_fault",
                    "native device reported faults",
                    details={"state": value, "effects": "partial"},
                )
            return value

        validate_target = action_validator(setter.input_schema)

        def preflight(args):
            sequence = "actions" in args
            if joint:
                targets = args["actions"] if sequence else [args["target"]]
                for target in targets:
                    for q, model in zip(target, joints, strict=True):
                        bounds = model["limits"]
                        if (
                            not math.isfinite(q)
                            or bounds
                            and not bounds["min"] <= q <= bounds["max"]
                        ):
                            raise ValueError("target exceeds native joint limits")
            if not joint and sequence:
                for target in args["actions"]:
                    validate_target({"target": target})
                    if setter.validate is not None:
                        setter.validate({"target": target})
        def execute(args, execution):
            schedule = motion_scheduler()
            if schedule is None:
                raise RuntimeError("device motion requires the world execution scheduler")
            preflight(args)
            sequence = "actions" in args
            targets = args.get("actions", [])
            # Round cumulative boundaries, not each duration: fractional rates
            # then alternate hold lengths without accumulating clock drift.
            ends = (
                [math.floor((i + 1) / args["freq_hz"] / period + 0.5) for i in range(len(targets))]
                if sequence
                else []
            )
            if sequence:
                ends[-1] = max(1, ends[-1])
            budget = args.get("for_s", ends[-1] * period if sequence else period)
            limit = math.floor(budget / period + 1e-8)
            count = min(limit, ends[-1]) if sequence else limit

            identity = request_identity.get() or uuid.uuid4().hex

            def motion():
                elapsed, completed = 0.0, 0
                stopped = False

                def stop():
                    nonlocal stopped
                    if stopped:
                        return
                    stopped = True
                    try:
                        result = stopper.handler({}, execution)
                    except Exception as error:
                        raise DeviceError("stop_failed", str(error)) from error
                    if result.get("stopped") is not True:
                        raise DeviceError("stop_failed", "native stop was not acknowledged")

                def apply(target):
                    setter.handler({"target": target}, execution)

                try:
                    value = state(execution)
                    if not sequence:
                        setter.handler({k: v for k, v in args.items() if k != "for_s"}, execution)
                    index = None
                    for tick in range(count):
                        if sequence:
                            next_index = min(bisect_right(ends, tick), len(targets) - 1)
                            if next_index != index:
                                apply(targets[next_index])
                                index = next_index
                        yield
                        elapsed = (tick + 1) * period
                        value = state(execution)
                        if sequence:
                            completed = bisect_right(ends, tick + 1)
                            if completed == len(targets) and index != len(targets) - 1:
                                # Faster input streams can skip samples, but the
                                # terminal reference must still be the final target.
                                apply(targets[-1])
                                index = len(targets) - 1
                    timeout = sequence and completed < len(targets)
                    if timeout or (sequence and context.execution_mode == "stepped"):
                        stop()
                        value = state(execution)
                    common = {
                        "execution_id": identity,
                        "elapsed_s": elapsed,
                        "state": value,
                        "terminated_by": "timeout"
                        if timeout
                        else "completed",
                        "reference_state": "stopped" if stopped else "target",
                    }
                    if sequence:
                        return {**common, "completed_actions": completed}
                    return {**common, "accepted": True, "reached": False}
                except Exception as error:
                    try:
                        stop()
                    except Exception as stop_error:
                        raise DeviceError(
                            "stop_failed",
                            str(stop_error),
                            details={
                                "effects": "unknown",
                                "execution_id": identity,
                                "cause": str(error),
                            },
                        ) from stop_error
                    raise DeviceError(
                        getattr(error, "code", "motion_failed"),
                        str(error),
                        details={
                            "effects": "partial",
                            "execution_id": identity,
                            "completed_actions": completed,
                            "elapsed_s": elapsed,
                            "state": reader.handler({}, execution),
                        },
                    ) from error

            return schedule(motion())

        def operation(schema, required):
            return Action(
                lambda args, execution: execute(args, execution),
                schema,
                output_schema={"type": "object", "required": required},
                validate=lambda args: preflight(args),
            )

        vector = {
            "type": "array",
            "minItems": len(joints) if joint else 1,
            "maxItems": len(joints) if joint else 1,
            "items": {"type": "number"},
        }
        positive = {"type": "number", "exclusiveMinimum": 0}
        sequence_fields = {
            "actions": {"type": "array", "minItems": 1, "items": vector},
            "freq_hz": positive,
            "for_s": positive,
        }
        if joint:
            device.actions["write"] = operation(
                object_schema(sequence_fields, ["actions", "freq_hz"]),
                ["completed_actions", "elapsed_s", "state"],
            )
        else:
            schema = setter.input_schema
            if device.definition.type in {"gripper", "suction"}:
                schema = {"oneOf": [schema, object_schema(sequence_fields, ["actions", "freq_hz"])]}
            device.actions["write"] = operation(schema, ["elapsed_s", "state"])
        return device

    return bind
