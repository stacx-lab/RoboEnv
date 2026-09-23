"""Velocity/posture commands, independent of simulators and agent harnesses."""

import math


def command(control, args):
    if control.get("mode") != "motion":
        raise ValueError("Expected motion control")
    if not isinstance(args, dict) or set(args) - {"mode", "velocity", "posture"}:
        raise ValueError("Expected mode with velocity and/or posture")
    name = args.get("mode")
    if not isinstance(name, str) or name not in control["modes"]:
        raise ValueError("Unsupported velocity controller mode")
    mode = control["modes"][name]
    result = {"mode": name}
    for channel in ("velocity", "posture"):
        capability = mode[channel]
        if capability is None:
            if channel in args:
                raise ValueError(f"{channel} is unsupported in this mode")
            continue
        target = args.get(channel)
        axes = {c["axis"] for c in capability["components"]}
        if not isinstance(target, dict) or set(target) != axes:
            raise ValueError(f"{channel} requires its declared axes")
        result[channel] = {}
        for component in capability["components"]:
            axis, bounds = component["axis"], component["bounds"]
            value = target[axis]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not bounds["min"] <= value <= bounds["max"]
            ):
                raise ValueError(f"{channel}.{axis} exceeds bounds")
            result[channel][axis] = value
    return result


def command_schema(control):
    variants = []
    for name, mode in control["modes"].items():
        properties = {"mode": {"const": name}, "for_s": {"type": "number", "exclusiveMinimum": 0}}
        required = ["mode"]
        for channel in ("velocity", "posture"):
            capability = mode[channel]
            if capability is None:
                continue
            components = capability["components"]
            axes = {
                c["axis"]: {
                    "type": "number",
                    "minimum": c["bounds"]["min"],
                    "maximum": c["bounds"]["max"],
                }
                for c in components
            }
            properties[channel] = {
                "type": "object",
                "properties": axes,
                "required": [c["axis"] for c in components],
                "additionalProperties": False,
            }
            required.append(channel)
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}
