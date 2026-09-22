"""Small binding construction helpers; the world owns all effects."""

from dataclasses import replace

from roboenv.devices.model import Action, BoundDevice

from .geometry import mounts

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def action(name, function, schema=EMPTY, *, mutating=False):
    return Action(
        lambda args, execution: execution.perform(
            "binding." + name, args, lambda: function(**args), mutating=mutating
        ),
        schema,
    )


def finish(spec, definition, context, intrinsics, actions):
    actions.update(
        {
            "load": action("load", lambda: {"loaded": True}),
            "unload": action("unload", lambda: {"unloaded": True}),
            "status": action("status", lambda: {"loaded": True}),
        }
    )

    return BoundDevice(
        replace(definition, intrinsics=intrinsics, extrinsics=mounts(spec, definition, context)),
        actions,
        implementation="robocasa-native-device",
        binding={"id": spec.binding, "entity": spec.entity},
    )


def target_schema(joints):
    return {
        "type": "object",
        "required": ["target"],
        "additionalProperties": False,
        "properties": {
            "target": {
                "type": "array",
                "minItems": len(joints),
                "maxItems": len(joints),
                "prefixItems": [
                    {
                        "type": "number",
                        **(
                            {"minimum": j["limits"]["min"], "maximum": j["limits"]["max"]}
                            if j["limits"]
                            else {}
                        ),
                    }
                    for j in joints
                ],
            },
            # Every write is a segment: for_s bounds how long the world runs the
            # command, arrive is the spec the device judges itself against.
            "for_s": {"type": "number", "exclusiveMinimum": 0},
            "arrive": {"type": "object"},
        },
    }
