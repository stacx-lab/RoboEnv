"""RoboBallet device controls and public, read-only workcell queries."""

from pathlib import Path

from roboenv.devices.model import Action, BoundDevice

from roboenv_runtime.bindings.motion import control_binding

from ..shared.profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


@control_binding
def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "roboballet_eight_panda":
        raise ValueError("RoboBallet bindings require the eight-arm reaching world")
    if spec.config or spec.entity != spec.name or context.devices.get(spec.entity) != spec.name:
        raise ValueError("RoboBallet requires canonical unique devices")
    world = context.native.environment
    definition = spec.definition(root)
    expected = world.profiles.get(spec.name)
    if expected is None or expected["type"] != definition.type:
        raise ValueError("Unknown RoboBallet device")
    validate(definition.intrinsics, expected["intrinsics"], spec.name + ".intrinsics")
    validate(definition.extrinsics, expected.get("extrinsics", {}), spec.name + ".extrinsics")
    actions = {}

    def register(name, fn, schema=EMPTY, mutation=False, preflight=None):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "isaaclab.roboballet_eight_panda." + name, args, lambda: fn(**args), mutating=mutation
            ), schema, validate=preflight,
        )

    if definition.type == "world":
        return BoundDevice(definition, {}, implementation="World reference", binding=spec.as_dict())
    register("load", lambda: {"loaded": True})
    register("unload", lambda: {"unloaded": True})
    register("status", lambda: {"loaded": True})
    joint = {"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "number"}}
    if definition.type == "rgbd":
        register("read", lambda: world.read_camera(spec.name, Path(context.native.run_dir) / "captures"))
    else:
        register("read", lambda: world.read_device(spec.name))
        if spec.name == "workcell":
            schema = {"type": "object", "required": ["configurations"],
                      "additionalProperties": False, "properties": {
                          "configurations": {"type": "array", "minItems": 1, "maxItems": 128,
                              "items": {"type": "array", "minItems": 8, "maxItems": 8,
                                        "items": joint}}}}
            register("check", world.check, schema)
        elif definition.type == "joint":
            register("stop", lambda: world.stop(spec.name), mutation=True)
            schema = {"type": "object", "properties": {"target": joint},
                      "required": ["target"], "additionalProperties": False}
            register("write", lambda target: world.set_target(spec.name, target), schema, True,
                     lambda args: world.validate_target(spec.name, args["target"]))
    return BoundDevice(definition, actions, implementation="RoboBallet eight-arm reaching",
                       binding=spec.as_dict())
