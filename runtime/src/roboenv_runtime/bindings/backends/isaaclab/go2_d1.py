"""Typed devices bound to one Go2–D1 simulation owner."""

from pathlib import Path
from roboenv.devices.model import Action, BoundDevice
from roboenv.devices.velocity import command_schema
from roboenv_runtime.bindings.motion import control_binding
from ..shared.profile import validate

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def read_robot_state(world, name):
    """Expose physical robot faults, keeping task failure verdicts owner-only."""
    state = world.read_device(name)
    state["faults"] = ["fallen"] if "robot_fallen" in state["faults"] else []
    return state


@control_binding
def bind(spec, root, context):
    if context.backend != "isaaclab" or context.embodiment != "go2_d1":
        raise ValueError("Go2–D1 binding requires the Go2–D1 world")
    world = context.native._env
    definition = spec.definition(root)
    if spec.config or spec.entity != spec.name or context.devices.get(spec.entity) != spec.name:
        raise ValueError("Go2–D1 requires canonical unique devices")
    expected = world.profiles.get(spec.name)
    if expected is None or expected["type"] != definition.type:
        raise ValueError("Unknown Go2–D1 device")
    for section in ("intrinsics", "extrinsics"):
        validate(getattr(definition, section), expected.get(section, {}), spec.name + "." + section)
    actions = {}

    def register(name, fn, schema=EMPTY, mutation=False, preflight=None):
        actions[name] = Action(
            lambda args, execution: execution.perform(
                "isaaclab.go2_d1." + name, args, lambda: fn(**args), mutating=mutation
            ),
            schema,
            validate=preflight,
        )

    if spec.name == "world":
        return BoundDevice(definition, {}, implementation="World reference", binding=spec.as_dict())
    register("load", lambda: {"loaded": True})
    register("unload", lambda: {"unloaded": True})
    register("status", lambda: {"loaded": True})
    if spec.name.startswith("camera_"):
        register(
            "read", lambda: world.read_camera(spec.name, Path(context.native.run_dir) / "captures")
        )
    else:
        register("read", lambda: read_robot_state(world, spec.name))
        register("stop", lambda: world.stop(spec.name), mutation=True)
        schema = (
            command_schema(world.profiles["base"]["intrinsics"]["control"])
            if spec.name == "base"
            else {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1 if spec.name == "gripper" else 6,
                        "maxItems": 1 if spec.name == "gripper" else 6,
                    }
                },
                "required": ["target"],
                "additionalProperties": False,
            }
        )
        def set_target(**args):
            if spec.name == "gripper":
                args = {"target": args["target"][0]}
            return world.set_target(spec.name, args)

        # Keep native Go2 validation ahead of replacing an active buffer.
        def preflight(args):
            world._validate(
                spec.name, {"target": args["target"][0]} if spec.name == "gripper" else args
            )

        register("write", set_target, schema, True, preflight)
    return BoundDevice(
        definition,
        actions,
        implementation="Go2 leg gait and independent D1 control",
        binding=spec.as_dict(),
    )
