"""Explicit control-contract factories for one native RoboCasa world."""


def register(registry):
    from . import camera, gripper, joint, velocity
    from roboenv.devices.model import BoundDevice

    for name, factory in (
        ("joint", joint.bind),
        ("velocity", velocity.bind),
        ("gripper", gripper.bind),
        ("camera", camera.bind),
    ):
        registry.register(f"robocasa.{name}/1", factory)

    from roboenv_runtime.bindings.motion import control_binding

    @control_binding
    def world(spec, root, context):
        definition = spec.definition(root)
        if definition.type != "world" or spec.config:
            raise ValueError("world binding requires a world profile and no configuration")
        return BoundDevice(definition, {}, implementation="robocasa-native-device")

    registry.register("robocasa.world/1", world)
