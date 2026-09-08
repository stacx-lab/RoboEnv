"""Simulator-specific binding factories; imported only when devices are configured."""


def default_registry():
    from roboenv.devices.bindings.registry import BindingRegistry

    from .fakeshell import bind_puck
    from .simulators import bind_simulator_device

    registry = BindingRegistry()
    registry.register("fakeshell.puck/1", bind_puck)
    from .backends.robocasa.devices import bind

    registry.register("robocasa.device/1", bind)
    from .backends.isaaclab.devices import bind as bind_isaaclab

    registry.register("isaaclab.device/1", bind_isaaclab)
    for backend in ("libero", "robotwin", "robodojo", "behavior"):
        registry.register(f"{backend}.device/1", bind_simulator_device)
    return registry
