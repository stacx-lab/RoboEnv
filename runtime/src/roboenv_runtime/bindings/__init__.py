"""Simulator-specific binding factories; imported only when devices are configured."""


def default_registry():
    from roboenv.bindings.catalog import bind_catalog
    from roboenv.bindings.registry import BindingRegistry

    from .fakeshell import bind_puck
    from .simulators import bind_simulator_device

    registry = BindingRegistry()
    registry.register("catalog/1", bind_catalog)
    registry.register("fakeshell.puck/1", bind_puck)
    for backend in ("libero", "robocasa", "robotwin", "robodojo", "behavior"):
        registry.register(f"{backend}.device/1", bind_simulator_device)
    return registry
