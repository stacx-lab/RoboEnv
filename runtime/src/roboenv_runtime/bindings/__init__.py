"""Simulator-specific binding factories; imported only when devices are configured."""


def default_registry():
    from roboenv.devices.bindings.registry import BindingRegistry

    from .fakeshell import bind_puck

    registry = BindingRegistry()
    registry.register("fakeshell.puck/1", bind_puck)
    from .backends.robocasa import register

    register(registry)
    from .backends.isaaclab.g1 import bind as bind_isaaclab

    registry.register("isaaclab.device/1", bind_isaaclab)
    from .backends.libero.devices import bind as bind_libero

    registry.register("libero.device/1", bind_libero)
    from .native_devices import bind as bind_native

    for backend in ("robotwin", "robodojo", "behavior"):
        registry.register(f"{backend}.device/1", bind_native)
    return registry
