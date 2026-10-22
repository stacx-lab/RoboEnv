"""Simulator-specific binding factories; imported only when devices are configured."""


def default_registry():
    from roboenv.devices.bindings.registry import BindingRegistry

    from .fakeshell import bind_puck

    registry = BindingRegistry()
    registry.register("fakeshell.puck/1", bind_puck)
    from .backends.robocasa import register

    register(registry)
    from .backends.libero.devices import bind as bind_libero

    registry.register("libero.device/1", bind_libero)
    from .native_devices import bind as bind_native

    for backend in ("robotwin", "robodojo", "behavior", "abc"):
        registry.register(f"{backend}.device/1", bind_native)
    from .backends.simple import register as register_simple

    register_simple(registry)
    from .backends.isaaclab.go2_d1 import bind as bind_go2

    registry.register("isaaclab.go2-d1/1", bind_go2)
    from .backends.isaaclab.booster_t2 import bind as bind_t2

    registry.register("isaaclab.booster-t2/1", bind_t2)
    from .backends.isaaclab.yam import bind as bind_yam

    registry.register("isaaclab.yam/1", bind_yam)
    from .backends.isaaclab.roboballet_eight_panda import bind as bind_roboballet_eight_panda

    registry.register("isaaclab.roboballet-eight-panda/1", bind_roboballet_eight_panda)
    return registry
