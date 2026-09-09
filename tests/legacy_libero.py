"""Historical primitive binding fixtures; never selected by production LIBERO."""

from dataclasses import replace
from pathlib import Path

from roboenv.devices.bindings.registry import BindingRegistry
from roboenv_runtime.bindings.simulators import bind_simulator_device

ROBOT = Path(__file__).parent / "fixtures/legacy-libero"


def registry():
    result = BindingRegistry()
    result.register(
        "legacy.libero/1",
        lambda spec, root, context: bind_simulator_device(
            replace(spec, binding="libero.device/1"), root, replace(context, backend="libero")
        ),
    )
    return result
