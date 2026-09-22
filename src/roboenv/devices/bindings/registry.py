"""Explicit binding factories. Configuration never supplies an import path."""

from dataclasses import dataclass, replace
from typing import Any

from roboenv.devices.service import DeviceService


@dataclass(frozen=True)
class BindingContext:
    backend: str
    embodiment: str
    native: Any
    public_dispatch: Any = None
    public_tools: Any = None
    devices: Any = None


class BindingRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name, factory):
        if name in self._factories or not callable(factory):
            raise ValueError(f"duplicate binding or invalid factory: {name}")
        self._factories[name] = factory

    def assemble(self, plan, root, context, *, manifest=None, execution_lock=None,
                 segment_scheduler=None, resolve=None):
        if plan.backend != context.backend:
            raise ValueError("binding backend mismatch")
        if plan.embodiment is not None and plan.embodiment != context.embodiment:
            raise ValueError("binding embodiment mismatch")
        for spec in plan.devices:
            if spec.binding not in self._factories:
                raise ValueError(f"unknown registered binding: {spec.binding}")
        context = replace(context, devices={spec.entity: spec.name for spec in plan.devices})
        devices = []
        try:
            for spec in plan.devices:
                device = self._factories[spec.binding](spec, root, context)
                devices.append(device)
                if device.describe().get("name") != spec.name:
                    raise ValueError("binding returned a different logical device name")
            return DeviceService(devices, manifest=manifest, execution_lock=execution_lock,
                                 segment_scheduler=segment_scheduler, resolve=resolve)
        except BaseException:
            for device in reversed(devices):
                try:
                    device.close()
                except Exception:
                    pass  # keep the original setup failure, release all acquired devices
            raise
