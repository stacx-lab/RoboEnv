"""Compatibility binding for migrated simulator-specific catalog drivers.

Only this binding knows the legacy public-primitive client and filesystem driver
format. New native bindings implement BoundDevice directly and need neither.
"""

import json
import os
import runpy
import sys
from pathlib import Path

from ..client import Client, _bind_client
from ..contract import UNSUPPORTED_ACTIONS
from ..device_model import Action, BoundDevice, DeviceError, ExecutionContext
from ..device_service import DeviceService
from ..devices import list_devices
from .registry import BindingContext
from .spec import BindingSpec, profile_path


class _LocalClient(Client):
    def __init__(self, dispatch, context):
        self.dispatch, self.context, self._api = dispatch, context, None

    def request(self, op, args=None, **kwargs):
        args = {} if args is None else args
        if op not in {"api", "observe", "call", "score.peek", "capture", "health"}:
            raise DeviceError("permission_denied", "Driver may only use public primitives")
        mutating = (
            op == "call"
            and self.api().get("tools", {}).get(args.get("name"), {}).get("read_only") is not True
        )
        result = self.context.perform(op, args, lambda: self.dispatch(op, args), mutating=mutating)
        if op == "health" and isinstance(result.get("worker"), str):
            result = {**result, "worker": {"state": result["worker"]}}
        return result


def bind_catalog(spec, root, context):
    if not callable(context.public_dispatch):
        raise ValueError("catalog binding requires an explicit public primitive dispatcher")
    if spec.config:
        raise ValueError("catalog binding does not accept native entity configuration")
    profile = spec.resolved_profile(root)
    directory = profile_path(root, spec.profile).parent
    if spec.entity != directory.name:
        raise ValueError("catalog entity must name the source driver directory")
    driver = profile.get("access", {}).get("driver", {})
    declared = driver.get("actions", [])
    if not isinstance(declared, list) or not all(isinstance(a, str) for a in declared):
        raise ValueError("invalid catalog driver action list")
    entry = (directory / driver.get("entry", "driver.py")).resolve(strict=True)
    if not entry.is_relative_to(directory):
        raise ValueError("driver entry escapes its device directory")
    old_path = sys.path[:]
    saved = {name: sys.modules.pop(name, None) for name in ("simlink", "device_driver")}

    def no_startup_rpc(*args, **kwargs):
        raise DeviceError("binding_setup_failed", "Catalog imports must not perform RPC")

    try:
        with _bind_client(_LocalClient(no_startup_rpc, ExecutionContext(read_only=True))):
            module = runpy.run_path(str(entry), run_name="_roboenv_device")
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value
    table, handle = module.get("ACTIONS"), module.get("handle")
    if not isinstance(table, dict) and not callable(handle):
        raise ValueError("catalog driver must export ACTIONS or handle(action, args)")

    def wrap(action):
        def execute(args, execution):
            with _bind_client(_LocalClient(context.public_dispatch, execution)):
                return table[action](args) if isinstance(table, dict) else handle(action, args)

        return execute

    actions = {}
    for action in declared:
        reason = UNSUPPORTED_ACTIONS.get(action)
        if isinstance(table, dict) and action not in table:
            reason = reason or "No implementation is registered for this action."
        actions[action] = Action(reason=reason) if reason else Action(wrap(action))
    return BoundDevice(
        spec.definition(root),
        actions,
        implementation="catalog-bridge",
        validation="implementation-present; hardware semantics not probed",
        binding={"id": "catalog/1", "entity": spec.entity, "backend": context.backend},
    )


class CatalogDeviceService(DeviceService):
    """Legacy construction API. Runtime deployments now compose explicit bindings."""

    def __init__(self, root, dispatch, *, manifest=None):
        self.root = Path(root).resolve(strict=True)
        self.profiles = list_devices(root=self.root)
        context = BindingContext("legacy", "unknown", None, dispatch)
        devices = [
            bind_catalog(
                BindingSpec.parse(name, {"binding": "catalog/1", "entity": name}),
                self.root,
                context,
            )
            for name in sorted(self.profiles)
        ]
        super().__init__(devices, manifest=manifest)

    @classmethod
    def from_environment(cls, dispatch):
        root = os.environ.get("ROBOENV_SERVER_ROBOT")
        if not root:
            return None
        path = os.environ.get("ROBOENV_SERVER_MANIFEST")
        return cls(root, dispatch, manifest=json.loads(Path(path).read_text()) if path else {})
