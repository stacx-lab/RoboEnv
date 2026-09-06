"""Registered device bindings for the five supported simulator backends."""

import importlib
import json
from importlib.resources import files

from roboenv.devices.contract import UNSUPPORTED_ACTIONS
from roboenv.devices.model import Action, BoundDevice

from .backend import BackendControllers


def binding_index():
    return json.loads(files("roboenv_runtime.bindings.backends").joinpath("index.json").read_text())


def bind_simulator_device(spec, root, context):
    if spec.binding != f"{context.backend}.device/1":
        raise ValueError("simulator binding/backend mismatch")
    if spec.config:
        raise ValueError("This binding accepts no unrecognized backend configuration")
    key = f"{context.backend}:{context.embodiment}:{spec.entity}"
    record = binding_index().get(key)
    if record is None:
        raise ValueError(f"Unknown device slot for this backend/embodiment: {spec.entity!r}")
    module = importlib.import_module("roboenv_runtime.bindings.backends." + record["module"])
    backend = BackendControllers(context)
    handlers = module.create_actions(backend)
    profile = spec.resolved_profile(root)
    declared = profile.get("access", {}).get("driver", {}).get("actions", [])
    actions = {}
    for name in sorted(set(handlers) | set(declared)):
        reason = UNSUPPORTED_ACTIONS.get(name)
        missing = sorted(set(module.REQUIRES.get(name, [])) - set(backend.tools))
        if missing:
            reason = "Backend controllers unavailable: " + ", ".join(missing)
        handler = handlers.get(name)
        if handler is None:
            reason = reason or "This backend does not implement this action"
        if reason:
            actions[name] = Action(reason=reason)
        else:
            actions[name] = Action(
                lambda args, execution, handler=handler: backend.execute(handler, args, execution)
            )
    return BoundDevice(
        spec.definition(root),
        actions,
        implementation="backend-controller-binding",
        validation="backend-slot-resolved; controller availability checked",
        binding={
            "id": spec.binding,
            "backend": context.backend,
            "entity": spec.entity,
            "source_sha256": record.get("source_sha256"),
        },
    )
