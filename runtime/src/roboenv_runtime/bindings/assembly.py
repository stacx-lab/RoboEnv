"""Compose owner configuration with a live world and the neutral device service."""

import json
from pathlib import Path

from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.bindings.spec import BINDING_CONTRACT, BindingPlan

from . import default_registry


def assemble_devices(
    root, world, *, manifest_path=None, registry=None
):
    root = Path(root).resolve(strict=True)
    plan = BindingPlan.load(root, backend=world.backend)
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    manifest["binding_contract"] = BINDING_CONTRACT
    manifest["binding_mode"] = "explicit"
    manifest["execution_mode"] = world.mode
    context = BindingContext(
        world.backend,
        world.embodiment,
        world.native,
        world=world,
    )
    return (registry or default_registry()).assemble(
        plan,
        root,
        context,
        manifest=manifest,
        execution_lock=world.lock,
        motion_scheduler=world.schedule,
        resolve=world.resolve,
        flush=world.poll_motions,
    )
