"""Compose owner configuration with a live world and the neutral device service."""

import json
from pathlib import Path

from roboenv.bindings.registry import BindingContext
from roboenv.bindings.spec import BINDING_CONTRACT, BindingPlan

from . import default_registry


def assemble_devices(
    root, world, dispatch, *, manifest_path=None, registry=None, public_tools=None
):
    root = Path(root).resolve(strict=True)
    plan = BindingPlan.load(root, backend=world.backend)
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else {}
    manifest["binding_contract"] = BINDING_CONTRACT
    manifest["binding_mode"] = "implicit-legacy" if plan.implicit_legacy else "explicit"
    context = BindingContext(world.backend, world.embodiment, world.native, dispatch, public_tools)
    return (registry or default_registry()).assemble(
        plan, root, context, manifest=manifest, execution_lock=world.lock
    )
