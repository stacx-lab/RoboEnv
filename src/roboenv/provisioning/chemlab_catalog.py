"""Generate discoverable ChemLab workstations from the pinned release contracts."""

import copy
import json

import yaml
from roboenv.project import project_root
from roboenv.tasks.artifacts import Bundle

from .chemlab_assets import REVISION, asset_root, catalog, resource


def public_capabilities(contract):
    """Visibility is authoritative, including agent-facing scene operations.

    Simulation is a capability kind, not an access level. Clock, plant and truth
    controls remain owner-only even in older permissive contracts. Public fault
    acknowledgement and diagnostics retain their declared agent permissions.
    """
    return {
        name: value
        for name, value in contract["capabilities"].items()
        if value.get("visibility") == "agent"
        and not name.startswith(("sim.", "host.", "truth.", "plant."))
    }


def instrument_facts(row, contract):
    return {
        "asset_id": row["id"],
        "catalog_id": row["chem_id"],
        "name": row["name"],
        "manufacturer": row["maker"],
        "model": row["model"],
        "domain": row["domain_name"],
        "size_m": [float(row[f"size_{axis}_m"]) for axis in "xyz"],
        "release": REVISION,
        "state_slots": {
            key: {field: value[field] for field in ("unit", "value_schema") if field in value}
            for key, value in contract["state_slots"].items()
            if value.get("visibility") == "agent"
        },
    }


def generate(destination, directory=None, *, project=None):
    """Export all instruments without importing their code or inventing task goals."""
    root = asset_root(directory)
    template = project_root(project) / "catalog/robots/isaaclab/isaaclab-chemlab-instruments"
    workspace = yaml.safe_load((template / "workspace.yaml").read_text())
    profile = yaml.safe_load((template / "devices/instrument/profile.yaml").read_text())
    world = yaml.safe_load((template / "devices/world/profile.yaml").read_text())
    bundle = Bundle(destination)
    index = []
    for asset, row in catalog(root).items():
        contract = json.loads(resource(root, row["contract"]).read_text())
        identifier = "chemlab-" + asset.removeprefix("instrument.").replace("_", "-")
        robot = f"isaaclab/{identifier}"
        base = f"robots/{robot}"
        facts = instrument_facts(row, contract)
        capabilities = public_capabilities(contract)
        item = copy.deepcopy(profile)
        item["intrinsics"] = {"source": "user", "instrument": facts}
        item["access"]["driver"]["actions"] = [
            "read",
            "load",
            "unload",
            "status",
            *(key.replace(".", "_") for key in capabilities),
        ]
        manifest = copy.deepcopy(workspace)
        manifest["identity"].update(id=identifier, name=row["name"], model=row["model"])
        bundle.add_yaml(f"{base}/workspace.yaml", manifest)
        bundle.add_yaml(f"{base}/devices/instrument/profile.yaml", item)
        bundle.add_yaml(f"{base}/devices/world/profile.yaml", world)
        for name in ("instrument/driver.py", "world/driver.py", "_roboenv.py", "_camera.py"):
            bundle.add(f"{base}/devices/{name}", (template / "devices" / name).read_text())
        bundle.add_yaml(
            f"{base}/bindings.yaml",
            {
                "schema_version": "roboenv/bindings/1",
                "backend": "isaaclab",
                "embodiment": "chemlab_instruments",
                "devices": {
                    name: {"binding": "isaaclab.chemlab/1", "entity": name}
                    for name in ("world", "instrument")
                },
            },
        )
        # These are launchable exploration environments, not scored benchmark tasks.
        bundle.add_yaml(
            f"tasks/{robot}/task.yaml",
            {
                "id": robot,
                "robot": robot,
                "instruction": f"Explore the {row['maker']} {row['model']} instrument through its "
                "declared commands and observations. No success goal is assigned.",
                "seed": 0,
                "budget_mode": "practice",
                "execution_mode": "realtime",
                "stop_on_success": False,
                "backend_kwargs": {
                    "task_name": "chemlab-instruments",
                    "task_spec": {
                        "instruments": {"instrument": {"asset": asset}},
                        "goals": [],
                    },
                },
            },
        )
        entry = {
            **facts,
            "robot": robot,
            "environment": robot,
            "capabilities": {
                name: {
                    field: cap[field]
                    for field in ("availability", "kind", "input_schema")
                    if field in cap
                }
                for name, cap in capabilities.items()
            },
        }
        bundle.add(f"{base}/instrument.json", json.dumps(entry, indent=2) + "\n")
        index.append(
            {
                key: entry[key]
                for key in (
                    "asset_id",
                    "catalog_id",
                    "name",
                    "manufacturer",
                    "model",
                    "domain",
                    "robot",
                )
            }
        )
    bundle.add("instruments.json", json.dumps(index, indent=2) + "\n")
    return bundle.commit(
        {
            "schema": "roboenv/chemlab-catalog/1",
            "revision": REVISION,
            "instruments": len(index),
            "scored_tasks": 0,
        }
    )
