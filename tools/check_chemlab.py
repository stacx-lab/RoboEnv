"""Audit every pinned instrument host; optionally load every URDF without a GPU."""

import argparse
import json
from collections import Counter
from pathlib import Path

from roboenv.provisioning.chemlab_assets import asset_root, catalog, verify_assets
from roboenv.validation import action_validator
from roboenv_runtime.adapters.isaaclab.tasks.chemlab.geometry import (
    binding_report,
    component_links,
    joint_bindings,
    panel_spec,
    screen_quad,
)
from roboenv_runtime.adapters.isaaclab.tasks.chemlab.instrument import Instrument
from roboenv_runtime.bindings.backends.isaaclab.chemlab import action_name


def audit(directory=None, *, geometry=False):
    root = asset_root(directory)
    receipt = verify_assets(root)
    bringup = json.loads((root / "examples/panel_instruments.json").read_text())["instruments"]
    results = []
    for name, row in catalog(root).items():
        instrument = None
        try:
            instrument = Instrument(root, row, "instrument")
            names = [action_name(capability) for capability in instrument.capabilities]
            if len(set(names)) != len(names):
                raise ValueError("Capability aliases collide")
            for capability in instrument.capabilities.values():
                action_validator(capability["input_schema"])
            public = instrument.read()
            assert set(public) <= {
                key
                for key, slot in instrument.contract["state_slots"].items()
                if slot["visibility"] == "agent"
            }
            instrument.read(owner=True)
            instrument.advance(0.1)
            instrument.read()
            details = {
                "capabilities": dict(
                    Counter(
                        cap.get("availability", "unknown")
                        for cap in instrument.capabilities.values()
                    )
                ),
                "channels": dict(
                    Counter(
                        binding["channel"]
                        for binding in instrument.contract.get("model_bindings", {}).values()
                    )
                ),
            }
            details["bringup"] = []
            for step in bringup.get(name.removeprefix("instrument."), {}).get("bringup", []):
                if "advance_s" in step:
                    remaining = step["advance_s"]
                    while remaining > 1e-9:
                        dt = min(remaining, 1.0)
                        instrument.advance(dt)
                        remaining -= dt
                    continue
                arguments = step.get("arguments", {})
                if arguments == "contract_example":
                    arguments = instrument.contract["capabilities"][step["capability"]].get(
                        "example_arguments", {}
                    )
                try:
                    response = instrument.invoke(
                        step["capability"], arguments, owner=step.get("evaluator", False)
                    )
                    details["bringup"].append(
                        {"capability": step["capability"], "status": response.get("status")}
                    )
                except Exception as error:
                    details["bringup"].append(
                        {"capability": step["capability"], "error": str(error)}
                    )
            if geometry:
                from yourdfpy import URDF

                model = URDF.load(str(root / row["urdf"]))
                if not model.scene.geometry:
                    raise ValueError("Empty instrument geometry")
                panels, missing = [], {}
                for display, spec in panel_spec(instrument).get("displays", {}).items():
                    links = component_links(instrument.contract, spec.get("component_id"))
                    nodes = [
                        node
                        for node in model.scene.graph.nodes_geometry
                        if model.scene.graph.transforms.parents[node] in links
                    ]
                    try:
                        if not nodes:
                            raise ValueError("No URDF display component")
                        meshes = [
                            model.scene.geometry[model.scene.graph[node][1]] for node in nodes
                        ]
                        screen_quad(max(meshes, key=lambda mesh: mesh.area))
                        response = instrument.invoke("panel.capture", {"display_id": display})
                        if response.get("status") != "succeeded":
                            raise ValueError(
                                "Native panel capture failed: "
                                + str(response.get("error", {}).get("code"))
                            )
                        panels.append(spec["component_id"])
                    except Exception as error:
                        missing[display] = str(error)
                details["binding_plan"] = binding_report(
                    instrument, joint_bindings(instrument, model), panels
                )
                details["panel_components"] = panels
                details["panel_failures"] = missing
            results.append({"asset": name, "passed": True, **details})
        except Exception as error:
            results.append({"asset": name, "passed": False, "error": str(error)})
        finally:
            if instrument is not None:
                instrument.close()
    return {
        "revision": receipt["revision"],
        "geometry": geometry,
        "scope": (
            "Initialization/read/clock/schema/geometry/panel audit; "
            "not exhaustive capability execution or GPU validation"
        ),
        "runtime_boot_passed": all(item["passed"] for item in results),
        "panels_complete": all(
            item["passed"] and not item.get("panel_failures") for item in results
        ),
        "integration_complete": False,
        "instruments": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, geometry=args.geometry)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runtime_boot_passed": result["runtime_boot_passed"],
                "panels_complete": result["panels_complete"],
                "instruments": len(result["instruments"]),
            }
        )
    )
    if not result["runtime_boot_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
