"""Native scene coverage for an explicit slice of the complete ChemLab catalog."""

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from roboenv.provisioning.chemlab_assets import catalog
from roboenv_runtime.adapters.isaaclab.common.spectator import look_at
from roboenv_runtime.adapters.isaaclab.tasks.chemlab.environment import ChemLabEnvironment
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer


def run(args):
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=False)
    rows = list(catalog(args.directory).values())[args.start : args.start + args.count]
    instruments = {f"instrument_{i}": {"asset": row["id"]} for i, row in enumerate(rows)}
    profiles = {
        name: {
            "extrinsics": {
                "mount": {
                    "parent": "world",
                    "parent_frame": "world",
                    "translation_m": [i * 8.0, 0.0, 0.0],
                    "rotation_xyzw": [0, 0, 0, 1],
                }
            }
        }
        for i, name in enumerate(instruments)
    }
    environment = ChemLabEnvironment(
        task_spec={"instruments": instruments, "goals": []},
        device_profiles=profiles,
        asset_directory=args.directory,
    )
    result = {
        "assets": [row["id"] for row in rows],
        "passed": False,
        "scope": "Native import, geometry and command projection; owner validation clock",
    }
    try:
        environment.reset(0)
        initial = environment.record()
        views = []
        for name, (model, _, pose) in environment.native.models.items():
            center = model.scene.bounds.mean(axis=0) + pose[:3, 3]
            extent = max(float(np.ptp(model.scene.bounds, axis=0).max()), 0.1)
            views.append(
                ViewSpec(
                    name,
                    width=640,
                    height=480,
                    pose=look_at(center + extent * np.array([0.55, -1.5, 0.75]), center),
                )
            )
        adapter = SimpleNamespace(environment=environment, name="isaaclab")
        renderer = create_spectator_renderer(adapter, views)
        for name, pixels in renderer.capture().items():
            Image.fromarray(pixels).save(args.out / f"{name}-initial.png")
        bringup = json.loads((args.directory / "examples/panel_instruments.json").read_text())[
            "instruments"
        ]
        commands = {}
        for name, instrument in environment.instruments.items():
            commands[name] = []
            asset = instrument.row["id"].removeprefix("instrument.")
            for step in bringup.get(asset, {}).get("bringup", []):
                if "advance_s" in step:
                    # Long thermal equilibration belongs to task acceptance.
                    # This probe only verifies the finite shared scene update.
                    continue
                cap = step["capability"]
                arguments = step.get("arguments", {})
                if arguments == "contract_example":
                    arguments = instrument.contract["capabilities"][cap].get(
                        "example_arguments", {}
                    )
                response = instrument.invoke(cap, arguments, owner=step.get("evaluator", False))
                commands[name].append(
                    {
                        "capability": cap,
                        "status": response.get("status"),
                        "error": response.get("error"),
                    }
                )
            if args.commands:
                for cap, spec in instrument.capabilities.items():
                    if spec.get("availability") != "implemented" or "example_arguments" not in spec:
                        continue
                    response = instrument.invoke(cap, spec["example_arguments"])
                    commands[name].append(
                        {
                            "capability": cap,
                            "status": response.get("status"),
                            "error": response.get("error"),
                        }
                    )
            environment._project()
            environment.native.sync()
        for _ in range(3):
            environment.tick()
        for name, pixels in renderer.capture().items():
            if pixels.std() < 1:
                raise AssertionError(f"Empty native image: {name}")
            Image.fromarray(pixels).save(args.out / f"{name}-final.png")
        final = environment.record()
        result.update(
            passed=True,
            sources=final["sources"],
            commands=commands,
            initial_scene=initial["scene_evidence"],
            final_scene=final["scene_evidence"],
            mappings=final["scene_mappings"],
        )
        renderer.close()
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        import isaaclab.sim as sim_utils

        sim = sim_utils.SimulationContext.instance()
        if sim is not None:
            sim.stage.Flatten().Export(str(args.out / "failed-stage.usda"))
    finally:
        (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"passed": result["passed"], "assets": result["assets"]}), flush=True)
        try:
            if not result["passed"]:
                raise RuntimeError(result["error"])
        finally:
            environment.close()
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--commands", action="store_true")
    run(parser.parse_args())
