"""Generate complete ABC rigs from task-free native models."""

import argparse
import json
from pathlib import Path

import yaml
from generate_native_rigs import write_rig
from roboenv.devices.export import driver_access, write_driver
from roboenv.tasks.abc import entries
from roboenv_runtime.adapters.abc.backend import SOURCE
from roboenv_runtime.adapters.abc.rig import describe, rig_id, task_free_model


def generate(out, *, bindings_out=None, mapping_out=None):
    from abc_sim.task_registry import get_task_scene_xml

    rigs = {}
    emitted = set()
    for scene in sorted({entry["env_task"] for entry in entries(SOURCE)}):
        facts = describe(task_free_model(get_task_scene_xml(scene)))
        fingerprint = rig_id(facts)
        if emitted and fingerprint not in emitted:
            raise ValueError("ABC scenes must share the same robot geometry")
        name = "abc-yam"
        rigs[scene] = name
        if fingerprint not in emitted:
            write_rig(out, name, "abc", "yam", facts, bindings_out=bindings_out)
            world = Path(out) / "robots/abc" / name / "devices/world"
            profile = yaml.safe_load((world / "profile.yaml").read_text())
            profile["access"] = driver_access(
                "world",
                "world",
                ["read", "wait", "load", "unload", "status"],
                "http://127.0.0.1:8123",
            )
            (world / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
            write_driver(world, "world")
            emitted.add(fingerprint)
    if mapping_out:
        Path(mapping_out).write_text(json.dumps(rigs, indent=2, sort_keys=True) + "\n")
    return {"rigs": len(emitted), "scenes": len(rigs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bindings-out", type=Path)
    parser.add_argument("--mapping-out", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(generate(args.out, bindings_out=args.bindings_out, mapping_out=args.mapping_out))
    )
