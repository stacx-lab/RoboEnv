"""Audit every pinned instrument host; optionally load every URDF without a GPU."""

import argparse
import json
from pathlib import Path

from roboenv.provisioning.chemlab_assets import asset_root, catalog, verify_assets
from roboenv.validation import action_validator
from roboenv_runtime.adapters.isaaclab.tasks.chemlab.instrument import Instrument
from roboenv_runtime.bindings.backends.isaaclab.chemlab import action_name


def audit(directory=None, *, geometry=False):
    root = asset_root(directory)
    receipt = verify_assets(root)
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
            if geometry:
                from yourdfpy import URDF

                model = URDF.load(str(root / row["urdf"]))
                if not model.scene.geometry:
                    raise ValueError("Empty instrument geometry")
            results.append({"asset": name, "passed": True})
        except Exception as error:
            results.append({"asset": name, "passed": False, "error": str(error)})
        finally:
            if instrument is not None:
                instrument.close()
    return {
        "revision": receipt["revision"],
        "geometry": geometry,
        "passed": all(item["passed"] for item in results),
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
    print(json.dumps({"passed": result["passed"], "instruments": len(result["instruments"])}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
