"""Reject altered SIMPLE commands, integration state, or native verdict in fresh replay."""

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--case", choices=("commands", "state", "verdict", "gravity"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new receipt")
    record = json.loads(args.record.read_text())
    if args.case == "commands":
        index = next(i for i, event in enumerate(record["commands"]) if event["op"] == "write")
        del record["commands"][index]
    elif args.case == "state":
        record["state"][0] += 0.1
    elif args.case == "gravity":
        metadata = record["initialization"]
        metadata["arm_gravity_compensation"] = not metadata["arm_gravity_compensation"]
    else:
        record["native_success"] = not record["native_success"]
    spec = importlib.util.spec_from_file_location("replay", ROOT / "tools/replay_simple_devices.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        result = module.replay(record)
        rejected = not result["passed"]
    except ValueError as error:
        expected = {"commands": "composed targets differ",
                    "gravity": "replay initialization changed"}
        if args.case not in expected or expected[args.case] not in str(error):
            raise
        rejected, result = True, {"error": str(error)}
    assert rejected, "Altered SIMPLE record was accepted"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"passed": True, "case": args.case, "result": result},
                                  indent=2) + "\n")


if __name__ == "__main__":
    main()
