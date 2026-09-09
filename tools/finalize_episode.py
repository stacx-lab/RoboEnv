"""Freeze a locally owned episode and save private replay evidence.

Uses the existing owner socket protocol. Never copy the output into an agent
workspace: it contains initial-state identity, seed, final state, and verdict.
"""

import argparse
import json
import os
from pathlib import Path

from roboenv.client import Client


def finalize(run_record, destination):
    record = json.loads(Path(run_record).read_text())
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use a fresh output directory; existing evidence is preserved")
    destination.mkdir(parents=True, mode=0o700)
    client = Client(record["socket"], session=record["session"], generation=record["generation"])
    client.allowed_ops = client.allowed_ops | {"submit"}
    verdict = client.request("submit")
    observation = client.request("observe")
    for name, value in (("verdict.json", verdict), ("owner-record.json", observation)):
        path = destination / name
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    return {"saved": str(destination), "verdict": verdict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_record, args.out), indent=2))
