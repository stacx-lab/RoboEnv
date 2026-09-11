"""Freeze a Harbor-produced workspace against its recorded Kinex bundle identity."""

import argparse
import json
from pathlib import Path

from artifacts import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--identity", type=Path, required=True,
                        help="agent/bundle-identity.json from the source Harbor trial")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tool", default="solve_task")
    parser.add_argument("--args", type=json.loads, default={})
    args = parser.parse_args()
    identity = {"kinex_sha256": json.loads(args.identity.read_text())["kinex_sha256"]}
    print(json.dumps(freeze(args.workspace, args.out, identity=identity,
                           tool=args.tool, arguments=args.args), indent=2))


if __name__ == "__main__":
    main()
