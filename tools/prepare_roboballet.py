"""Prepare only the pinned RoboBallet assets, without starting Isaac Lab."""

import argparse
import json
from pathlib import Path

from roboenv.provisioning.roboballet_assets import prepare_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare_assets(args.project), indent=2))


if __name__ == "__main__":
    main()
