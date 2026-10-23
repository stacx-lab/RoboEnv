"""Prepare the independently retained Booster T2 model, controller and scene depot."""

import argparse
import json
from pathlib import Path

from roboenv.provisioning.booster_t2_assets import prepare_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, help="Verified portable T2 asset depot")
    parser.add_argument(
        "--download", action="store_true", help="Download missing checksum-pinned assets"
    )
    parser.add_argument(
        "--robot-only", action="store_true", help="Prepare flat-ground practice only"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_assets(
                args.project,
                source=args.source,
                scenes=not args.robot_only,
                articulated=not args.robot_only,
                download=args.download,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
