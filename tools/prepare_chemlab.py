"""Acquire the pinned complete ChemLab release or verify an existing copy."""

import argparse
import json

from roboenv.provisioning.chemlab_assets import asset_root, prepare, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    receipt = (verify_assets if args.verify else prepare)(args.directory)
    print(
        json.dumps(
            {
                "directory": str(asset_root(args.directory)),
                "revision": receipt["revision"],
                "instruments": len(receipt["instruments"]),
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
