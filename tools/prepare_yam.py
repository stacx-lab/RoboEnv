"""Prepare the ABC-derived Isaac YAM MJCF bundle without starting either simulator.

USD conversion is cached on first Isaac launch. Download ABC assets separately.
"""

import argparse
from pathlib import Path

import yaml
from roboenv_runtime.adapters.isaaclab.robots.yam.assets import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "third_party/abc/abc_sim/models/yam_bimanual_empty.xml",
    )
    parser.add_argument("--out", type=Path, default=ROOT / ".cache/isaaclab/yam")
    args = parser.parse_args()
    profiles = {
        p.parent.name: {
            k: v
            for k, v in yaml.safe_load(p.read_text()).items()
            if k in {"type", "intrinsics", "extrinsics"}
        }
        for p in (ROOT / "catalog/robots/abc/abc-yam/devices").glob("*/profile.yaml")
    }
    print(prepare(args.source, profiles, args.out))


if __name__ == "__main__":
    main()
