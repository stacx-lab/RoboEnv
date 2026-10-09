"""Freeze the paper's 8-arm evaluation geometry and all ten 40-target sets.

Run in the author's geometry environment: mujoco 3.3.4, numpy 1.26.4,
scipy 1.12.0 and vhacdx 0.0.8.post2. The output contains no solutions.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified directory containing roboballet/")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from roboballet.environments import geometry
    from roboballet.environments.mujoco_simulator import MujocoWorld, MujocoWorldConfig

    source = args.source / "roboballet"
    world = MujocoWorld(MujocoWorldConfig(
        xml_path=str((source / "data/mujoco_world/8_pandas_world.xml").resolve())))
    np.random.seed(0)
    obstacles = []
    for mesh in world.GetMeshes("workpiece"):
        for vertices, _ in geometry.convex_decomposition(mesh.vertices, mesh.faces, max_parts=30):
            obb = geometry.obb_from_points(vertices)
            # evaluator.get_predefined_targets raises 8-arm targets by 0.6 m.
            # Apply that same documented transform to the workpiece geometry.
            center = obb.pose[:3, 3] + [0, 0, 0.6]
            obstacles.append({"spans": obb.spans.tolist(), "centroid": center.tolist(),
                              "rotation": obb.pose[:3, :3].tolist()})
    text = (source / "data/paper_test_tasksets/test_40targets.pbtxt").read_text()
    sets = []
    for block in text.split("target_sets {")[1:]:
        targets = []
        pattern = r"targets\s*{\s*position\s*{(.*?)}\s*rotation\s*{(.*?)}\s*}"
        for pos, rot in re.findall(pattern, block, re.S):
            def values(value):
                return np.array(re.findall(r"float_values:\s*([^\s]+)", value), dtype=np.float32)
            position = values(pos) + np.array([0, 0, 0.6], dtype=np.float32)
            targets.append({"position": position.tolist(),
                            "rotation": values(rot).reshape(3, 3).tolist()})
        assert len(targets) == 40
        sets.append(targets)
    assert len(sets) == 10 and len(obstacles) == 30
    paths = ["data/mujoco_world/8_pandas_world.xml",
             "data/paper_test_tasksets/test_40targets.pbtxt",
             "environments/geometry.py", "environments/mujoco_simulator.py"]
    content = {"schema": "roboenv/roboballet-scene/1", "source_doi": "10.5281/zenodo.16546403",
               "source_files": {
                   p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths},
               "geometry_seed": 0, "workpiece_translation_m": [0, 0, 0.6],
               "obstacles": obstacles, "target_sets": sets}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(content, separators=(",", ":")) + "\n")
    print(json.dumps({"obstacles": len(obstacles), "target_sets": len(sets),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
