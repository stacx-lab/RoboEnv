"""Prepare an explicitly approximate Go2–D1 practice dynamics bundle.

Consumes the reviewed camera-equipped robot bundle; never rewrites source meshes.
"""

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/src"))
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.limits import (  # noqa: E402
    ARM_EFFORT_LIMITS,
    ARM_VELOCITY_LIMIT_SIM,
    D1_SPECIFICATION_URL,
)


def prepare(source, out):
    import trimesh

    if out.exists():
        raise FileExistsError(out)
    root = ET.parse(source / "go2_d1.urdf").getroot()
    out.mkdir(parents=True)
    meshes = out / "meshes"
    meshes.mkdir()
    converted = {}
    for mesh in root.findall(".//mesh"):
        name = mesh.get("filename")
        if name not in converted:
            src = source / name
            dest = meshes / (hashlib.sha256(name.encode()).hexdigest()[:12] + ".obj")
            scene = trimesh.load(src, force="scene")
            scene.to_geometry().export(dest)
            converted[name] = dest.relative_to(out).as_posix()
        mesh.set("filename", converted[name])
    assumptions = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name.startswith("d1_Joint"):
            limit = joint.find("limit")
            finger = name.startswith("d1_Joint7")
            effort = 40 if finger else ARM_EFFORT_LIMITS[name]
            velocity = 0.1 if finger else ARM_VELOCITY_LIMIT_SIM
            limit.set("effort", str(effort))
            limit.set("velocity", str(velocity))
            assumptions[name] = {
                "effort": effort,
                "velocity": velocity if finger else None,
                "source": "practice approximation" if finger else D1_SPECIFICATION_URL,
                **({} if finger else {"velocity_policy": "uncapped simulation"}),
            }
    for name, size in [
        ("wrist_d405_link", [0.023, 0.042, 0.042]),
        ("head_d435_link", [0.02505, 0.09, 0.025]),
    ]:
        inertia = root.find(f"link[@name='{name}']/inertial")
        mass = float(inertia.find("mass").get("value"))
        values = [
            mass * (size[1] ** 2 + size[2] ** 2) / 12,
            mass * (size[0] ** 2 + size[2] ** 2) / 12,
            mass * (size[0] ** 2 + size[1] ** 2) / 12,
        ]
        inertia.find("inertia").attrib.update(
            dict(zip(["ixx", "iyy", "izz"], map(str, values), strict=True))
        )
        assumptions[name] = {"inertia": "uniform box using source mass and housing dimensions"}
    # Frames with tiny source inertias are fixed and merged by the importer.
    ET.indent(root)
    ET.ElementTree(root).write(out / "robot.urdf", encoding="utf-8", xml_declaration=True)
    shutil.copy2(source / "preview_pose.json", out / "home.json")
    home = json.loads((out / "home.json").read_text())
    # A folded arm reduces COM offset during first locomotion validation.
    home.update({"d1_Joint2": -0.65, "d1_Joint3": 1.2, "d1_Joint5": -0.55})
    (out / "home.json").write_text(json.dumps(home, indent=2) + "\n")
    manifest = {
        "schema": "roboenv/go2-d1-assets/1",
        "practice_assumptions": assumptions,
        "source_urdf_sha256": hashlib.sha256((source / "go2_d1.urdf").read_bytes()).hexdigest(),
        "files": {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    (out / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "assets.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(".cache/isaaclab/go2-d1"))
    a = p.parse_args()
    prepare(a.source, a.out)
