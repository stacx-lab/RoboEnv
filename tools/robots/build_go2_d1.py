"""Build a portable Go2–D1 geometry prototype from locally acquired source assets.

This does not certify actuator limits, mounting hardware, or dynamic stability.
"""

import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

D1_URL = "https://oss-global-cdn.unitree.com/static/90b2525be5d84531ab9814f48b2f86a7.zip"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(umi, d1, out):
    source = umi / "mani-centric-wbc/resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf"
    arm_source = d1 / "urdf/d1_description.urdf"
    root = ET.parse(source).getroot()
    arm = ET.parse(arm_source).getroot()
    root.set("name", "go2_d1")
    # Remove the whole ARX5 subtree, including any attached helper frames.
    removed = {"base_arm_link"}
    joints = root.findall("joint")
    while True:
        expanded = removed | {
            j.find("child").get("link") for j in joints if j.find("parent").get("link") in removed
        }
        if expanded == removed:
            break
        removed = expanded
    for elem in list(root):
        if (elem.tag == "link" and elem.get("name") in removed) or (
            elem.tag == "joint" and elem.find("child").get("link") in removed
        ):
            root.remove(elem)
    sources = {}
    for elem in root.findall(".//mesh"):
        src = umi / "mani-centric-wbc" / elem.get("filename")
        dst = Path("meshes/go2") / src.name
        sources[str(dst)] = src
        elem.set("filename", dst.as_posix())
    for original in arm:
        elem = copy.deepcopy(original)
        if "name" in elem.attrib:
            elem.set("name", "d1_" + elem.get("name"))
        for ref in elem.findall("parent") + elem.findall("child"):
            ref.set("link", "d1_" + ref.get("link"))
        for mimic in elem.findall("mimic"):
            mimic.set("joint", "d1_" + mimic.get("joint"))
        for mesh in elem.findall(".//mesh"):
            src = d1 / mesh.get("filename").removeprefix("package://d1_description/")
            dst = Path("meshes/d1") / src.name
            sources[str(dst)] = src
            mesh.set("filename", dst.as_posix())
        root.append(elem)
    mount = ET.SubElement(root, "joint", name="trunk_d1_joint", type="fixed")
    ET.SubElement(mount, "parent", link="base")
    ET.SubElement(mount, "child", link="d1_base_link")
    ET.SubElement(mount, "origin", xyz="0.085 0 0.094", rpy="0 0 0")
    # Validate one connected, acyclic tree and retain the source actuator metadata.
    names = [link.get("name") for link in root.findall("link")]
    assert len(names) == len(set(names))
    parents = {}
    for j in root.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        assert p in names and c in names and c not in parents
        parents[c] = p
    assert set(names) - parents.keys() == {"base"}
    for name in names:
        visited = set()
        while name in parents:
            assert name not in visited
            visited.add(name)
            name = parents[name]
    moving = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    assert len(moving) == 20  # 12 legs + 6 arm + 2 finger sliders.
    out.mkdir(parents=True, exist_ok=False)
    for dst, src in sources.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dst)
    ET.indent(root)
    ET.ElementTree(root).write(out / "go2_d1.urdf", encoding="utf-8", xml_declaration=True)
    pose = {j.get("name"): 0.0 for j in moving}
    for name in pose:
        if name.endswith("_thigh_joint"):
            pose[name] = 0.8
        elif name.endswith("_calf_joint"):
            pose[name] = -1.6
    pose.update(
        {
            "d1_Joint2": 0.5,
            "d1_Joint3": -0.8,
            "d1_Joint5": 0.3,
            "d1_Joint7_1": 0.015,
            "d1_Joint7_2": -0.015,
        }
    )
    for j in moving:
        limit = j.find("limit")
        assert float(limit.get("lower")) <= pose[j.get("name")] <= float(limit.get("upper"))
    (out / "preview_pose.json").write_text(json.dumps(pose, indent=2) + "\n")
    provenance = {
        "stage": "geometry prototype; not an Isaac Lab environment or validated controller",
        "d1_archive_url": D1_URL,
        "source_urdf_sha256": {"go2_arx5": digest(source), "d1": digest(arm_source)},
        "mount_xyz_m": [0.085, 0, 0.094],
        "mount_rpy_rad": [0, 0, 0],
        "mount_note": "Inherited ARX5 location; no D1 mounting plate modeled or verified.",
        "moving_joints": len(moving),
        "warnings": [
            "D1 source effort and velocity limits are zero; must be resolved before actuation.",
            "Existing Go2-ARX5 policy is not validated for this new morphology.",
        ],
        "mesh_sha256": {dst: digest(src) for dst, src in sources.items()},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(umi / "LICENSE", out / "UMI-LICENSE")
    shutil.copy2(d1 / "package.xml", out / "D1-source-package.xml")
    print(out / "go2_d1.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umi", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.umi, args.d1, args.out)
