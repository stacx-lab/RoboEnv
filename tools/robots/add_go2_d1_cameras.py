"""Add official RealSense housings and nominal frames to the Go2–D1 prototype.

Mounts and visual-only bracket placeholders are editable in the JSON config.
Requires xacro. Input models and previous visualizations are never overwritten.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro

REALSENSE_REVISION = "9a11121700cb4780e273e34141f6402fe184321d"


def numbers(values):
    return " ".join(str(value) for value in values)


def add_cameras(model_dir, source, mounts_file, out):
    root = ET.parse(model_dir / "go2_d1.urdf").getroot()
    mounts = json.loads(mounts_file.read_text())
    upstream = json.loads((source / "source.json").read_text())
    if upstream["revision"] != REALSENSE_REVISION:
        raise ValueError("RealSense source revision differs from the reviewed prototype")
    if out.exists():
        raise FileExistsError(out)
    link_names = {link.get("name") for link in root.findall("link")}
    for name, spec in mounts.items():
        if spec["parent"] not in link_names:
            raise ValueError(f"Unknown camera parent: {spec['parent']}")
        if spec["model"] not in {"d405", "d435"}:
            raise ValueError("Unsupported RealSense model")
        if name + "_link" in link_names:
            raise ValueError("Camera already present")
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        for path in source.glob("*.xacro"):
            text = path.read_text().replace(
                "$(find realsense2_description)/urdf/", work.as_posix() + "/"
            )
            (work / path.name).write_text(text)
        wrapper = ET.Element(
            "robot", name="cameras", attrib={"xmlns:xacro": "http://www.ros.org/wiki/xacro"}
        )
        for model in sorted({spec["model"] for spec in mounts.values()}):
            ET.SubElement(wrapper, "xacro:include", filename=str(work / f"_{model}.urdf.xacro"))
        for name, spec in mounts.items():
            sensor = ET.SubElement(
                wrapper,
                f"xacro:sensor_{spec['model']}",
                parent=spec["parent"],
                name=name,
                use_nominal_extrinsics="true",
            )
            ET.SubElement(
                sensor, "origin", xyz=numbers(spec["xyz_m"]), rpy=numbers(spec["rpy_rad"])
            )
        ET.ElementTree(wrapper).write(work / "cameras.xacro")
        expanded = ET.fromstring(xacro.process_file(str(work / "cameras.xacro")).toxml())
    for elem in expanded:
        for mesh in elem.findall(".//mesh"):
            mesh.set("filename", "meshes/realsense/" + Path(mesh.get("filename")).name)
        root.append(elem)
    # Placeholder brackets are explicitly visual-only: no unverified added inertia.
    for name, spec in mounts.items():
        link = ET.SubElement(root, "link", name=name + "_mount_bracket")
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", size=numbers(spec["bracket_size_m"]))
        material = ET.SubElement(visual, "material", name=name + "_bracket_color")
        ET.SubElement(material, "color", rgba="0.12 0.14 0.17 1")
        joint = ET.SubElement(root, "joint", name=name + "_bracket_joint", type="fixed")
        ET.SubElement(joint, "parent", link=spec["parent"])
        ET.SubElement(joint, "child", link=name + "_mount_bracket")
        ET.SubElement(joint, "origin", xyz=numbers(spec["bracket_xyz_m"]), rpy="0 0 0")
    links = [link.get("name") for link in root.findall("link")]
    assert len(links) == len(set(links))
    joints = root.findall("joint")
    assert len([j for j in joints if j.get("type") != "fixed"]) == 20
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in joints}
    assert len(parents) == len(joints)
    assert set(links) - parents.keys() == {"base"}
    for name in links:
        seen = set()
        while name in parents:
            assert name not in seen
            seen.add(name)
            name = parents[name]
        assert name == "base"
    out.mkdir(parents=True)
    shutil.copytree(model_dir / "meshes", out / "meshes")
    mesh_dir = out / "meshes/realsense"
    mesh_dir.mkdir()
    for name in ("d405.stl", "d435.dae"):
        shutil.copy2(source / name, mesh_dir / name)
    ET.indent(root)
    ET.ElementTree(root).write(out / "go2_d1.urdf", encoding="utf-8", xml_declaration=True)
    for name in ("preview_pose.json", "UMI-LICENSE", "D1-source-package.xml"):
        shutil.copy2(model_dir / name, out / name)
    shutil.copy2(source / "LICENSE", out / "RealSense-LICENSE")
    shutil.copy2(mounts_file, out / "camera_mounts.json")
    shutil.copytree(source, out / "realsense_source")
    provenance = json.loads((model_dir / "provenance.json").read_text())
    provenance["cameras"] = {
        **upstream,
        "mounts": mounts,
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.iterdir())
            if path.is_file()
        },
    }
    provenance["warnings"].extend(
        [
            "RealSense source inertias are marked unreliable; not dynamics validation.",
            "Mount brackets are visual placeholders, with no collision or mass model.",
            "Nominal optical frames are included; no simulated camera streams are configured yet.",
        ]
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out / "go2_d1.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--mounts", type=Path, default=Path(__file__).with_name("go2_d1_camera_mounts.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    add_cameras(args.model, args.source, args.mounts, args.out)
