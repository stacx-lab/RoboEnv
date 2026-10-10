"""Offline MJCF bundle preparation; no MuJoCo or Isaac imports in this module."""

import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_model(source):
    """Retain ABC hardware/station geometry; export independent fixed-base arms."""
    root = ET.parse(source).getroot()
    result = {}
    for name in ("left", "right", "station"):
        tree = copy.deepcopy(root)
        tree.set("model", name)
        world = tree.find("worldbody")
        for element in list(world):
            keep = element.tag == "body" and element.get("name") == name + "_arm"
            if name == "station":
                keep = not (
                    element.tag == "body"
                    and element.get("name")
                    in {
                        "left_arm",
                        "right_arm",
                        "overhead_camera",
                        "left_side_camera",
                        "right_side_camera",
                    }
                )
                # A finite tabletop replaces MuJoCo's infinite collision plane.
                if element.get("name") in {"floor", "floor_collision", "table_plane"}:
                    keep = False
            if not keep:
                world.remove(element)
        if name != "station":
            world.find("body").set("pos", "0 0 0")
        else:
            ET.SubElement(
                world,
                "geom",
                name="table_collision",
                type="box",
                size="0.2975 0.65 0.005",
                pos="0.6 0 0.745",
                rgba="0 0 0 0",
            )
        for parent in tree.iter():
            for child in list(parent):
                if child.tag in {"camera", "light"}:
                    parent.remove(child)
        # Coupling is authored explicitly as a PhysX mimic joint after conversion.
        for tag in ("equality", "keyframe", "sensor", "tendon", "contact"):
            for element in tree.findall(tag):
                tree.remove(element)
        actuator = tree.find("actuator")
        if actuator is not None:
            for element in list(actuator):
                if name == "station" or not element.get("joint", "").startswith(name + "_"):
                    actuator.remove(element)
        result[name] = tree
    return result


def rgb_profiles(profiles):
    """Reuse ABC geometry with the RGB-only sensors implemented by Isaac YAM."""
    result = copy.deepcopy(profiles)
    for profile in result.values():
        if profile["type"] == "rgbd":
            profile["type"] = "rgb"
            profile["intrinsics"].pop("depth")
    return result


def prepare(source, profiles, out):
    """Copy meshes into a portable, checksummed owner-only bundle."""
    source, out = Path(source).resolve(), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to replace an existing asset bundle: {out}")
    trees = split_model(source)
    compiler = ET.parse(source).getroot().find("compiler")
    meshdir = source.parent / compiler.get("meshdir", "")
    # Check dependencies before creating output, so missing assets are actionable.
    meshes = {element.get("file") for element in trees["left"].findall("asset/mesh")}
    for name in meshes:
        if not (meshdir / name).is_file():
            raise FileNotFoundError(f"ABC mesh missing: {meshdir / name}; prepare ABC assets first")
    out.mkdir(parents=True)
    (out / "meshes").mkdir()
    for name in sorted(meshes):
        shutil.copy2(meshdir / name, out / "meshes" / Path(name).name)
    for name, tree in trees.items():
        tree.find("compiler").set("meshdir", "meshes")
        for element in tree.findall("asset/mesh"):
            element.set("file", Path(element.get("file")).name)
        ET.indent(tree)
        ET.ElementTree(tree).write(out / f"{name}.xml", encoding="utf-8", xml_declaration=True)
    facts = {
        "schema": "roboenv/yam-assets/1",
        "source_mjcf_sha256": sha256(source),
        "physical_profiles": rgb_profiles(profiles),
        "files": {
            p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }
    (out / "assets.json").write_text(json.dumps(facts, sort_keys=True, indent=2) + "\n")
    return out / "assets.json"


def load_manifest(path):
    path = Path(path).resolve()
    facts = json.loads(path.read_text())
    if facts.get("schema") != "roboenv/yam-assets/1":
        raise ValueError("Unsupported YAM asset manifest")
    for name, expected in facts["files"].items():
        asset = (path.parent / name).resolve()
        if not asset.is_relative_to(path.parent) or sha256(asset) != expected:
            raise ValueError(f"Changed or unsafe YAM asset: {name}")
    if not {"left.xml", "right.xml", "station.xml"} <= facts["files"].keys():
        raise ValueError("YAM bundle is missing its MJCF descriptions")
    return facts
