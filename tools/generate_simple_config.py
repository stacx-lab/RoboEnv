"""Generate portable SIMPLE G1 geometry from the pinned MJCF, without simulation."""

import argparse
import ast
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "runtime/src")]
from roboenv.devices.export import driver_access, write_driver  # noqa: E402
from roboenv.provisioning.simple_assets import digest  # noqa: E402
from roboenv.tasks.simple import REPRESENTATIVES, discover_tasks  # noqa: E402
from roboenv_runtime.adapters.simple.profiles import IDENTITY, NAMES, profiles  # noqa: E402
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose  # noqa: E402
from simple_hand_geometry import extract_hand  # noqa: E402


def vector(node, key, default):
    return [float(value) for value in node.get(key, default).split()]


def extract(path):
    tree = ET.parse(path).getroot()
    bodies = {node.get("name"): node for node in tree.iter("body")}
    parent = {child: node for node in tree.iter() for child in node}
    joints = {node.get("name"): node for node in tree.iter("joint") if node.get("name")}

    def fact(name):
        node = joints[name]
        bounds = vector(node, "range", "")
        if len(bounds) != 2:
            raise ValueError(f"Missing SIMPLE joint limits: {name}")
        return {
            "name": name,
            "type": "revolute",
            "limits": {"min": bounds[0], "max": bounds[1]},
            "max_velocity": None,
            "max_effort": None,
        }

    result = {}
    for name in ("waist", *NAMES[1:]):
        hand = name.endswith("hand")
        side = name.split("_")[0]
        base = "pelvis" if name == "waist" else "torso_link"
        tip = "torso_link" if name == "waist" else side + "_wrist_yaw_link"
        if hand:
            names = [
                f"{side}_hand_{finger}_{index}_joint"
                for finger, count in (("thumb", 3), ("index", 2), ("middle", 2))
                for index in range(count)
            ]
            result[name] = {
                "base": tip,
                "tip": tip,
                "mount": IDENTITY,
                "joints": [fact(joint) for joint in names],
                "home": dict.fromkeys(names, 0.0),
                **extract_hand(tree, path, side, names),
            }
            continue
        route, current = [], bodies[tip]
        while current is not bodies[base]:
            route.insert(0, current)
            current = parent[current]
        chain, facts = [], []
        for body in route:
            q = np.asarray(vector(body, "quat", "1 0 0 0"))
            q /= np.linalg.norm(q)
            origin = vector(body, "pos", "0 0 0") + q[[1, 2, 3, 0]].tolist()
            chain.append({"origin": origin})
            for joint in body.findall("joint"):
                joint_name = joint.get("name")
                chain.append(
                    {
                        "origin": IDENTITY,
                        "joint": len(facts),
                        "axis": vector(joint, "axis", "0 0 1"),
                        "anchor": vector(joint, "pos", "0 0 0"),
                        "reference": 0,
                    }
                )
                facts.append(fact(joint_name))
        result[name] = {
            "base": base,
            "tip": tip,
            "chain": chain,
            "joints": facts,
            "home": {joint["name"]: 0.0 for joint in facts},
        }
    return result


def numeric(node):
    """Read only numeric camera literals and arithmetic from the pinned source."""
    if isinstance(node, ast.Constant) and type(node.value) in (float, int):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [numeric(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -numeric(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = numeric(node.left), numeric(node.right)
        return left + right if isinstance(node.op, ast.Add) else left - right
    raise ValueError("Unsupported native camera constant")


def cameras(source):
    # Native eye_in_head mount in the MuJoCo renderer, converted to optical axes.
    tree = ast.parse((source / "src/simple/engines/mujoco.py").read_text())
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"DEFAULT_HEAD_CAM_POSITION", "DEFAULT_HEAD_CAM_ORIENTATION"}:
                constants[name] = numeric(node.value.args[0])
    q = constants["DEFAULT_HEAD_CAM_ORIENTATION"]
    body = matrix(constants["DEFAULT_HEAD_CAM_POSITION"] + [*q[1:], q[0]])
    # SIMPLE camera axes are forward +X, left +Y, up +Z; public optical is +Z forward.
    optical = np.eye(4)
    optical[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    result = {}
    for side, offset in (("left", 0.0), ("right", -0.05)):
        pose = body.copy()
        pose[:3, 3] += [0, offset, 0]
        value = [round(v, 12) for v in matrix_pose(pose @ optical)]
        focal = 640 / (2 * math.tan(math.radians(110) / 2))
        result["camera_head_" + side] = {
            "native": "head_stereo_" + side,
            "width": 640,
            "height": 360,
            "projection": {
                "model": "pinhole",
                "fx": focal,
                "fy": focal,
                "cx": 320.0,
                "cy": 180.0,
                "skew": 0,
            },
            "mount": {
                "parent": "base",
                "parent_frame": "torso",
                "translation_m": value[:3],
                "rotation_xyzw": value[3:],
            },
        }
    return result


def generate(out, facts_out, provenance, *, bindings_out=None):
    source = ROOT / "third_party/SIMPLE"
    robot_xml = ROOT / ".cache/simple/data/robots/g1_sonic/g1_29dof_with_hand.xml"
    model = {
        "parts": extract(robot_xml),
        "cameras": cameras(source),
        "robot_sha256": digest(robot_xml),
        "control": {
            "mode": "motion",
            "body_frame": "pelvis",
            "modes": {
                "locomotion": {
                    "velocity": {
                        "frame": "body",
                        "components": [
                            {
                                "axis": axis,
                                "units": "rad/s" if axis == "wz" else "m/s",
                                "bounds": {"min": -cap, "max": cap},
                            }
                            for axis, cap in (("vx", 0.5), ("vy", 0.5), ("wz", 1.0))
                        ],
                    },
                    "posture": {
                        "reference": "ground",
                        "components": [
                            {
                                "axis": "height",
                                "units": "m",
                                "bounds": {"min": 0.2, "max": 0.74},
                                "default": 0.74,
                            }
                        ],
                    },
                }
            },
        },
    }
    waist = {joint["name"]: joint for joint in model["parts"]["waist"]["joints"]}
    posture = model["control"]["modes"]["locomotion"]["posture"]["components"]
    for axis in ("roll", "pitch", "yaw"):
        posture.append(
            {
                "axis": axis,
                "units": "rad",
                "default": 0.0,
                "bounds": waist[f"waist_{axis}_joint"]["limits"],
            }
        )
    robot = out / "robots/simple/simple-g1"
    robot.mkdir(parents=True, exist_ok=True)
    obsolete = robot / "devices/waist"
    if obsolete.exists():
        shutil.rmtree(obsolete)
    (robot / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "identity": {
                    "id": "g1",
                    "name": "G1",
                    "kind": "humanoid",
                    "model": "Unitree G1 Dex3",
                    "platform": "sim",
                    "source": "introspected",
                },
                "sim": {"simulator": "simple", "source": "introspected"},
                "devices": {"source": "introspected"},
                "safety": {"max_cartesian_velocity": None, "max_joint_velocity": None},
                "meta": {
                    "created_at": "2026-09-14T00:00:00Z",
                    "updated_at": "2026-09-14T00:00:00Z",
                },
            },
            sort_keys=False,
        )
    )
    bindings = {}
    for name, value in profiles(model).items():
        directory = robot / "devices" / name
        directory.mkdir(parents=True, exist_ok=True)
        profile = {
            "name": name,
            "loaded": False,
            **value,
            "meta": {"created_at": "2026-09-14T00:00:00Z", "updated_at": "2026-09-14T00:00:00Z"},
        }
        if name != "world":
            actions = ["read", "load", "unload", "status"]
            if name in NAMES:
                actions += ["write", "stop"]
            if value["type"] == "joint":
                actions += [ "fk", "ik"]
            profile["access"] = driver_access(value["type"], name, actions, "http://127.0.0.1:8123")
            write_driver(directory, value["type"])
        (directory / "profile.yaml").write_text(
            yaml.safe_dump(json.loads(json.dumps(profile)), sort_keys=False)
        )
        kind = {"dexterous_hand": "joint", "rgb": "camera"}.get(value["type"], value["type"])
        bindings[name] = {"binding": f"simple.{kind}/1", "entity": name}
    if bindings_out is not None:
        target = bindings_out / "simple/simple-g1.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "roboenv/bindings/1",
                    "backend": "simple",
                    "embodiment": "g1",
                    "devices": bindings,
                },
                sort_keys=False,
            )
        )
    from dataclasses import asdict

    for task, _ in discover_tasks(ROOT):
        if task.id.split("/")[1] not in REPRESENTATIVES.values():
            continue
        target = out / "tasks" / task.id / "task.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(asdict(task), sort_keys=False))
    facts_out.parent.mkdir(parents=True, exist_ok=True)
    facts_out.write_text(json.dumps(model, indent=2) + "\n")
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "schema": "roboenv/simple-config/1",
                "robot_sha256": model["robot_sha256"],
                "facts_sha256": digest(facts_out),
                "asset_pin_sha256": digest(ROOT / "src/roboenv/assets/simple-g1.json"),
                "source_lock_sha256": digest(ROOT / "upstreams.lock.json"),
                "hand_geometry": {
                    name: {
                        "definition": model["parts"][name]["geometry"]["description"],
                        "sources": model["parts"][name]["geometry_sources"],
                    }
                    for name in ("left_hand", "right_hand")
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "catalog")
    parser.add_argument(
        "--facts-out",
        type=Path,
        default=ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json",
    )
    parser.add_argument(
        "--provenance", type=Path, default=ROOT / "docs/simple-config.provenance.json"
    )
    parser.add_argument("--bindings-out", type=Path, help="Separate internal binding resource root")
    options = parser.parse_args()
    generate(options.out, options.facts_out, options.provenance, bindings_out=options.bindings_out)
