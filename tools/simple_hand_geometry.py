"""Extract portable finger chains and nominal references from pinned SIMPLE assets."""

import struct
from pathlib import Path

import numpy as np
from roboenv.provisioning.simple_assets import digest
from roboenv_runtime.adapters.urdf_geometry import matrix

IDENTITY = [0, 0, 0, 0, 0, 0, 1]


def vector(node, key, default):
    return [float(value) for value in node.get(key, default).split()]


def transform(node):
    if any(key in node.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("SIMPLE geometry requires explicit quaternion orientation")
    q = np.asarray(vector(node, "quat", "1 0 0 0"))
    q /= np.linalg.norm(q)
    return vector(node, "pos", "0 0 0") + q[[1, 2, 3, 0]].tolist()


def extract_hand(tree, path, side, names):
    bodies = {node.get("name"): node for node in tree.iter("body")}
    parents = {child: node for node in tree.iter() for child in node}
    meshes = {node.get("name"): node for node in tree.iter("mesh")}
    base = bodies[f"{side}_wrist_yaw_link"]
    meshdir = path.parent / tree.find("compiler").get("meshdir", ".")
    palm = next(g for g in base.findall("geom") if g.get("mesh") == f"{side}_hand_palm_link")
    fingers, frame_bodies, sources, home_tips = {}, {}, {}, {}
    for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)):
        body_name = f"{side}_hand_{finger}_{count - 1}_link"
        terminal = bodies[body_name]
        route, current = [], terminal
        while current is not base:
            route.insert(0, current)
            current = parents[current]
        chain, selected = [], []
        for body in route:
            chain.append({"origin": transform(body)})
            for joint in body.findall("joint"):
                if joint.get("name") not in names:
                    raise ValueError("Finger chain contains a non-hand joint")
                chain.append({"origin": IDENTITY, "joint": len(selected),
                              "axis": vector(joint, "axis", "0 0 1"),
                              "anchor": vector(joint, "pos", "0 0 0"),
                              "reference": float(joint.get("ref", "0"))})
                selected.append(joint.get("name"))
        # The terminal mesh is the native collision geometry, not its visual duplicate.
        geom = next(g for g in terminal.findall("geom") if g.get("mesh")
                    and g.get("contype", "1") != "0")
        mesh = meshes[geom.get("mesh")]
        if any(key in mesh.attrib for key in ("refpos", "refquat")):
            raise ValueError("Unsupported SIMPLE mesh reference transform")
        mesh_path = meshdir / mesh.get("file")
        raw = mesh_path.read_bytes()
        triangles = struct.unpack_from("<I", raw, 80)[0]
        if not triangles or len(raw) != 84 + triangles * 50:
            raise ValueError("Expected binary STL for SIMPLE fingertip geometry")
        vertices = np.asarray([struct.unpack_from("<3f", raw, 84 + i * 50 + 12 + j * 12)
                               for i in range(triangles) for j in range(3)])
        vertices *= np.asarray(vector(mesh, "scale", "1 1 1"))
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        axis, sign = (1, -1 if side == "left" else 1) if finger == "thumb" else (0, 1)
        point = (lower + upper) / 2
        point[axis] = upper[axis] if sign > 0 else lower[axis]
        position = (matrix(transform(geom)) @ np.asarray([*point, 1]))[:3]
        offset = [*position, 0, 0, 0, 1]  # Tip axes follow the distal link, not mesh axes.
        frame = finger + "_tip"
        fingers[finger] = {"joints": selected, "chain": chain, "tip_frame": frame,
                           "tip_offset": np.round(offset, 12).tolist()}
        frame_bodies[frame] = body_name
        sources[frame] = {"mesh": str(Path("meshes") / mesh.get("file")),
                          "sha256": digest(mesh_path), "distal_axis": axis, "distal_sign": sign,
                          "bounds_m": [lower.tolist(), upper.tolist()]}
        home = np.eye(4)
        for link in chain:
            home = home @ matrix(link["origin"])
            if "joint" in link and link["reference"] != 0:
                raise ValueError("SIMPLE home extraction requires zero joint references")
        home_tips[finger] = (home @ matrix(fingers[finger]["tip_offset"]))[:3, 3]
    weights = {"thumb": 0.5, "index": 0.25, "middle": 0.25}
    center = sum(weights[name] * value for name, value in home_tips.items())
    return {
        "geometry": {
            "description": "Nominal mesh-distal fingertip references; grasp_center is the "
            "midpoint between thumb and opposing-finger midpoint, with hand-aligned axes. "
            "grasp_reference fixes that center at home. "
            "These are geometric references, not contacts.",
            "palm_frame": "palm", "reference_frame": "grasp_reference", "fingers": fingers,
            "center": {"frame": "grasp_center", "weights": weights, "rotation_xyzw": [0, 0, 0, 1]},
        },
        "frames": {"palm": transform(palm),
                   "grasp_reference": [*np.round(center, 12).tolist(), 0, 0, 0, 1],
                   **dict.fromkeys(frame_bodies), "grasp_center": None},
        "frame_bodies": frame_bodies,
        "geometry_sources": sources,
    }
