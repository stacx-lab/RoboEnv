"""Resolve typed release bindings without interpreting executable prose."""

import json
import re

import numpy as np
from roboenv.provisioning.chemlab_assets import resource


def component_links(contract, component):
    links = []
    for item in contract.get("components", {}).get(component, {}).get("geometry", []):
        if item.get("artifact_id") != "model_urdf":
            continue
        selector = item.get("selector", "")
        match = re.fullmatch(r"link\[@name=[\"\']([^\"\']+)[\"\']\](?:/\w+)?", selector)
        if match:
            links.append(match[1])
        elif selector.startswith("link:"):
            links.append(selector.removeprefix("link:"))
        elif selector.isidentifier():
            links.append(selector)
    # Several panel components omit artifact selectors but use the same URDF ID.
    return links or ([component] if component else [])


def panel_spec(instrument):
    extension = instrument.contract.get("extensions", {}).get("chemistry_lab.panel_ui", {})
    artifact = instrument.contract.get("artifacts", {}).get(extension.get("artifact_id"))
    return json.loads(resource(instrument.root, artifact["path"]).read_text()) if artifact else {}


def joint_bindings(instrument, model):
    """Only typed affine transforms or equal scalar units authorize a joint map."""
    contract = instrument.contract
    knobs = contract.get("extensions", {}).get("matterix.bindings", {}).get("knobs", {})
    result = {}
    for key, binding in contract.get("model_bindings", {}).items():
        slot, target = binding.get("state_id"), binding.get("target")
        if binding.get("channel") != "joint" or not slot:
            continue
        if binding.get("direction") == "scene_to_process":
            continue
        if target not in model.joint_map:
            links = component_links(contract, binding.get("component_id"))
            if target in model.link_map:
                links = [target]
            candidates = [
                j.name
                for j in model.robot.joints
                if j.child in links and j.type in {"prismatic", "revolute", "continuous"}
            ]
            if len(candidates) != 1:
                continue
            target = candidates[0]
        joint = model.joint_map[target]
        if joint.type not in {"prismatic", "revolute", "continuous"}:
            continue
        transform = knobs.get(target, {})
        if transform and transform.get("state") != slot:
            continue
        unit = contract["state_slots"].get(slot, {}).get("unit")
        native_unit = "m" if joint.type == "prismatic" else "rad"
        velocity = binding.get("attribute") == "velocity"
        if velocity:
            native_unit += "/s"
        if unit in {"turn", "turns", "rev"} and native_unit == "rad":
            transform = {"scale": 2 * np.pi, "offset": 0.0}
        if not transform and (unit != binding.get("unit") or unit != native_unit):
            continue
        result[key] = {
            "target": target,
            "state": slot,
            "scale": transform.get("scale", 1.0),
            "offset": transform.get("offset", 0.0),
            "velocity": velocity,
        }
    return result


def screen_quad(mesh):
    """Fit the largest front-facing planar face of a declared display mesh.

    This is a geometric placement rule, not a new instrument display model.
    Pixel contents always come from the release's panel.capture capability.
    """
    normals = np.round(mesh.face_normals, 5)
    choices, indices = np.unique(normals, axis=0, return_inverse=True)
    areas = np.bincount(indices, weights=mesh.area_faces)
    # Assets use Z up, -Y front. Prefer the outward face over its back face.
    front = np.array([0.0, -1.0, 1.0])
    candidates = np.flatnonzero(choices @ front > 0.01)
    if not len(candidates):
        raise ValueError("Display has no front-facing surface")
    index = candidates[np.argmax(areas[candidates])]
    normal = choices[index] / np.linalg.norm(choices[index])
    right = np.array([1.0, 0.0, 0.0])
    right -= normal * (right @ normal)
    if np.linalg.norm(right) < 0.1:
        raise ValueError("Display requires an explicit side-facing placement")
    right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    points = mesh.vertices[np.unique(mesh.faces[indices == index])]
    x, y = points @ right, points @ up
    center = (
        right * (x.min() + x.max()) / 2
        + up * (y.min() + y.max()) / 2
        + normal * (np.mean(points @ normal) + 0.0006)
    )
    width, height = np.ptp(x), np.ptp(y)
    if min(width, height) < 1e-5:
        raise ValueError("Degenerate display surface")
    corners = np.array(
        [
            center + right * dx * width / 2 + up * dy * height / 2
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        ]
    )
    return corners, normal


def binding_report(
    instrument, joint_keys=(), panel_components=(), lamp_keys=(), visible_keys=(), readout_keys=()
):
    """Account for every channel, including unbound and unimplemented channels."""
    report = {
        "synchronized": [],
        "readout_only": [],
        "unbound": [],
        "unsupported": [],
        "input_only": [],
    }
    supported = set(joint_keys) | set(lamp_keys) | set(visible_keys)
    for key, binding in instrument.contract.get("model_bindings", {}).items():
        if binding.get("direction") == "scene_to_process":
            category = "input_only"
        elif binding.get("state_id") is None:
            category = "unbound"
        elif key in readout_keys:
            category = "readout_only"
        elif key in supported or (
            binding.get("channel") == "display" and binding.get("component_id") in panel_components
        ):
            category = "synchronized"
        else:
            category = "unsupported"
        report[category].append(key)
    return report
