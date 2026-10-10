"""Compare ABC catalogs and Kinex FK with native geometry; isolated ABC Python, no GPU."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from roboenv.tasks.abc import entries
from roboenv_runtime.adapters.abc.backend import SOURCE
from roboenv_runtime.adapters.abc.rig import aperture, describe, finger_pads

ROOT = Path(__file__).resolve().parents[1]


def check(kinex):
    sys.path.insert(0, str(kinex / "packages/core/assets/python"))
    from abc_sim.task_registry import get_task_scene_xml
    from kinex.geometry import compose_rt, mount_to_rt, xyzq_to_rt
    from kinex.joints import JointDevice

    profiles = {
        path.parent.name: yaml.safe_load(path.read_text())
        for path in (ROOT / "catalog/robots/abc/abc-yam/devices").glob("*/profile.yaml")
    }
    arms = {side: JointDevice(profiles[side + "_arm"]) for side in ("left", "right")}
    errors = {"translation_m": [], "rotation_matrix": []}

    def compare(actual, rotation, position):
        r, p = actual
        errors["translation_m"].append(float(np.max(np.abs(np.asarray(p) - position))))
        errors["rotation_matrix"].append(float(np.max(np.abs(np.asarray(r) - rotation))))
        np.testing.assert_allclose(p, position, atol=1e-9, rtol=0)
        np.testing.assert_allclose(r, rotation, atol=1e-9, rtol=0)

    tasks = list(entries(SOURCE))
    scenes = sorted({task["env_task"] for task in tasks})
    rng = np.random.default_rng(42)
    for scene in scenes:
        # Compile the full upstream scene independently of the task-free generator.
        model = mujoco.MjModel.from_xml_path(str(get_task_scene_xml(scene)))
        facts = describe(model)
        assert set(facts) == set(profiles)
        for name, fact in facts.items():
            for field in ("type", "intrinsics", "extrinsics"):
                assert fact[field] == profiles[name].get(field, {}), (scene, name, field)
        data = mujoco.MjData(model)
        # Home plus varied poses exercise every joint and each moving mount.
        for sample in range(6):
            for side, arm in arms.items():
                for joint in arm.joints:
                    q = (arm.intrinsics["home"][joint["name"]] if sample == 0 else
                         rng.uniform(joint["limits"]["min"], joint["limits"]["max"]))
                    data.qpos[model.joint(joint["name"]).qposadr[0]] = q
                opening = sample / 5
                data.qpos[model.joint(side + "_left_finger").qposadr[0]] = .0475 * opening
                data.qpos[model.joint(side + "_right_finger").qposadr[0]] = -.0475 * opening
            mujoco.mj_forward(model, data)
            flanges = {}
            for side, arm in arms.items():
                q = [float(data.qpos[model.joint(j["name"]).qposadr[0]]) for j in arm.joints]
                flange = compose_rt(mount_to_rt(profiles[side + "_arm"]["extrinsics"]["mount"]),
                                    xyzq_to_rt(arm.fk(q, frame="flange")["pose"]))
                body = model.body(side + "_link_6").id
                compare(flange, data.xmat[body].reshape(3, 3), data.xpos[body])
                flanges[side] = flange
                grip = profiles[side + "_gripper"]
                base = compose_rt(flange, mount_to_rt(grip["extrinsics"]["mount"]))
                tcp = compose_rt(base, mount_to_rt(grip["intrinsics"]["frames"]["tcp"]))
                site = model.site(side + "_grasp_site").id
                compare(tcp, data.site_xmat[site].reshape(3, 3), data.site_xpos[site])
                geometry = grip["intrinsics"]["geometry"]
                pad = finger_pads(model, side)[0]
                for axis, native in (("approach_axis", data.site_xmat[site].reshape(3, 3)[:, 2]),
                                     ("closing_axis", data.geom_xmat[pad].reshape(3, 3)[:, 1])):
                    np.testing.assert_allclose(np.asarray(base[0]) @ geometry[axis], native,
                                               atol=1e-9, rtol=0)
                bounds = grip["intrinsics"]["aperture_m"]
                expected = bounds["min"] + opening * (bounds["max"] - bounds["min"])
                np.testing.assert_allclose(aperture(model, data, side), expected, atol=1e-9)
                # Local IK must recover a native target from a displaced seed.
                if sample == 0:
                    target = arm.fk(q, frame="flange")["pose"]
                    arm.ik(target, arm.project([v + .01 for v in q]), frame="flange")
            for name in ("top", "left", "right"):
                profile = profiles["camera_" + name]
                mount = profile["extrinsics"]["mount"]
                optical = mount_to_rt(mount)
                if name != "top":
                    assert mount["parent"] == name + "_arm" and mount["parent_frame"] == "flange"
                    optical = compose_rt(flanges[name], optical)
                else:
                    assert mount["parent"] == mount["parent_frame"] == "world"
                cid = model.camera(name).id
                compare(optical, data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]),
                        data.cam_xpos[cid])
                color = profile["intrinsics"]["color"]
                k = color["projection"]
                focal = color["height"] / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
                np.testing.assert_allclose([k["fx"], k["fy"]], focal, atol=1e-9)
                assert [k["cx"], k["cy"]] == [(color["width"] - 1) / 2, (color["height"] - 1) / 2]
                depth = profile["intrinsics"]["depth"]
                assert depth["registration"] == "color" and depth["quantity"] == "z"
                assert [depth["width"], depth["height"]] == [color["width"], color["height"]]
                assert depth["scale_m_per_unit"] == .001 and depth["invalid_value"] == 0
    return {"passed": True, "tasks": len(tasks), "scenes": len(scenes), "poses_per_scene": 6,
            "arms": 2, "grippers": 2, "cameras": 3,
            "max_errors": {key: max(values) for key, values in errors.items()},
            "checks": ["catalog_equality", "native_fk", "native_tcp", "ik", "aperture", "grip_axes",
                       "moving_optical_mounts", "intrinsics", "depth_units"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.kinex.resolve())
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
