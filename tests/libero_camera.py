"""Render owner-only calibration markers through the real device camera path."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
from roboenv_runtime.runtime.worker import write_png

ROOT = Path(__file__).resolve().parents[1]


def matrix(pose):
    result = np.eye(4)
    quat = np.asarray([pose["rotation_xyzw"][3], *pose["rotation_xyzw"][:3]])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    result[:3, :3] = rotation.reshape(3, 3)
    result[:3, 3] = pose["translation_m"]
    return result


def check(directory):
    adapter = LiberoAdapter(enable_render=True, realtime=True)
    try:
        adapter.reset(0)
        world = adapter._env
        root = ROOT / "catalog/robots/libero-panda/devices"
        profiles = {
            name: yaml.safe_load((root / name / "profile.yaml").read_text())
            for name in ("arm", "camera", "camera_wrist")
        }
        records = []
        for movement in (0, 0.12):
            goal = world.sim.data.qpos[world._robot._ref_joint_pos_indexes].copy()
            goal[0] += movement
            world.step({"arm": goal})
            for _ in range(100):
                world.tick()
            for name, camera in [("camera", "agentview"), ("camera_wrist", "robot0_eye_in_hand")]:
                profile = profiles[name]
                mounted = matrix(profile["extrinsics"]["mount"])
                if name == "camera_wrist":
                    flange = world.sim.model.body_name2id(world._robot.robot_model.eef_name)
                    measured = np.eye(4)
                    measured[:3, :3] = world.sim.data.body_xmat[flange].reshape(3, 3)
                    measured[:3, 3] = world.sim.data.body_xpos[flange]
                    mounted = measured @ mounted
                camera_id = world.sim.model.camera_name2id(camera)
                assert np.allclose(mounted[:3, 3], world.sim.data.cam_xpos[camera_id], atol=1e-10)
                assert np.allclose(
                    mounted[:3, :3],
                    world.sim.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1]),
                    atol=1e-10,
                )
                for point in ([0.03, 0.02, 0.15], [-0.03, -0.02, 0.15]):
                    world_point = mounted[:3, :3] @ point + mounted[:3, 3]
                    original = mujoco.mjv_updateScene

                    def marker(*args, original=original, world_point=world_point, **kwargs):
                        original(*args, **kwargs)
                        scene = args[-1]
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.004] * 3),
                            world_point,
                            np.eye(3).ravel(),
                            np.array([1.0, 0.0, 1.0, 1.0]),
                        )
                        geom.emission = 1
                        scene.ngeom += 1

                    mujoco.mjv_updateScene = marker
                    try:
                        frame = world.render(camera)
                    finally:
                        mujoco.mjv_updateScene = original
                    k = profile["intrinsics"]["color"]["projection"]
                    expected = [
                        k["cx"] + k["fx"] * point[0] / point[2],
                        k["cy"] + k["fy"] * point[1] / point[2],
                    ]
                    mask = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] > 200)
                    y, x = np.nonzero(mask)
                    assert len(x) > 2
                    actual = [float(np.mean(x)), float(np.mean(y))]
                    error = float(np.linalg.norm(np.asarray(expected) - actual))
                    assert error < 1.5, (name, expected, actual)
                    file = directory / f"{len(records)}-{name}.png"
                    write_png(file, frame)
                    records.append(
                        {
                            "camera": name,
                            "motion_rad": movement,
                            "expected_uv": expected,
                            "measured_uv": actual,
                            "error_pixels": error,
                            "file": str(file),
                        }
                    )
        (directory / "report.json").write_text(
            json.dumps({"passed": True, "projections": records}, indent=2) + "\n"
        )
    finally:
        adapter.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check(args.out)
