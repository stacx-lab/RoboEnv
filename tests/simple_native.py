"""Explicit native SIMPLE conformance and owner record; no model or task solution."""

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.simple.backend import SimpleEnv
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose
from simple_hand_probe import compare as check_hands


def check_geometry(world):
    errors = {}
    for name in ("waist", "left_arm", "right_arm"):
        state = world.read_device(name)
        transform = np.eye(4)
        for link in world.model["parts"][name]["chain"]:
            transform = transform @ matrix(link["origin"])
            if "joint" in link:
                axis = np.asarray(link["axis"], dtype=float)
                angle = state["positions"][link["joint"]]
                rotation = matrix([0, 0, 0, *(axis * np.sin(angle / 2)), np.cos(angle / 2)])
                anchor = np.asarray(link["anchor"])
                rotation[:3, 3] = anchor - rotation[:3, :3] @ anchor
                transform = transform @ rotation
        part = world.model["parts"][name]
        measured = (
            np.linalg.inv(matrix(world._pose(part["base"]))) @ matrix(world._pose(part["tip"]))
        )
        assert "frame_poses" not in state
        errors[name] = float(np.max(np.abs(transform - measured)))
    assert max(errors.values()) < 1e-6, errors
    return errors


def check_cameras(world):
    """Compare catalog projection with Isaac's native world-to-pixel projection."""
    from roboenv_runtime.adapters.simple.cameras import validate_isaac

    validate_isaac(world)
    if world.env.isaac is None:
        return {}
    parent = matrix(world._pose("torso_link"))
    optical_points = np.array([[0, 0, 1], [0.15, -0.1, 1.5], [-0.2, 0.15, 2.0]])
    errors = {}
    for name, spec in world.model["cameras"].items():
        mount = spec["mount"]
        pose = parent @ matrix(mount["translation_m"] + mount["rotation_xyzw"])
        points = optical_points @ pose[:3, :3].T + pose[:3, 3]
        projection = spec["projection"]
        expected = optical_points[:, :2] / optical_points[:, 2:]
        expected *= [projection["fx"], projection["fy"]]
        expected += [projection["cx"], projection["cy"]]
        camera = world.env.isaac.cameras[spec["native"]]
        actual = camera.get_image_coords_from_world_points(points) - 0.5
        errors[name] = float(np.max(np.abs(actual - expected)))
    assert max(errors.values()) < 0.01, errors
    return errors


def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    world = SimpleEnv(
        env_id=args.env_id,
        sim_mode=args.sim_mode,
        arm_gravity_compensation=args.arm_gravity_compensation,
    )
    try:
        world.configure_execution_mode(args.execution_mode)
        world.reset(args.seed)
        projection_before = check_cameras(world)
        before = {"states": {n: world.read_device(n) for n in world.control.targets}}
        initial = world.state().copy()
        initial_pelvis = np.asarray(world._pose("pelvis")[:3]).copy()
        geometry_before = check_geometry(world)
        hands_before = check_hands(world, args.kinex)
        cameras = {name: world.read_camera(name, args.out) for name in world.model["cameras"]}
        assert np.array_equal(world.state(), initial), "Reads advanced physics"
        targets = copy.deepcopy(world.control.targets)
        target = list(targets["left_arm"])
        target[0] += 0.1
        command = {
            "mode": "locomotion",
            "velocity": {"vx": 0.15, "vy": 0, "wz": 0.1},
            "posture": {"height": 0.72, "roll": 0.03, "pitch": 0.05, "yaw": 0.08},
        }
        hand_target = list(targets["left_hand"])
        hand_target[0] += 0.1
        commands = [
            {"device": "base", "args": command},
            {"device": "left_arm", "args": {"target": target}},
            {"device": "left_hand", "args": {"target": hand_target}},
        ]
        try:
            world.control.write("left_arm", {"target": [0]})
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed target accepted")
        assert world.control.targets == targets and np.array_equal(world.state(), initial)
        for _ in range(60):
            for command in commands:
                world.control.write(command["device"], command["args"])
            world.control.tick()
        after = {"states": {n: world.read_device(n) for n in world.control.targets}}
        assert "pose" not in after["states"]["base"]
        assert abs(world.timestamp - 1.2) < 1e-9
        assert world.control.targets["right_arm"] == targets["right_arm"]
        displacement = np.linalg.norm(
            np.asarray(world._pose("pelvis")[:3]) - initial_pelvis
        )
        arm_motion = np.linalg.norm(
            np.asarray(after["states"]["left_arm"]["positions"])
            - before["states"]["left_arm"]["positions"]
        )
        hand_motion = np.linalg.norm(
            np.asarray(after["states"]["left_hand"]["positions"])
            - before["states"]["left_hand"]["positions"]
        )
        assert displacement > 0.02 and arm_motion > 0.03, (displacement, arm_motion)
        assert hand_motion > 0.02, hand_motion
        geometry_after = check_geometry(world)
        hands_after = check_hands(world, args.kinex)
        projection_after = check_cameras(world)
        final_cameras = {n: world.read_camera(n, args.out) for n in world.model["cameras"]}
        from PIL import Image

        for name in cameras:
            initial_image = np.asarray(Image.open(cameras[name]["file"]))
            final_image = np.asarray(Image.open(final_cameras[name]["file"]))
            assert initial_image.max() > 0 and final_image.max() > 0, name
            assert not np.array_equal(initial_image, final_image), name
            assert final_cameras[name]["timestamp"] == world.timestamp
        assert not np.array_equal(
            np.asarray(Image.open(final_cameras["camera_head_left"]["file"])),
            np.asarray(Image.open(final_cameras["camera_head_right"]["file"])),
        ), "Stereo cameras returned identical images"
        np.testing.assert_allclose(world.control.targets["base"][4:], [0.03, 0.05, 0.08])
        lower = world.wbc.policy.lower_body_policy
        np.testing.assert_allclose(
            [lower.roll_cmd, lower.pitch_cmd, lower.yaw_cmd], [0.03, 0.05, 0.08]
        )
        assert "waist" not in world.control.targets
        np.testing.assert_allclose(
            after["states"]["base"]["frame_poses"]["torso"],
            matrix_pose(np.linalg.inv(matrix(world._pose(world.model["parts"]["waist"]["base"])))
                        @ matrix(world._pose(world.model["parts"]["waist"]["tip"]))),
        )
        # Upper joint targets survive native lower-body inference unchanged.
        for group in ("arms", "hands"):
            indices = world.wbc.model.get_joint_group_indices(group)
            names = [
                name
                for name in world.wbc.upper_names
                if world.wbc.model.joint_to_dof_index[name] in indices
            ]
            expected = [
                world.wbc.upper_target[world.wbc.upper_names.index(n)] for n in names
            ]
            actual = [
                world.wbc.policy.last_action["q"][world.wbc.model.joint_to_dof_index[n]]
                for n in names
            ]
            np.testing.assert_array_equal(actual, expected)
        stamp = world.timestamp
        for name in world.control.targets:
            world.control.stop(name)
        assert world.timestamp == stamp
        assert world.control.targets["base"][:3] == [0, 0, 0]
        assert world.mjdata.ncon > 0, "No native contacts during standing/walking"
        (args.out / "record.json").write_text(json.dumps(world.record()) + "\n")
        report = {
            "passed": True,
            "model_calls": 0,
            "task_attempted": False,
            "native_success": world.completed,
            "frames": world.steps,
            "displacement_m": float(displacement),
            "arm_motion_rad": float(arm_motion),
            "hand_motion_rad": float(hand_motion),
            "geometry_before": geometry_before,
            "hands_before": hands_before,
            "hands_after": hands_after,
            "geometry_after": geometry_after,
            "projection_error_px_before": projection_before,
            "projection_error_px_after": projection_after,
            "cameras": cameras,
        }
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-id", default="simple/G1WholebodyXMovePickTeleop-v0")
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco")
    parser.add_argument("--execution-mode", choices=("stepped", "realtime"), default="stepped")
    parser.add_argument(
        "--arm-gravity-compensation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kinex", type=Path, default=Path(__file__).resolve().parents[2] / "kinex")
    code = 0
    try:
        check(parser.parse_args())
    except SystemExit as exc:
        code = exc.code
    except Exception:
        traceback.print_exc()
        code = 1
    # Like the worker, avoid native destructors after Kit has unloaded its plugins.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
