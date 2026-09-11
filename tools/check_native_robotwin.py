"""Owner-only RoboTwin conformance, with physical contact and rendered calibration.

Run each embodiment/seed/clock in a fresh process. Calibration boxes are separate
fixtures and never establish task success or enter controller development data.
"""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.adapters.robotwin.devices import RoboTwinDevices
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose
from roboenv_runtime.runtime.worker import write_png

TOLERANCES = {
    "fk_position_m": 2e-5,
    "fk_rotation_rad": 2e-4,
    "ik_position_m": 1e-4,
    "ik_rotation_rad": 1e-3,
    "arrival_rad": 0.01,
    "arrival_velocity_rad_s": 0.03,
    "stop_drift_rad": 0.002,
    "projection_px": 2.0,
    "depth_m": 0.003,
}
CAMERAS = ("head_camera", "left_camera", "right_camera")
ARMS = ("left_arm", "right_arm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(world, commands):
    frame = world.control.frame
    step = {"id": frame + 100, "devices": list(commands)} if len(commands) > 1 else None
    for name, target in commands.items():
        world.device_write(name, target, step)
    if world.realtime:
        require(world.control.frame == frame, "Realtime writes advanced physics")
        world.control.tick()
    require(world.control.frame == frame + 1, "Control interval did not advance exactly once")


def mount_transform(fact):
    value = fact["extrinsics"]["mount"]
    return matrix(value["translation_m"] + value["rotation_xyzw"])


def projection(world, models, out, stage):
    import sapien
    from envs.utils.create_actor import create_box

    report = {}
    for name in CAMERAS:
        fact = world.device_facts[name]
        transform = mount_transform(fact)
        if name != "head_camera":
            arm = name.split("_", 1)[0] + "_arm"
            q = world.device_read(arm)["positions"]
            transform = (
                mount_transform(world.device_facts[arm])
                @ matrix(models[arm].fk(q)["pose"])
                @ transform
            )
        # An off-axis cube checks signs, focal lengths, mounts and metric depth.
        xyz = np.array([0.025, -0.018, 0.18])
        half = 0.006
        pose = matrix_pose(transform)
        position = (transform @ np.r_[xyz, 1])[:3]
        marker = create_box(
            world.task,
            sapien.Pose(position, [pose[6], *pose[3:6]]),
            [half] * 3,
            color=[1, 0, 1],
            is_static=True,
            name="roboenv_projection_fixture",
        )
        try:
            before = world.owner_state()
            rgb, depth = world.device_image(name)
            labels = world._camera(name).get_picture("Segmentation")[..., 1]
            mask = labels == marker.actor.per_scene_id
            ys, xs = np.where(mask)
            require(len(xs) >= 20, f"{name}: calibration marker is occluded or not rendered")
            observed = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
            k = fact["intrinsics"]["color"]["projection"]
            corners = xyz + np.asarray(list(itertools.product([-half, half], repeat=3)))
            pixels = np.column_stack(
                [
                    k["cx"] + k["fx"] * corners[:, 0] / corners[:, 2],
                    k["cy"] + k["fy"] * corners[:, 1] / corners[:, 2],
                ]
            )
            expected = (pixels.min(0) + pixels.max(0)) / 2
            error = float(np.max(np.abs(observed - expected)))
            # Center pixel lies on the front face; side faces have greater depth.
            u, v = np.rint(
                [
                    k["cx"] + k["fx"] * xyz[0] / (xyz[2] - half),
                    k["cy"] + k["fy"] * xyz[1] / (xyz[2] - half),
                ]
            ).astype(int)
            depth_error = abs(float(depth[v, u]) - (xyz[2] - half))
            write_png(out / f"{stage}-{name}.png", rgb)
            report[name] = {
                "expected_uv": expected.tolist(),
                "observed_uv": observed.tolist(),
                "projection_error_px": error,
                "depth_error_m": depth_error,
                "marker_pixels": len(xs),
            }
            (out / f"{stage}-{name}.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(
                error <= TOLERANCES["projection_px"], f"{name}: rendered projection error {error}"
            )
            require(
                depth_error <= TOLERANCES["depth_m"], f"{name}: metric depth error {depth_error}"
            )
            require(world.owner_state() == before, "Rendering advanced native physics")
        finally:
            world.task.scene.remove_entity(marker.actor)
    return report


def contacts(world):
    import sapien
    from envs.utils.create_actor import create_box

    report = {}
    for side in ("left", "right"):
        name = side + "_gripper"
        for _ in range(25):
            interval(world, {name: [1.0]})
        tcp = getattr(world.task.robot, "get_" + side + "_tcp_pose")()
        marker = create_box(
            world.task,
            sapien.Pose(tcp[:3], tcp[3:]),
            [0.012] * 3,
            color=[1, 0, 1],
            is_static=True,
            name="roboenv_contact_fixture",
        )
        try:
            for _ in range(30):
                interval(world, {name: [0.0]})
            contact = world.gripper_contacts(marker, side)
            held = world.device_read(name)
            require(
                contact["touching"] and len(contact["finger_links"]) >= 2,
                f"{side}: both fingers did not contact the physical fixture",
            )
            frame = world.control.frame
            world.device_stop(name)
            require(world.control.frame == frame, "Gripper stop advanced physics")
            for _ in range(25):
                interval(world, {name: [1.0]})
            release = world.gripper_contacts(marker, side)
            require(not release["touching"], f"{side}: gripper did not release")
            report[name] = {
                "contact": contact,
                "release": release,
                "measured_contact": held,
                "measured_open": world.device_read(name),
            }
        finally:
            world.task.scene.remove_entity(marker.actor)
    return report


def check(world, out):
    from kinex.manipulator import Manipulator
    from kinex.motion import pose_error

    report = {
        "passed": False,
        "checks": {},
        "tolerances": TOLERANCES,
        "initialization": world.initialization,
        "initial_state_sha256": digest(world.initial_state),
        "implementation": world.implementation_identity,
    }
    models = {name: Manipulator({"name": name, **world.device_facts[name]}) for name in ARMS}
    before, frame = world.owner_state(), world.control.frame
    for name in CAMERAS:
        for _ in range(2):
            rgb, depth = world.device_image(name)
            require(
                rgb.max() > 0 and rgb.std() > 1 and np.isfinite(depth).all(), "Invalid native RGBD"
            )
    report["geometry"] = {}
    for name, model in models.items():
        reading = world.device_read(name)
        q = model.project(reading["positions"])
        fk_error = pose_error(model.fk(reading["positions"])["pose"], reading["flange_pose"])
        target_q = q.copy()
        target_q[0] += 0.03
        target_q = model.project(target_q)
        target = model.fk(target_q)["pose"]
        solved = model.ik(target, q)
        ik_error = pose_error(model.fk(solved["joints"])["pose"], target)
        require(
            fk_error["position_m"] < TOLERANCES["fk_position_m"]
            and fk_error["rotation_rad"] < TOLERANCES["fk_rotation_rad"],
            "Local FK mismatch",
        )
        require(
            ik_error["position_m"] < TOLERANCES["ik_position_m"]
            and ik_error["rotation_rad"] < TOLERANCES["ik_rotation_rad"],
            "Local IK residual",
        )
        report["geometry"][name] = {"fk_error": fk_error, "ik_error": ik_error}
    require(
        world.owner_state() == before and world.control.frame == frame,
        "Reads or local geometry changed state",
    )
    report["checks"]["read_and_geometry_invariance"] = True
    report["projection_initial"] = projection(world, models, out, "initial")
    goals = {name: models[name].project(world.device_read(name)["positions"]) for name in ARMS}
    origins = copy.deepcopy(goals)
    for value in goals.values():
        value[0] += 0.1
    state = copy.deepcopy(
        (world.owner_state(), world.control.targets, world.control.pending, world.control.commands)
    )
    invalid = [[float("nan")] * 6, [float("inf")] * 6, [0.0], [True] * 6, [1e9] * 6]
    for target in invalid:
        try:
            world.device_write("left_arm", target)
        except ValueError:
            pass
        else:
            raise ValueError("Malformed target was accepted")
        require(
            (
                world.owner_state(),
                world.control.targets,
                world.control.pending,
                world.control.commands,
            )
            == state,
            "Malformed target had an effect",
        )
    step = {"id": 1, "devices": list(ARMS)}
    world.device_write("left_arm", goals["left_arm"], step)
    require(
        world.control.frame == frame and world.owner_state() == before,
        "Partial atomic group advanced physics",
    )
    world.device_write("right_arm", goals["right_arm"], step)
    if world.realtime:
        require(world.control.frame == frame, "Realtime atomic write advanced physics")
        world.control.tick()
    require(world.control.frame == frame + 1, "Atomic group did not compose one interval")
    for _ in range(29):
        interval(world, goals)
    report["motion"] = {}
    for name, goal in goals.items():
        reading = world.device_read(name)
        error = float(np.max(np.abs(np.asarray(reading["positions"]) - goal)))
        movement = float(np.max(np.abs(np.asarray(reading["positions"]) - origins[name])))
        velocity = float(np.max(np.abs(reading["velocities"])))
        require(
            error < TOLERANCES["arrival_rad"]
            and velocity < TOLERANCES["arrival_velocity_rad_s"]
            and movement > 0.05,
            f"{name}: target acceptance did not produce measured arrival",
        )
        report["motion"][name] = {
            "error_rad": error,
            "movement_rad": movement,
            "velocity_rad_s": velocity,
        }
    require(world._sim_step_count == world.control.frame * 10, "Native control decimation mismatch")
    report["checks"].update(invalid_zero_effects=True, atomic_interval=True, measured_arrival=True)
    peer = copy.deepcopy(world.control.targets["right_arm"])
    frame = world.control.frame
    world.device_stop("left_arm")
    held = world.device_read("left_arm")["positions"]
    require(
        world.control.frame == frame and world.control.targets["right_arm"] == peer,
        "Stop changed time or peer target",
    )
    peer[0] += 0.15
    for _ in range(35):
        interval(world, {"right_arm": peer})
    drift = float(np.max(np.abs(np.asarray(held) - world.device_read("left_arm")["positions"])))
    require(drift < TOLERANCES["stop_drift_rad"], f"Stop did not hold measured joints: {drift}")
    report["stop_drift_rad"] = drift
    report["projection_moved"] = projection(world, models, out, "moved")
    world._verify_cameras()
    report["contact"] = contacts(world)
    report["checks"].update(
        stop_scope=True, contact_release=True, rendered_projection=True, dynamic_mounts=True
    )
    while world.control.frame < world.frame_limit:
        interval(world, {"right_arm": peer})
    before = world.owner_state()
    try:
        interval(world, {"right_arm": peer})
    except RuntimeError:
        pass
    else:
        raise ValueError("Exhausted episode accepted another interval")
    require(world.owner_state() == before, "Exhausted episode changed native state")
    report["checks"]["finite_budget_freeze"] = True
    report.update(passed=True, frames=world.control.frame, physics_steps=world._sim_step_count)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embodiment", choices=["piper", "aloha_agilex"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kinex.resolve() / "python"))
    report = {
        "passed": False,
        "schema": "roboenv/robotwin-conformance/1",
        "embodiment": args.embodiment,
        "seed": args.seed,
        "realtime": args.realtime,
        "scope": "native device conformance with owner calibration fixtures; no task success claim",
    }
    world = RoboTwinDevices(
        "stack_blocks_three", args.embodiment, max_steps=320, realtime=args.realtime
    )
    try:
        world.reset(seed=args.seed)
        report.update(check(world, out))
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        world.close()
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
