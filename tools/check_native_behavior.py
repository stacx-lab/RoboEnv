"""Owner-only R1Pro native conformance; fixtures never establish BDDL task success."""

import argparse
import copy
import faulthandler
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.behavior.devices import BehaviorDevices, pose
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose
from roboenv_runtime.runtime.worker import write_png

TOLERANCES = {
    "fk_position_m": 1e-3,
    "fk_rotation_rad": 2e-3,
    "ik_position_m": 1e-4,
    "ik_rotation_rad": 1e-3,
    "arrival_rad": 0.015,
    "arrival_velocity_rad_s": 0.05,
    "stop_drift_rad": 0.005,
    "base_stop_m_s": 0.01,
    "projection_px": 2.0,
    "depth_m": 0.005,
}
CHAINS = ("trunk", "left_arm", "right_arm")
FIXTURES = ("camera_0", "camera_1", "camera_2", "left_gripper", "right_gripper")


class CalibrationWorld(BehaviorDevices):
    """Preload owner markers so adding/removing objects cannot add hidden steps."""

    def build_config(self):
        config = super().build_config()
        config.setdefault("objects", []).extend(
            [
                {
                    "type": "PrimitiveObject",
                    "name": "roboenv_calibration_" + name,
                    "primitive_type": "Cube",
                    "fixed_base": True,
                    "size": 0.012 if name.startswith("camera_") else 0.024,
                    "position": [50.0 + index, 50.0, 1.0],
                    "rgba": [1.0, 0.0, 1.0, 1.0],
                    "include_default_states": False,
                }
                for index, name in enumerate(FIXTURES)
            ]
        )
        return config

    def fixture(self, name):
        return self.env.scene.object_registry("name", "roboenv_calibration_" + name)

    def park(self, name):
        import torch

        self.fixture(name).set_position_orientation(
            position=torch.tensor([50.0 + FIXTURES.index(name), 50.0, 1.0])
        )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(world, commands):
    frame = world.control.frame
    for name, target in commands.items():
        world.device_write(name, target)
    if world.realtime:
        require(world.control.frame == frame, "Realtime write advanced physics")
        world.control.tick()
    expected = 1 if world.realtime else len(commands)
    require(world.control.frame == frame + expected, "Unexpected control interval count")


def placement(world, models, name, frame=None):
    if name == "world":
        return np.eye(4)
    if name == "base":
        value = world.device_read(name)["pose"]
        return matrix(value["translation_m"] + value["rotation_xyzw"])
    mount = world.device_facts[name]["extrinsics"]["mount"]
    transform = placement(world, models, mount["parent"], mount.get("parent_frame")) @ matrix(
        mount["translation_m"] + mount["rotation_xyzw"]
    )
    if frame == "flange":
        transform = transform @ matrix(
            models[name].fk(world.device_read(name)["positions"])["pose"]
        )
    return transform


def verify_mounts(world, models):
    from kinex.motion import pose_error

    report = {}
    for name in CHAINS:
        actual = pose(world.robot.links[world.rig_frames[name]["base"]])
        expected = placement(world, models, name)
        report[name] = pose_error(matrix_pose(expected), matrix_pose(actual))
    for name, sensor_name in world.cameras.items():
        actual = pose(world.robot.sensors[sensor_name]) @ np.diag([1.0, -1.0, -1.0, 1.0])
        report[name] = pose_error(matrix_pose(placement(world, models, name)), matrix_pose(actual))
    for name, error in report.items():
        require(
            error["position_m"] < TOLERANCES["fk_position_m"]
            and error["rotation_rad"] < TOLERANCES["fk_rotation_rad"],
            f"{name}: mount mismatch {error}",
        )
    return report


def projection(world, models, out, phase):
    import omnigibson as og
    import torch

    report = {}
    for name in world.cameras:
        marker = world.fixture(name)
        transform = placement(world, models, name)
        xyz, half = np.array([0.025, -0.018, 0.18]), 0.006
        orientation = matrix_pose(transform)[3:]
        marker.set_position_orientation(
            position=torch.tensor((transform @ np.r_[xyz, 1])[:3]),
            orientation=torch.tensor(orientation),
        )
        try:
            before = world.owner_state()
            for _ in range(8):
                og.sim.render()
            rgb, depth = world.device_image(name)
            # The near-camera magenta fixture is visible in the public RGB-D
            # stream. Extra semantic annotators are unnecessary for calibration
            # and their graphs can become invalid after native instance reset.
            red, green, blue = np.moveaxis(rgb.astype(float), -1, 0)
            mask = (red > 80) & (blue > 80) & (green < 0.6 * np.minimum(red, blue))
            mask &= np.isfinite(depth) & (depth > 0.1) & (depth < 0.3)
            ys, xs = np.where(mask)
            require(len(xs) >= 20, f"{name}: calibration marker absent or occluded")
            observed = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
            k = world.device_facts[name]["intrinsics"]["color"]["projection"]
            corners = xyz + np.asarray(list(itertools.product([-half, half], repeat=3)))
            pixels = np.column_stack(
                [
                    k["cx"] + k["fx"] * corners[:, 0] / corners[:, 2],
                    k["cy"] + k["fy"] * corners[:, 1] / corners[:, 2],
                ]
            )
            expected = (pixels.min(0) + pixels.max(0)) / 2
            error = float(np.max(np.abs(observed - expected)))
            u, v = np.rint(
                [
                    k["cx"] + k["fx"] * xyz[0] / (xyz[2] - half),
                    k["cy"] + k["fy"] * xyz[1] / (xyz[2] - half),
                ]
            ).astype(int)
            depth_error = abs(float(depth[v, u]) - (xyz[2] - half))
            report[name] = {
                "expected_uv": expected.tolist(),
                "observed_uv": observed.tolist(),
                "projection_error_px": error,
                "depth_error_m": depth_error,
                "marker_pixels": len(xs),
            }
            write_png(out / f"{phase}-{name}.png", rgb)
            (out / f"{phase}-{name}.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(error <= TOLERANCES["projection_px"], f"{name}: projection error {error}")
            require(depth_error <= TOLERANCES["depth_m"], f"{name}: depth error {depth_error}")
            require(world.owner_state() == before, "Sensor reads advanced native state")
        finally:
            world.park(name)
    return report


def contacts(world, out):
    from omnigibson.utils.usd_utils import RigidContactAPI

    report = {}
    for side in ("left", "right"):
        name = side + "_gripper"
        marker = world.fixture(name)
        for _ in range(25):
            interval(world, {name: [1.0]})
        position, orientation = world.robot.eef_links[side].get_position_orientation()
        marker.set_position_orientation(position=position, orientation=orientation)
        try:
            for _ in range(40):
                interval(world, {name: [0.0]})

            def contact(marker=marker, side=side):
                return {
                    str(link.name): RigidContactAPI.is_in_contact(
                        scene_idx=world.env.scene.idx,
                        query_set=[link],
                        with_set=[marker],
                        ignore_set=None,
                        current_only=True,
                    )
                    for link in world.robot.finger_links[side]
                }

            closed = contact()
            report[name] = {"contact": closed, "measured_contact": world.device_read(name)}
            (out / f"{name}-contact.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(
                len(closed) >= 2 and all(closed.values()),
                f"{name}: both fingers did not contact {closed}",
            )
            frame = world.control.frame
            world.device_stop(name)
            require(world.control.frame == frame, "Gripper stop advanced time")
            for _ in range(30):
                interval(world, {name: [1.0]})
            opened = contact()
            report[name].update(release=opened, measured_open=world.device_read(name))
            require(not any(opened.values()), f"{name}: physical fixture was not released")
        finally:
            world.park(name)
    return report


def check(world, out, report):
    from kinex.joints import JointDevice
    from kinex.motion import pose_error

    report.update(
        tolerances=TOLERANCES,
        initialization=world.initialization,
        initial_state_sha256=digest(world.initial_state),
        implementation=world.implementation_identity,
    )
    models = {name: JointDevice({"name": name, **world.device_facts[name]}) for name in CHAINS}
    report["geometry"] = {}
    before, frame = world.owner_state(), world.control.frame
    report["sensors"] = {}
    for name in world.cameras:
        rgb, depth = world.device_image(name)
        from roboenv_runtime.bindings.backends.shared.depth import DEPTH_SCALE, write_depth

        valid = np.isfinite(depth) & (depth >= DEPTH_SCALE) & (depth <= 65535 * DEPTH_SCALE)
        report["sensors"][name] = {
            "rgb_dtype": str(rgb.dtype),
            "rgb_std": float(rgb.std()),
            "valid_depth_pixels": int(valid.sum()),
            "depth_pixels": int(depth.size),
        }
        write_png(out / f"sensor-{name}.png", rgb)
        path = out / f"sensor-{name}.pgm"
        write_depth(path, depth)
        encoded = np.frombuffer(path.read_bytes().split(b"\n", 3)[3], dtype=">u2").reshape(
            depth.shape
        )
        # Native rays missing geometry may be inf. The public metric encoding
        # must map them to the declared invalid value, not reject the whole image.
        require(
            rgb.dtype == np.uint8
            and rgb.std() > 1
            and valid.any()
            and np.all(encoded[~valid] == 0),
            f"{name}: invalid RGBD {report['sensors'][name]}",
        )
    goals = {}
    for name, model in models.items():
        reading = world.device_read(name)
        fk_error = pose_error(
            model.fk(reading["positions"])["pose"], reading["frame_poses"]["flange"]
        )
        q = model.project(reading["positions"])
        target_q = q.copy()
        target_q[0] += 0.08
        goals[name] = model.project(target_q)
        target = model.fk(goals[name])["pose"]
        solution = model.ik(target, q)
        ik_error = pose_error(model.fk(solution["joints"])["pose"], target)
        report["geometry"][name] = {"fk_error": fk_error, "ik_error": ik_error}
        require(
            fk_error["position_m"] < TOLERANCES["fk_position_m"]
            and fk_error["rotation_rad"] < TOLERANCES["fk_rotation_rad"],
            f"{name}: FK mismatch",
        )
        require(
            ik_error["position_m"] < TOLERANCES["ik_position_m"]
            and ik_error["rotation_rad"] < TOLERANCES["ik_rotation_rad"],
            f"{name}: IK residual",
        )
    report["mounts_initial"] = verify_mounts(world, models)
    require(world.owner_state() == before and world.control.frame == frame, "Read/geometry effects")
    report["checks"]["read_and_geometry_invariance"] = True
    report["projection_initial"] = projection(world, models, out, "initial")
    before = world.owner_state()
    state = copy.deepcopy((before, world.control.targets, world.control.commands))
    for name in world.groups:
        size = len(world.control.targets[name])
        for target in (
            [float("nan")] * size,
            [float("inf")] * size,
            [],
            [True] * size,
            [1e9] * size,
        ):
            try:
                world.device_write(name, target)
            except ValueError:
                pass
            else:
                raise ValueError(f"{name}: malformed command accepted")
            require(
                (
                    world.owner_state(),
                    world.control.targets,
                    world.control.commands,
                )
                == state,
                "Malformed command had effects",
            )
    starts = {name: np.asarray(world.device_read(name)["positions"]) for name in CHAINS}
    base_start = placement(world, models, "base")
    commands = {**goals, "base": [0.05, 0.0, 0.0]}
    interval(world, commands)
    for _ in range(49):
        interval(world, commands)
    report["motion"] = {}
    for name, goal in goals.items():
        reading = world.device_read(name)
        error = float(np.max(np.abs(np.asarray(reading["positions"]) - goal)))
        movement = float(np.max(np.abs(np.asarray(reading["positions"]) - starts[name])))
        velocity = float(np.max(np.abs(reading["velocities"])))
        report["motion"][name] = {
            "error_rad": error,
            "movement_rad": movement,
            "velocity_rad_s": velocity,
        }
        require(
            error < TOLERANCES["arrival_rad"]
            and velocity < TOLERANCES["arrival_velocity_rad_s"]
            and movement > 0.04,
            f"{name}: measured arrival failed",
        )
    displacement = float(np.linalg.norm((placement(world, models, "base") - base_start)[:3, 3]))
    report["motion"]["base"] = {
        "displacement_m": displacement,
        "measured": world.device_read("base"),
    }
    require(displacement > 0.025, "Base velocity command did not produce measured translation")
    report["checks"].update(
        invalid_zero_effects=True, per_device_intervals=True, measured_arrival=True
    )
    peer = copy.deepcopy(world.control.targets["right_arm"])
    frame = world.control.frame
    for name in ("left_arm", "trunk", "base"):
        world.device_stop(name)
    require(
        world.control.frame == frame and world.control.targets["right_arm"] == peer,
        "Stop scope failed",
    )
    held = {
        name: np.asarray(world.device_read(name)["positions"]) for name in ("left_arm", "trunk")
    }
    peer[0] += 0.08
    peer = models["right_arm"].project(peer)
    for _ in range(40):
        interval(world, {"right_arm": peer})
    report["stop"] = {}
    for name, positions in held.items():
        drift = float(np.max(np.abs(positions - world.device_read(name)["positions"])))
        report["stop"][name] = {"drift_rad": drift}
        require(drift < TOLERANCES["stop_drift_rad"], f"{name}: stop drift {drift}")
    base_velocity = float(np.max(np.abs(world.device_read("base")["velocities"])))
    report["stop"]["base"] = {"velocity": base_velocity}
    require(base_velocity < TOLERANCES["base_stop_m_s"], "Base stop did not arrest motion")
    report["mounts_moved"] = verify_mounts(world, models)
    report["projection_moved"] = projection(world, models, out, "moved")
    report["checks"].update(stop_scope=True, rendered_projection=True, dynamic_mounts=True)
    report["contact"] = contacts(world, out)
    report["checks"]["contact_release"] = True
    clock = world.owner_state()["state"]["roboenv_clock"]
    require(
        abs(clock["elapsed_native_time_s"] - world.control.frame * world.control_interval) < 1e-5,
        "Native time differs from declared control intervals",
    )
    report["clock"] = clock
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
    report.update(passed=True, frames=world.control.frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kinex.resolve() / "python"))
    faulthandler.dump_traceback_later(180, repeat=True)
    report = {
        "passed": False,
        "schema": "roboenv/behavior-conformance/1",
        "checks": {},
        "embodiment": "r1pro",
        "seed": args.seed,
        "activity_instance_id": args.instance,
        "realtime": args.realtime,
        "scope": "owner native calibration fixtures; no task success claim",
    }
    world = None
    try:
        world = CalibrationWorld(
            "turning_on_radio",
            "house_double_floor_lower",
            args.instance,
            max_steps=500,
            realtime=args.realtime,
        )
        world.reset(seed=args.seed)
        check(world, out, report)
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if world is not None:
            world.close()
        import omnigibson as og

        og.shutdown()


if __name__ == "__main__":
    main()
