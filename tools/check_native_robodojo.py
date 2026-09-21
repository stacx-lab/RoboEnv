"""Owner-only RoboDojo conformance in a fresh native process for each rig and clock.

Physical calibration fixtures are kept outside controller workspaces and do not
establish task success. RGB cameras do not advertise or validate a depth stream.
"""

import argparse
import copy
import faulthandler
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.native_record import digest
from roboenv_runtime.adapters.robodojo.devices import RoboDojoDevices, pose_matrix
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
    "projection_px": 2.0,
    "contact_impulse_ns": 1e-6,
    "unsupported_grip_displacement_m": 0.01,
    "released_displacement_m": 0.03,
}
CAMERAS = ("head_camera", "left_camera", "right_camera")
ARMS = ("left_arm", "right_arm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(world, commands):
    frame = world.control.frame
    for name, target in commands.items():
        world.device_write(name, target)
    if world.realtime:
        require(world.control.frame == frame, "Realtime writes advanced physics")
        world.control.tick()
    expected = 1 if world.realtime else len(commands)
    require(world.control.frame == frame + expected, "Unexpected control interval count")


def mount_transform(fact):
    value = fact["extrinsics"]["mount"]
    return matrix(value["translation_m"] + value["rotation_xyzw"])


def projection(world, models, out, phase):
    import omni.replicator.core as rep
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.prims import delete_prim

    report = {}
    for name in CAMERAS:
        fact = world.device_facts[name]
        transform = mount_transform(fact)
        if name != "head_camera":
            arm = name.split("_", 1)[0] + "_arm"
            transform = (
                mount_transform(world.device_facts[arm])
                @ matrix(models[arm].fk(world.device_read(arm)["positions"])["pose"])
                @ transform
            )
        xyz, half = np.array([0.025, -0.018, 0.18]), 0.006
        pose = matrix_pose(transform)
        marker_path = "/World/roboenv_projection_" + name
        FixedCuboid(
            prim_path=marker_path,
            name="roboenv_projection_" + name,
            position=(transform @ np.r_[xyz, 1])[:3],
            orientation=np.array([pose[6], *pose[3:6]]),
            size=half * 2,
            color=np.array([1.0, 0.0, 1.0]),
        )
        camera = world._camera(name)
        index = world._env.camera_manager.cameras[0].index(camera)
        product = world._env.capture_manager.tiled_render_products[index]
        annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", device="cpu")
        annotator.attach(product)
        try:
            before = world.owner_state()
            for _ in range(8):
                rgb, depth = world.device_image(name)
            require(depth is None, "RoboDojo RGB device unexpectedly returned depth")
            labels = annotator.get_data()
            ids = [
                int(key)
                for key, value in labels["info"]["idToLabels"].items()
                if str(value).startswith(marker_path)
            ]
            ys, xs = np.where(np.isin(np.asarray(labels["data"]).squeeze(), ids))
            require(len(xs) >= 20, f"{name}: calibration marker not rendered; labels={ids}")
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
            report[name] = {
                "expected_uv": expected.tolist(),
                "observed_uv": observed.tolist(),
                "projection_error_px": error,
                "marker_pixels": len(xs),
            }
            write_png(out / f"{phase}-{name}.png", rgb)
            (out / f"{phase}-{name}.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(error <= TOLERANCES["projection_px"], f"{name}: projection error {error}")
            require(world.owner_state() == before, "Rendering advanced native state")
        finally:
            annotator.detach(product)
            delete_prim(marker_path)
    return report


def contacts(world, out):
    import carb
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

    report = {}
    for side in ("left", "right"):
        name = side + "_gripper"
        robot, key, _ = world.actuators[name]
        manager = world._env.robot_manager
        for _ in range(25):
            interval(world, {name: [1.0]})
        # These locations select the physical pad region of each source model.
        # The contact reports, not this placement or proximity, prove clamping.
        ee = pose_matrix(manager.get_real_endpose(robot, [0], is_relative=False)[0])
        # The source-model distal finger band clears the fixture when open.
        offset = [robot.gripper_bias, 0.0, 0.0] if world.embodiment == "arx_x5" else [0, 0, -0.125]
        pose = matrix_pose(ee)
        marker_path = "/World/roboenv_contact_" + side
        fixture = DynamicCuboid(
            prim_path=marker_path,
            name="roboenv_contact_" + side,
            position=(ee @ np.r_[offset, 1])[:3],
            orientation=np.array([pose[6], *pose[3:6]]),
            size=0.024,
            mass=0.02,
            color=np.array([1.0, 0.0, 1.0]),
        )
        fingers = ("link7", "link8") if world.embodiment == "arx_x5" else ("tip_left", "tip_right")
        links = key.root_physx_view.link_paths[0]
        finger_paths = {
            path: path.rsplit("/", 1)[-1] for path in links if path.rsplit("/", 1)[-1] in fingers
        }
        require(set(finger_paths.values()) == set(fingers), f"Missing native fingers: {links}")
        stage = get_current_stage()
        body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(marker_path))
        body.CreateKinematicEnabledAttr().Set(True)
        settings = carb.settings.get_settings()
        previous = settings.get("/physics/disableContactProcessing")
        # Isaac Lab disables CPU contact reports by default. Enable observation
        # for this owner fixture; this does not alter collision solving.
        settings.set_bool("/physics/disableContactProcessing", False)
        for path in finger_paths:
            prim = stage.GetPrimAtPath(path)
            require(prim.IsValid(), f"Missing native finger: {path}")
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0)
        PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr().Set(0)
        diagnostic = {
            "initial_open": world.device_read(name),
            "native_gripper_config": {"move": robot.gripper_move, "scale": robot.gripper_scale},
            "finger_poses_in_flange": {
                finger: matrix_pose(
                    np.linalg.inv(ee)
                    @ pose_matrix(manager.get_link_pose(robot, finger, [0], is_relative=False)[0])
                )
                for finger in fingers
            },
            "fixture_pose_in_flange": list(offset),
        }
        (out / f"{name}-placement.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        touching, events = {}, []

        def on_contact(
            headers,
            data,
            marker_path=marker_path,
            finger_paths=finger_paths,
            touching=touching,
            events=events,
        ):
            for header in headers:
                paths = [
                    str(PhysicsSchemaTools.intToSdfPath(value))
                    for value in (header.actor0, header.actor1)
                ]
                if marker_path not in paths:
                    continue
                other = paths[1] if paths[0] == marker_path else paths[0]
                if other not in finger_paths:
                    continue
                finger = finger_paths[other]
                offset = header.contact_data_offset
                points = [data[i] for i in range(offset, offset + header.num_contact_data)]
                impulses = [float(np.linalg.norm(point.impulse)) for point in points]
                loaded = sum(value > TOLERANCES["contact_impulse_ns"] for value in impulses)
                # PhysX may keep a speculative manifold within contactOffset
                # after the fingers separate. Only solver impulses prove load.
                if header.type == ContactEventType.CONTACT_LOST or not loaded:
                    touching.pop(finger, None)
                else:
                    touching[finger] = loaded
                events.append(
                    {
                        "finger": finger,
                        "event": int(header.type),
                        "contacts": int(header.num_contact_data),
                        "impulses_ns": impulses,
                        "separations_m": [float(point.separation) for point in points],
                        "physics_step": world.physics_steps,
                    }
                )

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        try:
            # Register reports before the new kinematic actor/pairs enter PhysX.
            interval(world, {name: [1.0]})
            fixture.initialize()
            for _ in range(40):
                interval(world, {name: [0.0]})
            closed = dict(touching)
            measured = world.device_read(name)
            report[name] = {"contact": closed, "measured_contact": measured, "events": events}
            (out / f"{name}-contact.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(
                set(closed) == set(fingers),
                f"{name}: both fingers did not contact fixture: {closed}",
            )
            # Pinned contact alone can pass for a wedge that ejects a free body.
            # Remove the fixture support and require actual dynamic retention.
            clamped = np.asarray(fixture.get_world_pose()[0])
            body.GetKinematicEnabledAttr().Set(False)
            held = []
            for _ in range(30):
                interval(world, {name: [0.0]})
                held.append(np.asarray(fixture.get_world_pose()[0]).tolist())
            displacement = float(np.max(np.linalg.norm(np.asarray(held) - clamped, axis=1)))
            report[name].update(
                fixture_mass_kg=0.02,
                fixture_size_m=0.024,
                clamped_position=clamped.tolist(),
                unsupported_positions=held,
                unsupported_displacement_m=displacement,
            )
            (out / f"{name}-contact.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(
                displacement < TOLERANCES["unsupported_grip_displacement_m"],
                f"{name}: free fixture slipped or was expelled: {displacement}",
            )
            frame = world.control.frame
            world.device_stop(name)
            require(world.control.frame == frame, "Gripper stop advanced physics")
            for _ in range(30):
                interval(world, {name: [1.0]})
            report[name].update(release=dict(touching), measured_open=world.device_read(name))
            released = np.asarray(fixture.get_world_pose()[0])
            release_displacement = float(np.linalg.norm(released - np.asarray(held[-1])))
            report[name].update(
                released_position=released.tolist(), release_displacement_m=release_displacement
            )
            (out / f"{name}-contact.json").write_text(json.dumps(report[name], indent=2) + "\n")
            require(not touching, f"{name}: physical fixture was not released")
            require(
                release_displacement > TOLERANCES["released_displacement_m"],
                f"{name}: opened fingers did not release the unsupported fixture",
            )
        finally:
            del subscription
            settings.set_bool("/physics/disableContactProcessing", previous)
            # Removing a dynamic actor invalidates Isaac's shared tensor view,
            # including the robot articulation. Keep it inert until world close.
            body.GetKinematicEnabledAttr().Set(True)
            UsdPhysics.CollisionAPI(body.GetPrim()).GetCollisionEnabledAttr().Set(False)
            fixture.set_visibility(False)
            fixture.set_world_pose(position=np.array([0.0, 0.0, -10.0]))
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
    models = {name: JointDevice({"name": name, **world.device_facts[name]}) for name in ARMS}
    before, frame = world.owner_state(), world.control.frame
    for name in CAMERAS:
        for _ in range(2):
            rgb, depth = world.device_image(name)
            require(rgb.dtype == np.uint8 and rgb.std() > 1 and depth is None, "Invalid native RGB")
    report["geometry"] = {}
    for name, model in models.items():
        reading = world.device_read(name)
        q = model.project(reading["positions"])
        fk_error = pose_error(
            model.fk(reading["positions"])["pose"], reading["frame_poses"]["flange"]
        )
        target_q = q.copy()
        target_q[0] += 0.03
        target = model.fk(model.project(target_q))["pose"]
        solved = model.ik(target, q)
        ik_error = pose_error(model.fk(solved["joints"])["pose"], target)
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
    require(world.owner_state() == before and world.control.frame == frame, "Read/geometry effects")
    report["checks"]["read_and_geometry_invariance"] = True
    report["projection_initial"] = projection(world, models, out, "initial")
    goals = {name: models[name].project(world.device_read(name)["positions"]) for name in ARMS}
    origins = copy.deepcopy(goals)
    for name, value in goals.items():
        value[0] += 0.1
        goals[name] = models[name].project(value)
        # Exercise FK after motion at an interior configuration. Native motors
        # can have tiny soft-limit overshoot; Kinex FK accepts legal joint inputs.
        # Stop still holds the unmodified measurement and its drift is measured.
        for index, bounds in enumerate(world.control.limits[name]):
            if bounds is not None:
                low, high = bounds
                margin = min(0.05, (high - low) / 4)
                goals[name][index] = min(high - margin, max(low + margin, goals[name][index]))
    state = copy.deepcopy((world.owner_state(), world.control.targets, world.control.commands))
    for target in ([float("nan")] * 6, [float("inf")] * 6, [0.0], [True] * 6, [1e9] * 6):
        try:
            world.device_write("left_arm", target)
        except ValueError:
            pass
        else:
            raise ValueError("Malformed target accepted")
        require(
            (
                world.owner_state(),
                world.control.targets,
                world.control.commands,
            )
            == state,
            "Malformed target caused effects",
        )
    interval(world, goals)
    for _ in range(49):
        interval(world, goals)
    report["motion"] = {}
    for name, goal in goals.items():
        reading = world.device_read(name)
        error = float(np.max(np.abs(np.asarray(reading["positions"]) - goal)))
        movement = float(np.max(np.abs(np.asarray(reading["positions"]) - origins[name])))
        velocity = float(np.max(np.abs(reading["velocities"])))
        report["motion"][name] = {
            "error_rad": error,
            "movement_rad": movement,
            "velocity_rad_s": velocity,
            "measured_positions": reading["positions"],
        }
        require(
            error < TOLERANCES["arrival_rad"]
            and velocity < TOLERANCES["arrival_velocity_rad_s"]
            and movement > 0.05,
            f"{name}: target acceptance did not produce measured arrival",
        )
    require(
        world.physics_steps - world.episode_start_steps
        == world.control.frame * world.native_interval,
        "Native control decimation mismatch",
    )
    report["checks"].update(
        invalid_zero_effects=True, per_device_intervals=True, measured_arrival=True
    )
    peer, frame = copy.deepcopy(world.control.targets["right_arm"]), world.control.frame
    world.device_stop("left_arm")
    held = world.device_read("left_arm")["positions"]
    require(
        world.control.frame == frame and world.control.targets["right_arm"] == peer,
        "Stop changed time or peer target",
    )
    peer[0] += 0.15
    peer = models["right_arm"].project(peer)
    for _ in range(40):
        interval(world, {"right_arm": peer})
    drift = float(np.max(np.abs(np.asarray(held) - world.device_read("left_arm")["positions"])))
    report["stop_drift_rad"] = drift
    require(drift < TOLERANCES["stop_drift_rad"], f"Stop did not hold measured joints: {drift}")
    report["projection_moved"] = projection(world, models, out, "moved")
    world._verify_cameras()
    report["checks"].update(stop_scope=True, rendered_projection=True, dynamic_mounts=True)
    report["contact"] = contacts(world, out)
    report["checks"]["contact_release"] = True
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
    report.update(passed=True, frames=world.control.frame, physics_steps=world.physics_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embodiment", choices=["arx_x5", "yam"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layout-config", choices=["arx_x5", "yam"])
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.kinex.resolve() / "packages/core/assets/python"))
    faulthandler.dump_traceback_later(180, repeat=True)
    report = {
        "passed": False,
        "schema": "roboenv/robodojo-conformance/1",
        "checks": {},
        "embodiment": args.embodiment,
        "seed": args.seed,
        "realtime": args.realtime,
        "scope": "owner native calibration fixtures; no task success claim",
    }
    world = None
    try:
        world = RoboDojoDevices(
            "stack_bowls",
            args.embodiment,
            max_steps=500,
            realtime=args.realtime,
            layout_config=args.layout_config,
        )
        world.reset(seed=args.seed)
        check(world, out, report)
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        # Isaac shutdown may exit the interpreter. Persist failure before cleanup.
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if world is not None:
            world.close()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
