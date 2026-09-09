"""Native Go2 observations in device-local SI units; no physics stepping or policy."""

from pathlib import Path

from .geometry import relative_pose, rotation
from .rig import FEET


def body_state(robot, joints, feet_ids, stamp):
    data = robot.data
    root = data.root_link_pose_w.torch[0].cpu().numpy()
    feet = data.body_link_pose_w.torch[0, feet_ids].cpu().numpy()
    return {
        "positions": data.joint_pos.torch[0, joints].cpu().tolist(),
        "velocities": data.joint_vel.torch[0, joints].cpu().tolist(),
        "efforts": [], "pose": None, "body_velocities": [],
        "frame_poses": {
            name: relative_pose(root, pose) for name, pose in zip(FEET, feet, strict=True)
        },
        "faults": [],
        "sample_time_s": stamp,
    }


def imu_state(sensor, stamp):
    data = sensor.data
    return {
        "angular_velocity": dict(zip(("x", "y", "z"), data.ang_vel_b.torch[0].cpu().tolist())),
        "linear_acceleration": dict(zip(("x", "y", "z"), data.lin_acc_b.torch[0].cpu().tolist())),
        "orientation": None, "magnetic_field": None, "sample_time_s": stamp,
    }


def contact_state(sensor, robot, foot_id, stamp):
    q = robot.data.body_link_pose_w.torch[0, foot_id, 3:].cpu().numpy()
    force = sensor.data.net_forces_w.torch[0, 0].cpu().numpy()
    return {"force": dict(zip(("x", "y", "z"), (rotation(q).T @ force).tolist())),
            "torque": None, "sample_time_s": stamp}


def camera_state(camera, directory, stamp):
    from roboenv_runtime.runtime.worker import write_png
    from roboenv_runtime.bindings.backends.shared.depth import write_depth
    import uuid

    camera.update(0.0, force_recompute=True)
    outputs = camera.data.output
    rgb = outputs["rgb"].torch[0, :, :, :3].cpu().numpy()
    depth = outputs["distance_to_image_plane"].torch[0].cpu().numpy().squeeze(-1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{uuid.uuid4().hex}.png"
    h, w, _ = write_png(file, rgb)
    write_depth(file.with_suffix(".pgm"), depth)
    return {
        "file": str(file),
        "depth_file": str(file.with_suffix(".pgm")),
        "height": h,
        "width": w,
        "sample_time_s": stamp,
        "depth_width": w, "depth_height": h,
        "depth_sample_time_s": stamp,
    }
