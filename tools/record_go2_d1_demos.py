"""Record real Isaac Go2–D1 velocity tracking with a spectator camera and telemetry."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roboenv_runtime.adapters.isaaclab.go2_d1.adapter import matrix_quat, quat_matrix
from roboenv_runtime.adapters.isaaclab.go2_d1.environment import Go2Env

CASES = {
    "forward-backward": (0, 0.08, -0.06, "Forward / backward", "vx", "m/s"),
    "sideways": (1, 0.05, -0.05, "Sideways", "vy", "m/s"),
    "turning": (2, 0.18, -0.18, "Turning", "wz", "rad/s"),
}
FAST_CASES = {
    "forward-backward": (0, 1.0, -1.0, "Forward / backward", "vx", "m/s"),
    "sideways": (1, 0.5, -0.5, "Sideways", "vy", "m/s"),
    "turning": (2, math.pi / 6, -math.pi / 6, "Turning", "wz", "deg/s"),
}
POSTURE_CASES = {
    "height": (0, 0.34, 0.26, "Body height", "height", "m"),
    "roll": (1, 0.12, -0.12, "Body roll", "roll", "deg"),
    "pitch": (2, 0.12, -0.12, "Body pitch", "pitch", "deg"),
}
FPS = 25
DURATION = 22


def command_at(t, positive, negative, high_speed=False):
    if 2 <= t < 10:
        return positive
    if (13 if high_speed else 12) <= t < (21 if high_speed else 20):
        return negative
    return 0.0


def compose(rgb, title, axis, units, history, limit, duration=DURATION, center=0.0, posture=False):
    frame = Image.new("RGB", (960, 720), "#111827")
    frame.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(frame)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 14)
    large = ImageFont.truetype(font_path, 24)
    t, goal, measured = history[-1]
    draw.rectangle((0, 0, 960, 76), fill="#111827")
    draw.text((22, 9), f"Go2 + D1-550  |  {title}", font=large, fill="white")
    draw.text(
        (22, 44),
        "Isaac Lab physics  |  leg gait + IK  |  1x simulation time",
        font=small,
        fill="#bbc7db",
    )
    draw.text((780, 20), f"t = {t:05.2f} s", font=font, fill="white")
    draw.text((22, 547), f"{axis} command: {goal:+.3f} {units}", font=font, fill="#38d9f8")
    draw.text((390, 547), f"measured: {measured:+.3f} {units}", font=font, fill="#ffcc66")
    draw.text(
        (22, 576),
        ("Body posture" if posture else "Body-frame velocity")
        + "; measured curve uses a 0.25 s low-pass filter",
        font=small,
        fill="#bbc7db",
    )
    x0, x1, y0, y1 = 65, 935, 607, 692

    def point(time, value):
        return (
            x0 + (x1 - x0) * time / duration,
            (y0 + y1) / 2 - np.clip((value - center) / limit, -1.0, 1.0) * (y1 - y0) / 2,
        )

    for value in (center - limit, center, center + limit):
        y = point(0, value)[1]
        draw.line((x0, y, x1, y), fill="#344155")
        draw.text((8, y - 8), f"{value:+.2f}", font=small, fill="#bbc7db")
    for tick in range(0, int(duration) + 1, 2):
        x = point(tick, 0)[0]
        draw.text((x - 4, 699), str(tick), font=small, fill="#bbc7db")
    if len(history) > 1:
        draw.line([point(row[0], row[1]) for row in history], fill="#38d9f8", width=2)
        draw.line([point(row[0], row[2]) for row in history], fill="#ffcc66", width=2)
    return frame


def run(out, cases, smoke=False, high_speed=False, posture=False, walk_speed=0.0):
    out.mkdir(parents=True, exist_ok=False)
    duration = 24 if high_speed or posture else DURATION
    world = Go2Env()
    native = world.native
    try:
        import isaaclab.sim as sim
        from isaaclab.sensors import Camera, CameraCfg

        eye = np.array([1.5, -2.0, 1.4])
        forward = np.array([0.20, 0.0, 0.30]) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        orientation = tuple(matrix_quat(np.column_stack((right, up, -forward))))
        camera = Camera(
            CameraCfg(
                prim_path="/World/DemoSpectator",
                offset=CameraCfg.OffsetCfg(pos=tuple(eye), rot=orientation, convention="opengl"),
                width=960,
                height=540,
                data_types=["rgb"],
                spawn=sim.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=30.0,
                    vertical_aperture=16.875,
                    clipping_range=(0.05, 30.0),
                ),
            )
        )
        # The spectator is added after the native world has booted. Initialize
        # only its sensor handles; do not restart the physics backend.
        camera._initialize_callback(None)
        for case in cases:
            index, positive, negative, title, axis, units = (
                POSTURE_CASES if posture else FAST_CASES if high_speed else CASES
            )[case]
            display_scale = 180 / math.pi if units in ("deg/s", "deg") else 1.0
            world.reset(0)
            for _ in range(4):
                native.sim.render()
            history, samples = [], []
            filtered = np.array([0.30, 0.0, 0.0]) if posture else np.zeros(3)
            path = out / (case + ".mp4")
            encoder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "960x720",
                    "-r",
                    str(FPS),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
            try:
                for frame in range(10 if smoke else duration * FPS):
                    for _ in range(2):
                        t = world.steps * native.interval
                        goal = command_at(t, positive, negative, high_speed or posture)
                        if posture and index == 0 and goal == 0.0:
                            goal = 0.30
                        posture_target = [0.30, 0.0, 0.0]
                        velocity = [0.0, 0.0, 0.0]
                        if posture:
                            posture_target[index] = goal
                            velocity[0] = walk_speed if 2 <= t < 21 else 0.0
                        else:
                            velocity[index] = goal
                        world.set_target(
                            "base",
                            {
                                "mode": "locomotion",
                                "velocity": dict(zip(("vx", "vy", "wz"), velocity, strict=True)),
                                "posture": dict(
                                    zip(("height", "roll", "pitch"), posture_target, strict=True)
                                ),
                            },
                        )
                        world.advance()
                        lin = native.array(native.robot.data.root_lin_vel_b)
                        ang = native.array(native.robot.data.root_ang_vel_b)
                        measured = np.array([lin[0], lin[1], ang[2]])
                        if posture:
                            root = native.root_pose()
                            rotation = quat_matrix(root[3:])
                            measured = np.array(
                                [
                                    root[2],
                                    np.arctan2(rotation[2, 1], rotation[2, 2]),
                                    np.arcsin(np.clip(-rotation[2, 0], -1, 1)),
                                ]
                            )
                        filtered += native.interval / 0.25 * (measured - filtered)
                        samples.append(
                            {
                                "time_s": world.steps * native.interval,
                                "velocity_command": velocity,
                                "command": posture_target if posture else velocity,
                                "measured": measured.tolist(),
                                "filtered": filtered.tolist(),
                                "root": native.root_pose().tolist(),
                            }
                        )
                        if world.done:
                            raise ValueError(f"{case}: robot fell at {t:.2f}s")
                    if high_speed or walk_speed:
                        offset = native.root_pose().copy()[:3]
                        offset[2] = 0.0
                        camera.set_world_poses(
                            positions=(eye + offset)[None],
                            orientations=np.array(orientation)[None],
                            convention="opengl",
                        )
                    native.sim.render()
                    camera.update(0.0, force_recompute=True)
                    rgb = camera.data.output["rgb"].torch[0, :, :, :3].cpu().numpy()
                    history.append(
                        (
                            world.steps * native.interval,
                            goal * display_scale,
                            filtered[index] * display_scale,
                        )
                    )
                    image = compose(
                        rgb,
                        title + (f" | walking {walk_speed:g} m/s" if walk_speed else ""),
                        axis,
                        units,
                        history,
                        (
                            0.06
                            if posture and index == 0
                            else max(abs(positive), abs(negative)) * display_scale * 1.6
                        ),
                        duration,
                        center=0.30 if posture and index == 0 else 0.0,
                        posture=posture,
                    )
                    encoder.stdin.write(image.tobytes())
                    if frame == (5 if smoke else 200):
                        image.save(out / (case + "-preview.png"))
                    if frame % 100 == 0:
                        print(case, frame, "/", duration * FPS, flush=True)
            finally:
                encoder.stdin.close()
                if encoder.wait(timeout=60) != 0:
                    raise RuntimeError("ffmpeg failed")
            (out / (case + "-telemetry.json")).write_text(json.dumps(samples, indent=2))
            (out / (case + "-record.json")).write_text(json.dumps(world.record(), indent=2))
            print("VIDEO_COMPLETE", path, flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=(*CASES, *POSTURE_CASES))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--high-speed",
        action="store_true",
        help="1 m/s, 0.5 m/s and 30 deg/s, with a following camera",
    )
    parser.add_argument(
        "--posture", action="store_true", help="Record height, roll and pitch endpoints"
    )
    parser.add_argument(
        "--walk-speed", type=float, default=0.0, help="Forward speed during posture commands"
    )
    args = parser.parse_args()
    selected = POSTURE_CASES if args.posture else CASES
    if args.high_speed and args.posture:
        parser.error("--high-speed and --posture are mutually exclusive")
    if args.walk_speed and (not args.posture or not -1 <= args.walk_speed <= 1):
        parser.error("--walk-speed requires --posture and a speed in [-1, 1]")
    if args.cases and any(case not in selected for case in args.cases):
        parser.error("cases must match the selected recording mode")
    run(
        args.out,
        args.cases or list(selected),
        args.smoke,
        args.high_speed,
        args.posture,
        args.walk_speed,
    )
