"""Spectator video for owner command replay; never part of agent observations."""

import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roboenv_runtime.adapters.isaaclab.go2_d1.kinematics import matrix_quat


class TaskVideo:
    def __init__(self, world, path):
        import isaaclab.sim as sim
        from isaaclab.sensors import Camera, CameraCfg

        self.world = world
        self.path = path
        self.eye = np.array([1.6, -2.2, 1.4])
        forward = np.array([0.45, 0, 0.30]) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.orientation = matrix_quat(np.column_stack((right, up, -forward)))
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/ReplaySpectator",
                width=960,
                height=540,
                data_types=["rgb"],
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(self.eye), rot=tuple(self.orientation), convention="opengl"
                ),
                spawn=sim.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=30.0,
                    vertical_aperture=16.875,
                    clipping_range=(0.05, 30.0),
                ),
            )
        )
        self.camera._initialize_callback(None)
        self.encoder = subprocess.Popen(
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
                "960x600",
                "-r",
                "25",
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
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        self.stages = set()

    def capture(self):
        n = self.world.native
        offset = n.root_pose()[:3]
        offset[2] = 0
        self.camera.set_world_poses(
            positions=(self.eye + offset)[None],
            orientations=self.orientation[None],
            convention="opengl",
        )
        n.sim.render()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"].torch[0, :, :, :3].cpu().numpy()
        frame = Image.new("RGB", (960, 600), "#111827")
        frame.paste(Image.fromarray(rgb), (0, 0))
        draw = ImageDraw.Draw(frame)
        stage = n.task.stage
        draw.text(
            (15, 548),
            f"{self.world.task_name} | {stage} | t={n.steps * 0.02:.2f}s",
            font=self.font,
            fill="white",
        )
        draw.text(
            (15, 574),
            "Isaac Lab physics | base gait + independent arm IK | command replay, 1x speed",
            font=self.font,
            fill="#bbc7db",
        )
        self.encoder.stdin.write(frame.tobytes())
        if stage not in self.stages or n.steps == 1600:
            frame.save(self.path.with_name(self.path.stem + f"-{stage}-{n.steps}.png"))
            self.stages.add(stage)

    def close(self):
        self.encoder.stdin.close()
        if self.encoder.wait(timeout=60) != 0:
            raise RuntimeError("Task video encoding failed")
