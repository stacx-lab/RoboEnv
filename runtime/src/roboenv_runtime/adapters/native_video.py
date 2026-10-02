"""Owner-only native camera rendering and bounded-memory MP4 encoding during replay."""

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from roboenv.recording import file_digest


class Video:
    def __init__(self, path, width, height, fps):
        self.shape = (height, width, 3)
        self.log = Path(path).with_suffix(".ffmpeg.log").open("xb")
        try:
            self.process = subprocess.Popen([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
                "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ], stdin=subprocess.PIPE, stderr=self.log)
        except Exception:
            self.log.close()
            raise

    def append(self, frame):
        if frame.dtype != np.uint8 or frame.shape != self.shape:
            raise ValueError("Native renderer returned the wrong RGB dimensions or dtype")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        try:
            self.process.stdin.close()
            if self.process.wait(timeout=120):
                raise RuntimeError("Native video encoder failed; inspect its ffmpeg log")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


class SapienCamera:
    def __init__(self, world, name, width, height, expected):
        self.world, self.source = world, world._camera(name)
        near, far = self.source.get_near(), self.source.get_far()
        self.camera = world.task.scene.add_camera(
            "roboenv_replay_" + name, width, height, 1.0, near, far)
        try:
            # SAPIEN uses pixel edges; RoboEnv profiles use integer pixel centers.
            self.camera.set_perspective_parameters(
                near, far, expected[0, 0], expected[1, 1],
                expected[0, 2] + 0.5, expected[1, 2] + 0.5, expected[0, 1])
        except Exception:
            self.close()
            raise
        self.expected = expected

    def read(self):
        # Native wrist-camera poses follow the original task's render hook.
        self.world.task._update_render()
        self.camera.entity.set_pose(self.source.get_global_pose())
        self.world.task.scene.update_render()
        self.camera.take_picture()
        actual = self.camera.get_intrinsic_matrix().copy()
        actual[:2, 2] -= 0.5
        if not np.allclose(actual, self.expected, atol=1e-3, rtol=1e-6):
            raise ValueError("Replay camera projection differs from the native camera")
        pixels = np.asarray(self.camera.get_picture("Color"))
        return (np.clip(pixels[..., :3], 0, 1) * 255).astype(np.uint8)

    def close(self):
        self.world.task.scene.remove_camera(self.camera)


class IsaacCamera:
    def __init__(self, world, backend, name, width, height, expected):
        import omni.replicator.core as rep

        self.world, self.backend = world, backend
        source = (world._camera(name) if backend == "robodojo"
                  else world.robot.sensors[world.cameras[name]])
        if backend == "behavior":
            import omnigibson as og
            context = og.sim.editing_usd()
        else:
            context = nullcontext()
        with context:
            self.product = rep.create.render_product(str(source.prim_path), (width, height), force_new=True)
            self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self.parameters = rep.AnnotatorRegistry.get_annotator("camera_params", device="cpu")
            for annotator in (self.rgb, self.parameters):
                annotator.attach(self.product)
        self.width, self.height, self.expected = width, height, expected
        self.render_subframes = 8 if backend == "robodojo" else 1

    def read(self):
        # Render without stepping physics. A separate product keeps the original
        # sensor, pose, aperture, and evaluation observation resolution intact.
        # RoboDojo's temporal antialiasing needs several renders of the same
        # physics state, even when cached RGB already has the expected shape.
        for subframe in range(12 if self.backend == "robodojo" else 8):
            if self.backend == "robodojo":
                self.world._env.render()
            else:
                import omnigibson as og
                og.sim.render()
            params = self.parameters.get_data()
            pixels = np.asarray(self.rgb.get_data())
            if (subframe + 1 >= self.render_subframes
                    and tuple(params.get("renderProductResolution", ())) == (self.width, self.height)
                    and pixels.shape[:2] == (self.height, self.width)):
                break
        else:
            raise RuntimeError("Native replay render product did not produce a frame")
        projection = np.asarray(params["cameraProjection"]).reshape(4, 4)
        if (params.get("cameraModel") != "pinhole" or not np.isfinite(projection).all()
                or not np.allclose([projection[0, 1], projection[1, 0],
                                    projection[2, 0], projection[2, 1]], 0, atol=1e-6)):
            raise ValueError("Unsupported native replay camera projection")
        actual = np.array([[projection[0, 0] * self.width / 2, 0, self.width / 2],
                           [0, projection[1, 1] * self.height / 2, self.height / 2], [0, 0, 1]])
        if not np.allclose(actual, self.expected, atol=1e-2, rtol=1e-5):
            raise ValueError("Replay render product projection differs from the native camera")
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint8:
            raise ValueError("Invalid native replay RGB frame")
        return pixels[..., :3].copy()

    def close(self):
        # Isaac's shared SyntheticData graph can hang on independent detach.
        # The dedicated replay process releases it with adapter.close().
        pass


class NativeVideo:
    """Stream sampled native control frames; never keep an episode of RGB in RAM."""

    def __init__(self, world, backend, directory, *, height=1080, stride=1, cameras=None):
        if type(height) is not int or height < 2 or height % 2:
            raise ValueError("Render height must be a positive even integer")
        if type(stride) is not int or stride < 1:
            raise ValueError("Render stride must be a positive integer")
        self.world, self.directory, self.stride = world, Path(directory), stride
        self.interval = float(world.control_interval)
        if not np.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("Native control interval must be positive and finite")
        available = [name for name, fact in world.device_facts.items() if fact["type"] in {"rgb", "rgbd"}]
        names = list(cameras) if cameras is not None else available[:2]
        if not names or len(set(names)) != len(names) or set(names) - set(available):
            raise ValueError("Select distinct camera names from the native device catalog")
        self.material_overrides = world.prepare_replay_materials() if backend == "robodojo" else []
        self.renderers, self.videos, self.cameras = [], [], []
        self.count, self.last_frame, self.closed = 0, None, False
        self.frames = (self.directory / "frames.json").open("x")
        self.frames.write("[\n")
        try:
            for index, name in enumerate(names):
                source = world.device_facts[name]["intrinsics"]["color"]
                width = int(round(height * source["width"] / source["height"] / 2)) * 2
                if width < 2:
                    raise ValueError("Native camera has invalid dimensions")
                p = source["projection"]
                expected = np.array([[p["fx"], p.get("skew", 0), p["cx"]],
                                     [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=float)
                scale_x, scale_y = width / source["width"], height / source["height"]
                expected[0] *= scale_x
                expected[1] *= scale_y
                if backend == "robotwin":
                    # Scale pixel edges before returning to integer pixel centers.
                    expected[0, 2] += (scale_x - 1) / 2
                    expected[1, 2] += (scale_y - 1) / 2
                renderer = (SapienCamera(world, name, width, height, expected) if backend == "robotwin"
                            else IsaacCamera(world, backend, name, width, height, expected))
                self.renderers.append(renderer)
                filename = f"camera_{index}.mp4"
                self.videos.append(Video(self.directory / filename, width, height, 1 / (self.interval * stride)))
                self.cameras.append({"name": name, "file": filename, "width": width, "height": height,
                                     "source_width": source["width"], "source_height": source["height"],
                                     "intrinsic_matrix": expected.tolist()})
                if backend == "robodojo":
                    self.cameras[-1]["render_subframes"] = renderer.render_subframes
        except BaseException:
            self.close()
            raise

    def capture(self, *, force=False):
        frame = self.world.control.frame
        if frame == self.last_frame or not force and frame % self.stride:
            return
        for renderer, video in zip(self.renderers, self.videos, strict=True):
            video.append(renderer.read())
        row = {"index": self.count, "frame": frame, "simulation_seconds": frame * self.interval}
        self.frames.write((",\n" if self.count else "") + json.dumps(row))
        self.last_frame = frame
        self.count += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for resource in [*self.videos, *self.renderers]:
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        self.frames.write("\n]\n")
        self.frames.close()
        if errors:
            raise RuntimeError("Native media cleanup failed") from errors[0]

    def receipt(self):
        if not self.closed or not self.count:
            raise ValueError("Native video is not finalized")
        files = ["frames.json", *(camera["file"] for camera in self.cameras)]
        return {"schema": "roboenv/native-video/1", "kind": "native-command-replay",
                "frames": self.count, "cameras": self.cameras,
                "material_overrides": self.material_overrides,
                "fps": 1 / (self.interval * self.stride), "stride": self.stride,
                "waiting_policy": "Native control frames retained; wall-clock gaps without native steps omitted",
                "files": {name: file_digest(self.directory / name) for name in files}}
