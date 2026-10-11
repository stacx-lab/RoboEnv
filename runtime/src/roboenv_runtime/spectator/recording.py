"""Live spectator sampling in the owner lane, bounded encoding in a child process."""

import contextlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import shutil
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path

from roboenv.recording import recording_config

from roboenv_runtime.runtime.atomic import atomic_write_json
from roboenv_runtime.spectator import ViewSpec, create_spectator_renderer

log = logging.getLogger(__name__)


def _encode(directory, views, fps, executable, frames, finished, accepted, end_time, pids, reply):
    """No simulator imports or handles cross this process boundary."""
    parent = os.getppid()
    encoders = []
    received = written = represented = 0
    error = None
    try:
        with (directory / "encoder.log").open("wb") as diagnostics:
            # Exercise the actual optional codec before announcing readiness.
            subprocess.run(
                [executable, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "2x2", "-i", "pipe:0", "-frames:v", "1", "-c:v", "libx264",
                 "-threads", "1", "-pix_fmt", "yuv420p", "-f", "null", "-"],
                input=bytes(12), stdout=subprocess.DEVNULL, stderr=diagnostics,
                timeout=10, check=True,
            )
            for index, view in enumerate(views):
                process = subprocess.Popen(
                    [executable, "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{view.width}x{view.height}",
                     "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                     "-threads", "1", "-preset", "veryfast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-movflags", "+frag_keyframe+empty_moov",
                     str(directory / f"{view.name}.mp4")],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=diagnostics,
                )
                encoders.append(process)
                pids[index] = process.pid
            reply.send({"ready": True})
            previous = None
            source_time = 0.0
            with (directory / "frames.jsonl").open("w") as timeline:

                def write_until(stop):
                    nonlocal written, represented
                    start = written
                    while written < stop and previous is not None:
                        for view, encoder in zip(views, encoders, strict=True):
                            encoder.stdin.write(previous[view.name].tobytes())
                        written += 1
                    if written > start:
                        represented += 1
                        timeline.write(json.dumps({
                            "source_time_s": source_time, "frame_start": start,
                            "frame_stop": written,
                        }) + "\n")

                while not finished.value or received < accepted.value:
                    try:
                        timestamp, images = frames.get(timeout=0.1)
                    except queue.Empty:
                        if os.getppid() != parent:
                            raise RuntimeError("recording owner exited") from None
                        continue
                    # An off-grid terminal capture can share the previous sample's
                    # CFR bin. Write that sample before replacing it, even then.
                    write_until(max(written + 1, math.floor(timestamp * fps + 1e-8)))
                    previous, source_time = images, timestamp
                    received += 1
                # Include the endpoint and hold the latest image across dropped samples.
                write_until(max(written + 1, math.floor(end_time.value * fps + 1e-8) + 1))
            for encoder in encoders:
                encoder.stdin.close()
            for encoder in encoders:
                if encoder.wait(timeout=3) != 0:
                    raise RuntimeError("FFmpeg failed; see encoder.log")
    except Exception as exc:
        error = str(exc)
    finally:
        for index, encoder in enumerate(encoders):
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            pids[index] = 0
        with contextlib.suppress(BrokenPipeError, EOFError):
            reply.send({"encoded_frames": written, "received_samples": received,
                        "represented_samples": represented, "error": error})
        reply.close()


class Recorder:
    """One episode, one creating thread. Failures after startup only stop recording."""

    def __init__(self, adapter, directory, config):
        config = recording_config(config)
        self.fps = config["fps"]
        views = None if config["views"] is None else [ViewSpec(**v) for v in config["views"]]
        if views and any(v.height % 2 or (v.width is not None and v.width % 2) for v in views):
            raise ValueError("Recording dimensions must be even for H.264")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValueError("Recording requires FFmpeg with the libx264 encoder on PATH")
        self.directory = Path(config.get("directory", directory))
        self.directory.mkdir(parents=True, mode=0o700, exist_ok="directory" in config)
        self.renderer = create_spectator_renderer(adapter, views)
        self.process = None
        self.closed = False
        self.next_sample = 0.0
        self.last_sample = None
        self.summary = {
            "schema": "roboenv/recording/1", "experimental": True,
            "status": "incomplete", "state": "starting", "clock": "simulation",
            "fps": self.fps, "views": [asdict(v) for v in self.renderer.views],
            "captured_samples": 0, "dropped_samples": 0,
        }
        ctx = mp.get_context("spawn")
        self.frames = ctx.Queue(maxsize=2)
        # Single-writer scalars: a stopped/killed child must never hold a lock
        # needed by the owner to finish recording.
        self.finished = ctx.Value("b", False, lock=False)
        self.accepted = ctx.Value("q", 0, lock=False)
        self.end_time = ctx.Value("d", 0, lock=False)
        self.pids = ctx.Array("q", len(self.renderer.views), lock=False)
        self.reply, child_reply = ctx.Pipe(duplex=False)
        try:
            self._save()
            self.process = ctx.Process(
                target=_encode,
                args=(self.directory, self.renderer.views, self.fps, executable, self.frames,
                      self.finished, self.accepted, self.end_time, self.pids, child_reply),
                name="roboenv-recorder",
            )
            self.process.start()
            child_reply.close()
            if not self.reply.poll(15):
                raise RuntimeError("Recording encoder startup timed out")
            response = self.reply.recv()
            if not response.get("ready"):
                raise RuntimeError(response.get("error") or "Recording encoder failed to start")
            self._sample(0.0)
            self.summary["state"] = "recording"
            self._save()
        except BaseException:
            self.close(0, status="incomplete", error="Recording startup failed", capture=False)
            raise

    def _save(self):
        atomic_write_json(self.directory / "manifest.json", self.summary)

    def _sample(self, timestamp):
        images = self.renderer.capture()
        self.summary["captured_samples"] += 1
        self.last_sample = timestamp
        try:
            self.frames.put_nowait((timestamp, images))
            self.accepted.value += 1
        except queue.Full:
            self.summary["dropped_samples"] += 1
        self.next_sample = (math.floor(timestamp * self.fps + 1e-8) + 1) / self.fps

    def tick(self, timestamp):
        if self.closed:
            return
        try:
            if not self.process.is_alive():
                raise RuntimeError("Recording encoder exited; see encoder.log")
            if timestamp + 1e-8 >= self.next_sample:
                self._sample(timestamp)
        except Exception as exc:
            self.close(timestamp, status="incomplete", error=str(exc), capture=False)

    def close(self, timestamp, *, status="complete", error=None, capture=True):
        if self.closed:
            return
        self.closed = True
        try:
            if capture and self.last_sample != timestamp and self.process.is_alive():
                self._sample(timestamp)
        except Exception as exc:
            error = str(exc)
        self.end_time.value = timestamp
        self.finished.value = True
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                error = error or "Recording encoder exceeded the 5 second drain deadline"
                # Unblock pipe writes first, so the encoder can reap its own children.
                for pid in self.pids:
                    if pid:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                with contextlib.suppress(ProcessLookupError):
                    os.kill(self.process.pid, signal.SIGCONT)
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.5)
            for pid in self.pids:
                if pid:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
            try:
                if self.reply.poll():
                    response = self.reply.recv()
                    self.summary.update(response)
                    error = error or response.get("error")
                else:
                    error = error or "Recording encoder exited without a final report"
            except EOFError:
                error = error or "Recording encoder exited without a final report"
        self.frames.cancel_join_thread()
        self.frames.close()
        self.reply.close()
        try:
            self.renderer.close()
        except Exception as exc:
            error = error or str(exc)
        self.summary.update(state="closed", status="incomplete" if error else status,
                            error=error, end_time_s=timestamp, accepted_samples=self.accepted.value)
        try:
            self._save()
        except OSError as exc:
            error = error or str(exc)
            self.summary.update(status="incomplete", error=error)
        if error:
            log.warning("Recording incomplete: %s", error)
