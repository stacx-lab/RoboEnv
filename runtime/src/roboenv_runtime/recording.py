"""Bounded owner capture; a separate process owns recording I/O and finalization."""

import json
import multiprocessing
import queue
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from roboenv.recording import FINALIZE_TIMEOUT_S, SCHEMA, file_digest, write_json


def _write(directory, manifest, events, counters, finished):
    """No simulator objects or imports cross into the writer process."""
    count = states = 0
    try:
        with (directory / "timeline.jsonl").open("w") as timeline, (
            directory / "commands.jsonl"
        ).open("w") as commands, ExitStack() as stack:
            names = manifest.get("state_files", ["states.f64"])
            if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
                raise ValueError("Invalid native state file names")
            streams = {name: stack.enter_context((directory / name).open("wb")) for name in names}
            while not finished.is_set() or count < counters[0]:
                try:
                    kind, row, state = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                count += 1
                if kind == "command":
                    commands.write(json.dumps(row) + "\n")
                else:
                    row["index"] = states
                    timeline.write(json.dumps(row) + "\n")
                    payloads = state if isinstance(state, dict) else {"states.f64": state}
                    if set(payloads) != set(streams):
                        raise ValueError("Native state payload files differ from the manifest")
                    for name, payload in payloads.items():
                        if name != "states.f64":
                            streams[name].write(len(payload).to_bytes(8, "little"))
                        streams[name].write(payload)
                    states += 1
            native = directory.parent / "native-episode.json"
            if native.exists():
                shutil.copy2(native, directory / native.name)
                record = json.loads(native.read_text())
                for event in record["commands"]:
                    commands.write(json.dumps(event) + "\n")
        terminal = directory.parent / "terminal.json"
        if terminal.exists():
            shutil.copy2(terminal, directory / terminal.name)
        manifest.update(
            complete=counters[1] == 0 and counters[5] == 0 and states > 0,
            frames=states,
            states=(states if manifest["playback_kind"] != "native-command-replay"
                    else 0),
            dropped_events=counters[1],
            capture_errors=counters[5],
            final_frame=counters[3],
            native_success=bool(counters[4]),
            sealed=bool(counters[2]),
        )
        manifest["complete"] = manifest["complete"] and manifest["sealed"]
        if manifest["playback_kind"] != "recorded-native-integration-states":
            manifest["complete"] = manifest["complete"] and native.exists()
        manifest["files"] = {
            p.name: file_digest(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name != "manifest.json"
        }
    except Exception as exc:
        manifest.update(complete=False, error=f"{type(exc).__name__}: {exc}")
    write_json(directory / "manifest.json", manifest)


class Recording:
    """Capture never waits for capacity, disk, encoders, or writer completion.

    The bounded queue drops events on overload. Missing events invalidate the
    recording, not the world. finish() runs only after control has stopped.
    """

    def __init__(self, directory, identity, *, source=None, buffer_frames=256, stride=1):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.source = source
        self.stride = stride
        self.started = time.monotonic()
        self.last_frame = None
        self.closed = False
        self.manifest = {
            "schema": SCHEMA,
            **identity,
            "complete": False,
            "sealed": False,
            "stride": stride,
            "buffer_frames": buffer_frames,
            "media_policy": "deferred",
            "playback_kind": (
                "recorded-native-integration-states" if source else "native-command-replay"
            ),
            "intermediate_states_recorded": source is not None,
        }
        if source is not None:
            self.manifest.update(source.prepare(self.directory))
        write_json(self.directory / "manifest.json", self.manifest)
        context = multiprocessing.get_context("spawn")
        self.events = context.Queue(maxsize=buffer_frames)
        # Published, dropped, sealed, final frame, success, capture errors.
        self.counters = context.Array("q", 6, lock=False)
        self.finished = context.Event()
        self.writer = context.Process(
            target=_write,
            args=(self.directory, self.manifest, self.events, self.counters, self.finished),
            name="roboenv-recording-writer",
        )
        self.writer.start()
        self.events.cancel_join_thread()

    def _publish(self, kind, row, state=b""):
        if self.writer.exitcode is not None:
            self.counters[1] += 1
            return
        try:
            self.events.put_nowait((kind, row, state))
        except (queue.Full, OSError, ValueError):
            self.counters[1] += 1
        else:
            self.counters[0] += 1

    def capture(self, frame, simulation_seconds, *, force=False):
        if self.closed or frame == self.last_frame or not force and frame % self.stride:
            return
        try:
            state = self.source.capture() if self.source is not None else b""
        except Exception:
            self.counters[5] += 1
            return
        self._publish("frame", {
            "frame": frame,
            "simulation_seconds": simulation_seconds,
            "wall_seconds": time.monotonic() - self.started,
        }, state)
        self.last_frame = frame

    def command(self, operation, value):
        if not self.closed:
            self._publish("command", {
                "frame": self.source.frame,
                "wall_seconds": time.monotonic() - self.started,
                "op": operation,
                "commands" if operation == "write" else "device": value,
            })

    def seal(self, frame, simulation_seconds, success):
        if self.closed:
            return
        self.capture(frame, simulation_seconds, force=True)
        self.counters[3] = frame
        self.counters[4] = int(success)
        self.counters[2] = 1
        self.closed = True
        # Drain in the writer now, before simulator shutdown can hang.
        self.finished.set()

    def invalidate(self):
        self.counters[5] += 1
        self.closed = True

    def finish(self, timeout=FINALIZE_TIMEOUT_S):
        """Drain after the owner has stopped; retain incomplete evidence on failure."""
        self.finished.set()
        self.writer.join(timeout)
        if self.writer.is_alive():
            self.writer.terminate()
            self.writer.join()
        if self.writer.exitcode != 0:
            self.manifest.update(
                complete=False, error="recording writer did not finish",
                dropped_events=self.counters[1], capture_errors=self.counters[5],
                files={p.name: file_digest(p) for p in self.directory.iterdir()
                       if p.is_file() and p.name != "manifest.json"},
            )
            write_json(self.directory / "manifest.json", self.manifest)
        self.events.close()


class MujocoSource:
    """Copy native integration state on the owner thread; never render or step."""

    def __init__(self, world):
        import mujoco
        import numpy as np

        self.world = world
        self.model = world.sim.model._model
        self.data = world.sim.data._data
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.spec))

    @property
    def frame(self):
        return self.world._step_count

    def prepare(self, directory):
        import mujoco
        import numpy as np

        mujoco.mj_saveModel(self.model, str(directory / "model.mjb"), None)
        (directory / "model.xml").write_text(self.world.sim.model.get_xml())
        context = getattr(self.world.sim, "_render_context_offscreen", None)
        visual = context.vopt if context is not None else mujoco.MjvOption()
        fields = ("geomgroup", "sitegroup", "jointgroup", "tendongroup", "actuatorgroup",
                  "flexgroup", "skingroup", "flags", "label", "frame", "bvh_depth", "flex_layer")
        return {
            "mujoco_version": mujoco.__version__,
            "cameras": list(dict.fromkeys((
                self.world._camera_names[0], self.world._camera_names[-1],
            ))),
            "camera_size": list(self.world._camera_size),
            "control_hz": float(self.world.env.control_freq),
            "state_spec": int(self.spec),
            "state_width": self.state.size,
            "visual_options": {
                name: np.asarray(getattr(visual, name)).tolist()
                for name in fields if hasattr(visual, name)
            },
        }

    def capture(self):
        import mujoco

        mujoco.mj_getState(self.model, self.data, self.state, self.spec)
        return self.state.tobytes()
