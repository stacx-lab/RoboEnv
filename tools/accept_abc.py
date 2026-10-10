"""Small native/Kinex ABC acceptance using the existing owner and replay tools."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import roboenv
import yaml
from finalize_episode import finalize
from prepare_kinex_workspace import prepare
from roboenv.provisioning.interpreter import resolve_interpreter

ROOT = Path(__file__).resolve().parents[1]


def check_rgbd(client, *, frozen):
    before = client.device("world").read()
    result = {}
    for name in ("camera_top", "camera_left", "camera_right"):
        frame = client.device(name).read(output="array")
        assert frame["color"].shape == (480, 640, 3)
        assert frame["depth"].shape == (480, 640)
        assert frame["color"].dtype == np.uint8 and frame["depth"].dtype == np.uint16
        assert np.count_nonzero(frame["depth"]) > 0
        assert np.isfinite(frame["timestamp"])
        if frozen:
            again = client.device(name).read(output="array")
            assert again["timestamp"] == frame["timestamp"]
            np.testing.assert_array_equal(again["depth"], frame["depth"])
        result[name] = {"timestamp": frame["timestamp"],
                        "valid_pixels": int(np.count_nonzero(frame["depth"]))}
    if frozen:
        assert client.device("world").read()["time_s"] == before["time_s"]
    return result


def check_depth_recordings(out):
    files = sorted(out.rglob("depth.mkv"))
    assert len(files) == 3, f"Expected three depth recordings, found {files}"
    for file in files:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=pix_fmt,width,height", "-of", "json", str(file)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream == {"width": 640, "height": 480, "pix_fmt": "gray16le"}, stream
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(file), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1"],
            capture_output=True, check=True, timeout=30,
        )
        pixels = np.frombuffer(decoded.stdout, dtype="<u2")
        assert pixels.size == 640 * 480 and np.count_nonzero(pixels) > 0
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(file), "-f", "null", "-"],
                       capture_output=True, check=True, timeout=60)
    return {"streams": len(files), "decoded": True, "encoding": "gray16le"}


def run(out, kinex, node, renderer="mujoco"):
    out.mkdir(parents=True, exist_ok=False)
    python = resolve_interpreter(ROOT, "abc", None)
    environment = {**os.environ, "MUJOCO_GL": "egl"}
    os.environ.setdefault("MUJOCO_GL", "egl")
    report = {"passed": False, "model_calls": 0, "renderer": renderer, "checks": {}}

    def task(name):
        value = yaml.safe_load((ROOT / "catalog/tasks/abc" / name / "task.yaml").read_text())
        value["backend_kwargs"]["camera_backend"] = renderer
        target = out / (name + ".yaml")
        target.write_text(yaml.safe_dump(value, sort_keys=False))
        return target

    primary = task(
        "put-plastic-bottles-in-bin" if renderer == "blender" else "count-two-into-opaque-box"
    )
    secondary = task("multi-drawer-search" if renderer == "blender" else "ball-tray-balancing")
    recording = {"directory": str(out / "spectator")}
    if renderer == "blender":
        recording.update(
            fps=2,
            views=[
                {"name": "overview", "source": "third_person"},
                {"name": "wrist", "source": "wrist"},
                {"name": "fixed", "pose": [0.6, 0, 1.8, 1, 0, 0, 0], "width": 640, "height": 480},
            ],
        )

    def execute(label, command, timeout=300):
        with (out / (label + ".log")).open("x") as log:
            subprocess.run(
                [str(arg) for arg in command],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=log,
                check=True,
                timeout=timeout,
            )

    try:
        execute("native", [python, ROOT / "tests/abc_native.py", "--out", out / "native.json"])
        report["checks"]["native"] = json.loads((out / "native.json").read_text())
        workspace = out / "workspace"
        prepare(primary, workspace, project=ROOT)
        execute("profiles", [node, ROOT / "tests/native_rigs_kinex.mjs", kinex, workspace])
        with roboenv.launch(
            primary,
            project=ROOT,
            interpreter=python,
            execution_mode="stepped",
            recording=recording,
        ) as owner:
            with roboenv.connect(owner.endpoint) as client:
                report["checks"]["rgbd_stepped"] = check_rgbd(client, frozen=True)
            endpoint = owner.endpoint
            identity = {key: owner.record[key] for key in ("session", "generation")}
            for driver in (workspace / ".kinex/devices").glob("*/driver.py"):
                driver.with_name("connection.json").write_text(
                    json.dumps({**identity, "endpoint": endpoint})
                )
            environment.update(
                KINEX_DATA_DIR=str(out / "kinex-data"), KINEX_CONFIG_DIR=str(out / "kinex-config")
            )
            execute("task", [python, workspace / "task.py"])
            execute(
                "kinex",
                [node, ROOT / "tests/abc_kinex_smoke.mjs", kinex, workspace, out / "kinex.json"],
            )
            # Another patched scene and realtime semantics, without a full task matrix.
            with roboenv.launch(
                secondary,
                project=ROOT,
                interpreter=python,
                execution_mode="realtime",
            ) as peer:
                with roboenv.connect(peer.endpoint) as client:
                    before = client.device("world").read()
                    time.sleep(0.15)
                    after = client.device("world").read()
                    assert after["time_s"] > before["time_s"]
                    assert after["budget"]["max_s"] is None
                    assert peer.endpoint != endpoint
                    report["checks"]["rgbd_realtime"] = check_rgbd(client, frozen=False)
                report["checks"]["realtime_second_scene"] = True
            result = finalize(owner.path, out / "owner")
            report["task_success"] = result["verdict"]["completed"]
        execute(
            "replay",
            [
                python,
                ROOT / "tools/replay_native_devices.py",
                out / "owner/owner-record.json",
                "--out",
                out / "replay.json",
            ],
        )
        # Decode the short recordings once; an encoder exit alone is insufficient.
        execute(
            "media",
            [
                python,
                "-c",
                "import imageio.v3 as iio, pathlib, sys; "
                'files=list(pathlib.Path(sys.argv[1]).rglob("*.mp4")); assert files; '
                "counts=[sum(1 for frame in iio.imiter(p) if frame.max()>0) for p in files]; "
                "assert all(n>0 for n in counts); print(counts)",
                out,
            ],
        )
        report["checks"]["depth_recording"] = check_depth_recordings(out)
        report["checks"].update(kinex=True, recording=True, replay=True)
        report["passed"] = True
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinex", type=Path, default=ROOT.parent / "kinex")
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    parser.add_argument("--renderer", choices=("mujoco", "blender"), default="mujoco")
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve(), args.kinex.resolve(), args.node, args.renderer)))
