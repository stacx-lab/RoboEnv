"""Explicit RTX acceptance run: public devices, copied drivers, owner verdict."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import roboenv
from roboenv.client import Client


class Owner(Client):
    allowed_ops = Client.allowed_ops | {"submit"}


out = Path(sys.argv[1])
out.mkdir(parents=True)
root = Path(__file__).resolve().parents[1]
with roboenv.launch(task="isaaclab/go2-posture", interpreter=sys.executable, project=root) as run:
    with roboenv.connect(run.endpoint) as env:
        body = env.device("body")
        initial = body.read()
        assert "pose" not in initial and len(initial["frame_poses"]) == 4
        target = [
            q + (0.1 if i % 3 == 1 else -0.2 if i % 3 == 2 else 0)
            for i, q in enumerate(body.profile["intrinsics"]["home_joints"])
        ]
        for _ in range(150):
            assert body.write(target)["accepted"]
        state = body.read()
        assert max(abs(a - b) for a, b in zip(state["positions"], target, strict=True)) < 0.05
        stamp = state["sample_time_s"]
        assert abs(stamp - 3.0) < 1e-8
        env.device("imu").read()
        env.device("contact_fl_foot").read()
        camera = env.device("camera_front").read()
        assert Path(camera["file"]).read_bytes().startswith(b"\x89PNG")
        assert Path(camera["depth_file"]).read_bytes().startswith(b"P5\n")
        body.stop()
        assert body.read()["sample_time_s"] == stamp
        copied = out / "camera_front"
        copied.mkdir()
        shutil.copyfile(
            root / "catalog/robots/isaaclab-go2/devices/camera_front/driver.py",
            copied / "driver.py",
        )
        result = subprocess.run(
            [sys.executable, "-I", str(copied / "driver.py"), "read", "{}"],
            env={**os.environ, "ROBOENV_ENDPOINT": run.endpoint},
            capture_output=True,
            text=True,
            check=True,
        )
        image = json.loads(result.stdout)
        assert image["ok"] and Path(image["file"]).is_file()
    # Attached public clients do not own finalization; use the protected owner socket.
    owner = Owner(
        run.record["socket"], session=run.record["session"], generation=run.record["generation"]
    )
    report = owner.request("submit")
    assert report["completed"] and report["reward"] == 1.0, report
    owner.close()
print("Go2 native device and posture check passed")
