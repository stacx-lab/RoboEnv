"""Explicit RTX acceptance run: public devices, copied drivers, owner verdict."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import roboenv
import yaml
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
        assert initial["pose"] is None and len(initial["frame_poses"]) == 4
        target = [
            q + (0.1 if i % 3 == 1 else -0.2 if i % 3 == 2 else 0)
            for i, q in enumerate(
                body.profile["intrinsics"]["home"][j["name"]]
                for j in body.profile["intrinsics"]["kinematics"]["model"]["joints"]
            )
        ]
        for _ in range(150):
            assert body.write(target)["accepted"]
        state = body.read()
        assert max(abs(a - b) for a, b in zip(state["positions"], target, strict=True)) < 0.05
        stamp = state["sample_time_s"]
        assert abs(stamp - 3.0) < 1e-8
        kinex_root = root.parent / "kinex"
        workspace = out / "kinex"
        config = workspace / ".kinex"
        copied_body = config / "devices/body"
        robot = root / "catalog/robots/isaaclab/isaaclab-go2"
        shutil.copytree(robot / "devices/body", copied_body)
        shutil.copyfile(robot / "workspace.yaml", config / "workspace.yaml")
        profile_file = copied_body / "profile.yaml"
        profile = yaml.safe_load(profile_file.read_text())
        profile["access"]["driver"]["interpreter"] = sys.executable
        profile_file.write_text(yaml.safe_dump(profile, sort_keys=False))
        subprocess.run(
            ["bun", str(root / "tests/isaaclab_kinex_smoke.ts"), str(kinex_root),
             str(workspace), str(out / "kinematics-servo.json")],
            env={**os.environ, "ROBOENV_ENDPOINT": run.endpoint}, check=True, timeout=180,
        )
        stamp = body.read()["sample_time_s"]
        env.device("imu").read()
        env.device("contact_fl_foot").read()
        camera = env.device("camera_front").read()
        assert Path(camera["file"]).read_bytes().startswith(b"\x89PNG")
        assert Path(camera["depth_file"]).read_bytes().startswith(b"P5\n")
        body.stop()
        assert body.read()["sample_time_s"] == stamp
        copied = out / "camera_front"
        shutil.copytree(
            root / "catalog/robots/isaaclab/isaaclab-go2/devices/camera_front", copied,
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             "import runpy,sys; sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; "
             "runpy.run_path(sys.argv[0],run_name='__main__')",
             str(copied), str(copied / "driver.py"), "read", "{}"],
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
