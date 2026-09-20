"""Real Kinex discovery, controller, recording and owner replay; no model calls."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import roboenv

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(out, kinex, node):
    out.mkdir(parents=True, exist_ok=False)
    workspace = out / "workspace"
    tool("prepare_kinex_workspace").prepare("libero/bowl-onto-plate", workspace, project=ROOT)
    env = {**os.environ, "KINEX_DATA_DIR": str(out / "kinex-data"),
           "KINEX_CONFIG_DIR": str(out / "kinex-config")}
    with (out / "python.log").open("w") as log:
        subprocess.run([
            node, "--input-type=module", "-e",
            "import { pathToFileURL } from 'node:url'; "
            "const { createKinexRuntime } = await import(pathToFileURL(process.argv[1])); "
            "const runtime = createKinexRuntime({ directory: process.argv[2] }); "
            "try { await runtime.python.prepare(); } finally { await runtime.dispose(); }",
            str(kinex / "packages/core/dist/index.js"), str(workspace),
        ], cwd=workspace, env=env, check=True, stdout=log, stderr=log, timeout=180)
    with roboenv.launch("libero/bowl-onto-plate", project=ROOT, realtime=False,
                        interpreter=sys.executable) as run:
        identity = {k: run.record[k] for k in ("session", "generation")}
        for driver in (workspace / ".kinex/devices").glob("*/driver.py"):
            driver.with_name("connection.json").write_text(json.dumps({
                **identity, "endpoint": run.endpoint,
            }))
        with (out / "kinex.log").open("w") as log:
            subprocess.run([
                node, str(ROOT / "tests/libero_kinex_smoke.ts"), str(kinex),
                str(workspace), str(out / "smoke.json"),
            ], cwd=workspace, env=env, stdout=log, stderr=log, check=True, timeout=150)
        result = tool("finalize_episode").finalize(run.path, out / "owner")
        (out / "episode.json").write_text(json.dumps({**identity, "verdict": result["verdict"]}))
    # Decode every recorded camera frame, rather than accepting an encoder exit code.
    import cv2

    videos = []
    for manifest in (workspace / ".kinex/recordings").glob("*/recording.json"):
        value = json.loads(manifest.read_text())
        assert value["status"] == "stopped"
        if value["kind"] != "stream":
            continue
        for name in ("camera", "camera_wrist"):
            stats = value["devices"][name]
            assert stats["stream"]["status"] == "stopped", stats
            file = manifest.parent / stats["stream"]["video"]
            cap = cv2.VideoCapture(str(file))
            count = black = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                    black += int(frame.max() == 0)
            finally:
                cap.release()
            assert count >= 3 and not black, (file, count, black)
            videos.append({"device": name, "file": str(file), "decoded_frames": count,
                           "black_frames": black, "effective_hz": stats.get("effectiveHz")})
    assert len(videos) == 2
    subprocess.run([
        os.environ.get("ROBOENV_LIBERO_TEST_PYTHON", str(ROOT / ".venv-sim/libero/bin/python")),
        str(ROOT / "tools/replay_libero_devices.py"), str(out / "owner/owner-record.json"),
        "--out", str(out / "replay.json"),
    ], check=True, timeout=120)
    report = {"passed": True, "model_calls": 0, "task_success_claimed": False,
              "videos": videos, "episode": identity}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinex", type=Path, default=ROOT.parent / "kinex")
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    check(args.out.resolve(), args.kinex.resolve(), args.node)
