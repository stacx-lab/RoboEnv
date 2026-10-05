"""Fresh SIMPLE worker, portable Kinex workspace, recording, and command replay."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import roboenv
import yaml
from roboenv.tasks.resolve import resolve_task

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    definition = yaml.safe_load(resolve_task(args.task, project=ROOT).path.read_text())
    definition["execution_mode"] = args.execution_mode
    definition["backend_kwargs"]["sim_mode"] = args.sim_mode
    definition["backend_kwargs"]["arm_gravity_compensation"] = args.arm_gravity_compensation
    task = out / "task.yaml"
    task.write_text(yaml.safe_dump(definition, sort_keys=False))
    workspace = out / "workspace"
    tool("prepare_kinex_workspace").prepare(task, workspace, project=ROOT)
    shutil.copytree(
        ROOT / "tests/fixtures/device-composition",
        workspace / ".kinex/tool/device-composition",
    )
    hand_tool = workspace / ".kinex/tool/hand-geometry-probe"
    hand_tool.mkdir()
    shutil.copy2(ROOT / "tests/simple_hand_tool.py", hand_tool / "tool.py")
    (hand_tool / "tool.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "hand_geometry_probe",
                "kind": "perception",
                "description": "Validate hand geometry without commands.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                "devices": ["joint", "dexterous_hand"],
            }
        )
    )
    clean = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env = {
        **clean,
        "PYTHONNOUSERSITE": "1",
        "KINEX_DATA_DIR": str(out / "kinex-data"),
        "KINEX_CONFIG_DIR": str(out / "kinex-config"),
    }
    subprocess.run(
        [args.node, str(ROOT / "tests/native_rigs_kinex.mjs"), str(args.kinex), str(workspace)],
        check=True,
        env=env,
        timeout=60,
    )
    with roboenv.launch(
        task, project=ROOT, interpreter=sys.executable, timeout=args.startup_timeout, gpu=args.gpu,
        run_dir=out / "runs",
    ) as run:
        for driver in (workspace / ".kinex/devices").glob("*/driver.py"):
            driver.with_name("connection.json").write_text(
                json.dumps(
                    {
                        "endpoint": run.endpoint,
                        **{key: run.record[key] for key in ("session", "generation")},
                    }
                )
            )
        with (out / "kinex.log").open("x") as log:
            subprocess.run(
                [
                    args.node,
                    str(ROOT / "tests/simple_kinex_smoke.mjs"),
                    str(args.kinex),
                    str(workspace),
                    str(out / "smoke.json"),
                    args.execution_mode,
                ],
                cwd=workspace,
                env=env,
                check=True,
                stdout=log,
                stderr=log,
                timeout=180,
            )
        verdict = tool("finalize_episode").finalize(run.path, out / "owner")["verdict"]
    import cv2

    videos = []
    for manifest in (workspace / ".kinex/recordings").glob("*/recording.json"):
        value = json.loads(manifest.read_text())
        assert value["status"] == "stopped"
        if value["kind"] != "stream":
            continue
        for name in ("camera_head_left", "camera_head_right"):
            stats = value["devices"][name]
            assert stats["stream"]["status"] == "stopped", stats
            video = manifest.parent / stats["stream"]["video"]
            cap, count, black = cv2.VideoCapture(str(video)), 0, 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    count += 1
                    black += int(frame.max() == 0)
            finally:
                cap.release()
            assert count >= 3 and not black, (video, count, black)
            videos.append({"device": name, "frames": count, "black": black})
    assert len(videos) == 2
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/replay_simple_devices.py"),
            str(out / "owner/owner-record.json"),
            "--out",
            str(out / "replay.json"),
            *([] if args.gpu is None else ["--gpu", str(args.gpu)]),
        ],
        check=True,
        timeout=args.startup_timeout,
        env=env,
    )
    assert json.loads((out / "replay.json").read_text())["passed"]
    (out / "report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "model_calls": 0,
                "task_attempted": False,
                "verdict": verdict,
                "videos": videos,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="simple/xmove-pick")
    parser.add_argument("--gpu", type=int, help="Host GPU index for Isaac rendering")
    parser.add_argument("--execution-mode", choices=("stepped", "realtime"), default="stepped")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--kinex", type=Path,
        default=Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")),
    )
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco")
    parser.add_argument(
        "--arm-gravity-compensation", action=argparse.BooleanOptionalAction, default=True
    )
    check(parser.parse_args())
