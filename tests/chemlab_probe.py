"""Bounded native/HTTP and optional no-model Kinex instrument acceptance."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from helpers import load_tool
from roboenv.client import Client, RemoteError
from roboenv.environment import launch

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    definition = yaml.safe_load(
        (ROOT / "catalog/tasks/isaaclab/chemlab-hood-light-sash/task.yaml").read_text()
    )
    definition["seed"] = args.seed
    definition["execution_mode"] = args.execution_mode
    definition["backend_kwargs"].update(
        asset_directory=str(args.assets.resolve()), render=args.render
    )
    task = out / "task.yaml"
    task.write_text(yaml.safe_dump(definition, sort_keys=False))
    with launch(
        task,
        project=ROOT,
        interpreter=args.interpreter,
        timeout=300,
        run_dir=out / "runs",
        python_path=[ROOT / "src", ROOT / "runtime/src", *args.python_path],
        recording={} if args.render else None,
    ) as run:
        public = Client(endpoint=run.endpoint)
        world, instrument = public.device("world"), public.device("instrument")
        before = world.read()["data"]["time_s"]
        initial = instrument.read()["data"]
        after = world.read()["data"]["time_s"]
        if args.execution_mode == "stepped":
            assert after == before
        else:
            assert after >= before
        for action, arguments in [("plant_configure", {}), ("light_set", {"on": "invalid"})]:
            try:
                instrument.call(action, arguments)
            except RemoteError as error:
                assert error.code in {"unsupported_action", "invalid_args"}
            else:
                raise AssertionError("Invalid command was accepted")
        request = "chemlab-duplicate"
        response = instrument.call("light_set", {"on": False}, request_id=request)
        assert instrument.call("light_set", {"on": False}, request_id=request) == response
        if args.kinex:
            workspace = out / "workspace"
            load_tool("prepare_kinex_workspace").prepare(
                task,
                workspace,
                project=ROOT,
                endpoint=run.endpoint,
            )
            shutil.copytree(
                ROOT / "tests/fixtures/chemlab-probe", workspace / ".kinex/tools/chemlab-probe"
            )
            with (out / "kinex.log").open("x") as log:
                subprocess.run(
                    [
                        args.node,
                        str(ROOT / "tests/chemlab_kinex_probe.mjs"),
                        str(args.kinex),
                        str(workspace),
                        str(out / "kinex-result.json"),
                    ],
                    check=True,
                    timeout=240,
                    stdout=log,
                    stderr=log,
                    env={
                        **os.environ,
                        "KINEX_CONFIG_DIR": str(out / "kinex-config"),
                        "KINEX_DATA_DIR": str(out / "kinex-data"),
                    },
                )
            assert json.loads((out / "kinex-result.json").read_text())["passed"]
        else:
            for action, arguments in [
                ("light_set", {"on": True}),
                ("sash_set", {"height_m": 0.457}),
            ]:
                assert instrument.call(action, arguments)["data"]["status"] == "succeeded"
            try:
                world.wait(1.2)
            except RemoteError as error:
                assert error.code == "episode_finished"
        feedback = instrument.read()["data"]
        assert feedback["observations"]["light.on"]["value"] is True
        assert abs(feedback["observations"]["sash.height_m"]["value"] - 0.457) < 0.001
        finalized = load_tool("finalize_episode").finalize(run.path, out / "owner")
        assert finalized["verdict"]["completed"]
        run_path = run.path
    # Replay in a different process using exactly the selected runtime dependencies.
    subprocess.run(
        [
            str(args.interpreter),
            str(ROOT / "tools/replay_chemlab.py"),
            str(out / "owner/owner-record.json"),
            "--directory",
            str(args.assets),
            "--out",
            str(out / "replay.json"),
        ],
        check=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                map(
                    str,
                    [
                        ROOT / "src",
                        ROOT / "runtime/src",
                        *args.python_path,
                    ],
                )
            ),
        },
    )
    replay = json.loads((out / "replay.json").read_text())
    assert replay["passed"]
    videos = []
    if args.render:
        import cv2

        for path in run_path.parent.rglob("*.mp4"):
            capture = cv2.VideoCapture(str(path))
            count, nonempty = 0, 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    count += 1
                    nonempty += int(frame.std() > 1)
            finally:
                capture.release()
            assert count > 1 and nonempty == count
            videos.append({"path": str(path.relative_to(out)), "frames": count})
        assert videos
    receipt = {
        "passed": True,
        "seed": args.seed,
        "execution_mode": args.execution_mode,
        "rendered": args.render,
        "kinex": bool(args.kinex),
        "native_success": True,
        "replay": replay,
        "videos": videos,
        "initial": initial,
        "final": feedback,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--interpreter", type=Path, default=Path(sys.executable))
    parser.add_argument("--python-path", type=Path, action="append", default=[])
    parser.add_argument("--kinex", type=Path)
    parser.add_argument("--node", default="node")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-mode", choices=["stepped", "realtime"], default="stepped")
    parser.add_argument("--render", action="store_true")
    check(parser.parse_args())
