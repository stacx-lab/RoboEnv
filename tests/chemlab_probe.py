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
from test_chemlab import SOLUTIONS

ROOT = Path(__file__).resolve().parents[1]


def check(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    definition = yaml.safe_load(
        (ROOT / f"catalog/tasks/isaaclab/chemlab-{args.task}/task.yaml").read_text()
    )
    definition["seed"] = args.seed
    definition["execution_mode"] = args.execution_mode
    definition["backend_kwargs"].update(
        asset_directory=str(args.assets.resolve()), render=args.render
    )
    task = out / "task.yaml"
    task.write_text(yaml.safe_dump(definition, sort_keys=False))
    screen = "terminal" if args.task == "balance-power" else "lcd"
    views = ["third_person", f"instrument_{screen}"]
    with launch(
        task,
        project=ROOT,
        interpreter=args.interpreter,
        timeout=300,
        run_dir=out / "runs",
        python_path=[ROOT / "src", ROOT / "runtime/src", *args.python_path],
        recording={"views": [{"name": name, "source": name} for name in views]}
        if args.render
        else None,
    ) as run:
        public = Client(endpoint=run.endpoint)
        world, instrument = public.device("world"), public.device("instrument")
        before = world.read()["data"]["time_s"]
        initial = instrument.read()["data"]
        after = world.read()["data"]["time_s"]
        assert after >= before
        commands = [(cap.replace(".", "_"), value) for cap, value in SOLUTIONS[args.task]]
        for action, arguments in [("plant_configure", {}), (commands[0][0], {})]:
            try:
                instrument.call(action, arguments)
            except RemoteError as error:
                assert error.code in {"unsupported_action", "invalid_args"}
            else:
                raise AssertionError("Invalid command was accepted")
        # Capture the actual public panel before any test-controller commands.
        asset = definition["backend_kwargs"]["task_spec"]["instruments"]["instrument"]["asset"]
        from roboenv.provisioning.chemlab_assets import catalog

        row = catalog(args.assets)[asset]
        contract = json.loads((args.assets / row["contract"]).read_text())
        displays = contract["capabilities"]["panel.capture"]["input_schema"]["properties"][
            "display_id"
        ]["enum"]

        def panels(label):
            import base64

            for display in displays:
                # Panel capture is read-only, but stop-on-success closes mutating
                # command transport. Final textures are retained with owner evidence.
                result = instrument.call("panel_capture", {"display_id": display})["data"]
                data = result["result"]["image"]
                (out / f"{label}-{display}.png").write_bytes(base64.b64decode(data["base64"]))

        panels("initial")
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
            # Cover the 180-second world wait plus transport overhead using
            # the existing Kinex per-call setting; cold starts also fit this cap.
            for profile in (workspace / ".kinex/devices").glob("*/profile.yaml"):
                value = yaml.safe_load(profile.read_text())
                value["access"]["driver"]["timeout_s"] = 240
                profile.write_text(yaml.safe_dump(value, sort_keys=False))
            inputs = out / "probe-input.json"
            inputs.write_text(json.dumps({"commands": commands, "wait_s": 180.0}))
            with (out / "kinex.log").open("x") as log:
                subprocess.run(
                    [
                        args.node,
                        str(ROOT / "tests/chemlab_kinex_probe.mjs"),
                        str(args.kinex),
                        str(workspace),
                        str(out / "kinex-result.json"),
                        str(inputs),
                    ],
                    check=True,
                    timeout=420,
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
            for action, arguments in commands:
                assert instrument.call(action, arguments)["data"]["status"] == "succeeded"
            try:
                world.wait(180.0)
            except RemoteError as error:
                assert error.code == "episode_finished"
        feedback = instrument.read()["data"]
        finalized = load_tool("finalize_episode").finalize(run.path, out / "owner")
        assert finalized["verdict"]["completed"]
        if args.render:
            import hashlib

            owner = json.loads((out / "owner/owner-record.json").read_text())
            candidates = list(run.path.parent.rglob("chemlab-panel-*/*.png"))
            for display, image in owner["scene_evidence"]["instrument"]["panels"].items():
                source = next(
                    path
                    for path in candidates
                    if hashlib.sha256(path.read_bytes()).hexdigest() == image["sha256"]
                )
                shutil.copyfile(source, out / f"final-{display}.png")
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
            *(["--render"] if args.render else []),
        ],
        check=True,
        timeout=300,
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
        record = json.loads((out / "owner/owner-record.json").read_text())
        assert replay["scene_matched"] is True
        assert len(videos) >= 2, "An overview and a native display close-up are required"
        assert record["scene_evidence"]["instrument"]["panels"]
    receipt = {
        "passed": True,
        "task": definition["id"],
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
    parser.add_argument("--execution-mode", choices=["realtime"], default="realtime")
    parser.add_argument("--task", choices=sorted(SOLUTIONS), default="hood-light-sash")
    parser.add_argument("--render", action="store_true")
    check(parser.parse_args())
