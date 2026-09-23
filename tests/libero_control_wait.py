"""Fresh LIBERO and Kinex bounded-write smoke in both clocks; no task attempt."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import roboenv
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check(out, kinex, node):
    out.mkdir(parents=True, exist_ok=False)
    for mode in ("synchronous",):
        workspace = out / mode
        config = workspace / ".kinex"
        config.mkdir(parents=True)
        with roboenv.launch("libero/drawer-and-bowl", project=ROOT, port=0) as run:
            run_directory = run.path.parent
            shutil.copytree(run_directory / "devices", config / "devices")
            shutil.copy2(run_directory / "server/robot/workspace.yaml", config / "workspace.yaml")
            # Device activation goes through the Kinex device tool in the smoke.
            for profile in (config / "devices").glob("*/profile.yaml"):
                value = yaml.safe_load(profile.read_text())
                value["loaded"] = False
                profile.write_text(yaml.safe_dump(value, sort_keys=False))
            env = {
                **os.environ,
                "KINEX_DATA_DIR": str(out / "kinex-data"),
                "KINEX_CONFIG_DIR": str(out / "kinex-config"),
            }
            subprocess.run(
                [
                    node,
                    str(ROOT / "tests/kinex_control_wait.mjs"),
                    str(kinex),
                    str(workspace),
                    mode,
                    str(out / f"{mode}.json"),
                ],
                cwd=workspace,
                env=env,
                check=True,
                timeout=180,
            )
            (out / f"{mode}-run.json").write_text(
                json.dumps(
                    {
                        "session": run.record["session"],
                        "record": str(run.path),
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinex", type=Path, required=True)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    check(args.out.resolve(), args.kinex.resolve(), args.node)
