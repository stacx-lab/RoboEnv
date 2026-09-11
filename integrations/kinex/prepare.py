"""Prepare only public device facts and reusable content inside a Harbor agent container."""

import json
import os
import shutil
import sys
from pathlib import Path

from artifacts import devices, restore, unchanged


def prepare(config):
    workspace = Path(config["workspace"])
    public = Path(os.environ["ROBOENV_ROBOT"])
    root = workspace / ".kinex"
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(public / "devices", root / "devices")
    definition = public / "workspace.yaml"
    if not definition.is_file():
        definition = Path("/env/robot/workspace.yaml")
    shutil.copy2(definition, root / "workspace.yaml")
    if config.get("artifact"):
        restore(config["artifact"], workspace, identity=config["identity"],
                development=config["mode"] == "develop")
    if config.get("tool_directory"):
        shutil.copytree(config["tool_directory"], root / "tool" / config["tool"])
    if config["tool"] == "integration_probe" and not (root / "tool/integration_probe").exists():
        shutil.copytree(Path(__file__).with_name("probe"), root / "tool/integration_probe")
    (workspace / "instruction.md").write_text(config["instruction"] + "\n")
    return {"devices": devices(workspace)}


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1] == "prepare":
        report = prepare(config)
        Path(config["before"]).write_text(json.dumps(report))
    else:
        before = json.loads(Path(config["before"]).read_text())
        report = {"devices_unchanged": devices(config["workspace"]) == before["devices"],
                  "frozen_unchanged": None}
        if config.get("artifact") and config["mode"] != "develop":
            report["frozen_unchanged"] = unchanged(config["workspace"], config["artifact"])
        report["passed"] = all(value is not False for value in report.values())
        Path(config["integrity"]).write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise SystemExit(1)
