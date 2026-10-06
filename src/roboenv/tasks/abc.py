"""ABC registry discovery without importing MuJoCo or training packages."""

import ast
import json
from importlib.resources import files

from roboenv.tasks.model import Task


def entries(root):
    source = root / "abc_sim/task_specs.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SimTaskSpec"
        ):
            yield {item.arg: ast.literal_eval(item.value) for item in node.keywords}


def instruction(entry):
    name = entry["name"]
    if name.startswith("put_markers_in_"):
        level = name.removeprefix("put_markers_in_").removesuffix("_drawer")
        return f"Put all five markers in the {level} drawer, release them, and close the drawer."
    if name == "build_wood_block_tower":
        return (
            "Build a self-supporting tower of at least five blocks and release it for one second."
        )
    return entry["prompt"]


def discover_tasks(root):
    source = root / "third_party/abc"
    rigs = json.loads(files("roboenv").joinpath("assets/abc-rigs.json").read_text())
    for entry in entries(source):
        yield (
            Task(
                id="abc/" + entry["name"].replace("_", "-"),
                robot="abc/" + rigs[entry["env_task"]],
                instruction=instruction(entry),
                budget_mode="practice",
                backend_kwargs={"task_name": entry["name"]},
            ),
            (source / "abc_sim/task_specs.py"),
        )
