"""Read official LIBERO task and rig metadata without importing a simulator."""

import ast
import re
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
RIGS = {
    "libero_tabletop_manipulation": "libero-panda",
    "libero_floor_manipulation": "libero-panda-floor",
    "libero_kitchen_tabletop_manipulation": "libero-panda-kitchen",
    "libero_living_room_tabletop_manipulation": "libero-panda-living-room",
    "libero_study_tabletop_manipulation": "libero-panda-study",
}


def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == name for n in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing native registry: {name}")


def entries(base):
    base = Path(base)
    mapping = literal(base / "benchmark/libero_suite_task_map.py", "libero_task_map")
    orders = literal(base / "benchmark/__init__.py", "task_orders")
    if orders[0] != list(range(10)):
        raise ValueError("Official LIBERO task order changed; review discovery")
    for suite in SUITES:
        for index, name in enumerate(mapping[suite]):
            bddl = base / "bddl_files" / suite / (name + ".bddl")
            states = base / "init_files" / suite / (name + ".pruned_init")
            if not bddl.is_file() or not states.is_file():
                raise ValueError(f"Official LIBERO task assets missing: {suite}/{name}")
            text = bddl.read_text()
            problem = re.search(r"\(problem\s+([^\s)]+)", text)
            language = re.search(r"\(:language\s+(.*?)\)", text, re.S)
            if not problem or problem[1].lower() not in RIGS or not language:
                raise ValueError(f"Unsupported LIBERO task metadata: {bddl.name}")
            yield {
                "suite_name": suite,
                "task_id": index,
                "name": name,
                "robot": RIGS[problem[1].lower()],
                "bddl": bddl,
                "states": states,
                "instruction": " ".join(language[1].split()).strip('"'),
            }
