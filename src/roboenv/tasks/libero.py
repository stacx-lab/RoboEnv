"""Read official LIBERO task and rig metadata without importing a simulator."""

import ast
import re
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
ROBOTS = {
    "libero_tabletop_manipulation": "libero/libero-panda-tabletop",
    "libero_floor_manipulation": "libero/libero-panda-floor",
    "libero_kitchen_tabletop_manipulation": "libero/libero-panda-kitchen",
    "libero_living_room_tabletop_manipulation": "libero/libero-panda-living-room",
    "libero_study_tabletop_manipulation": "libero/libero-panda-study",
}

CONTROL_GUIDANCE = """
LIBERO gripper writes select a persistent direction: +1 closes, -1 opens, and 0
retains the accumulated actuator target. Gripper stop instead resets that target
to the measured finger positions; it can reduce gripping force and is not a
grasp-preserving cleanup operation. Arm stop affects only the arm. A write being
accepted does not establish arrival or a grasp: check measured motion and finger
aperture, then use a small lift and fresh images to check that the object follows.

Before contact, check the measured tool pose and finger-pad geometry, allowing
clearance for the whole gripper and arm around articulated fixtures. If motion
stalls, re-observe and revise the approach before closing the fingers. After an
object or fixture moves, refresh its observed location. For wrist-camera geometry,
hold the arm still while reading the image and pose; do not pair an old image
with a later arm pose.
"""

COMPLETION_CHECK = """
Before ending the task or claiming success, define concrete, task-specific checks
for every requested outcome and verify all of them using fresh public sensor
observations, including the object's final relation to its target and any required
drawer, door, or switch state. Use multiple views or measured feedback when one
view is ambiguous. Tool acceptance, a completed todo, and absence of errors do not
prove task completion. If a check fails or remains uncertain and time is available,
diagnose it and attempt a safe correction; do not end solely because one grasp or
motion failed. If further progress is blocked or the run limit is reached, report
the task as incomplete and state which checks failed. Do not access private
verifier state or change the benchmark success rules.
"""


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
            if not problem or problem[1].lower() not in ROBOTS or not language:
                raise ValueError(f"Unsupported LIBERO task metadata: {bddl.name}")
            yield {
                "suite_name": suite,
                "task_id": index,
                "name": name,
                "robot": ROBOTS[problem[1].lower()],
                "problem": problem[1].lower(),
                "bddl": bddl,
                "states": states,
                "instruction": " ".join(language[1].split()).strip('"'),
            }
