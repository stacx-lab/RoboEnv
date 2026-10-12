"""Read and discover environment tasks without importing simulator engines."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml
from roboenv.project import project_root
from roboenv.provisioning.sources import load_lock, status
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.instructions import robocasa_instruction, robodojo_instruction
from roboenv.tasks.model import Task

BACKENDS = ("abc", "libero", "robocasa", "robotwin", "robodojo", "behavior", "isaaclab", "simple")
ROBOTS = {
    "simple": "simple/simple-g1",
    "libero": "libero/libero-panda-tabletop",
    "robocasa": "robocasa/robocasa-panda-omron",
    "robotwin": "robotwin/robotwin-dual-piper",
    "robodojo": "robodojo/robodojo-arx-x5",
    "behavior": "behavior/behavior-r1pro",
}


def load_catalog(root):
    result = {}
    for path in sorted(Path(root).rglob("task.yaml")):
        task = Task.load(path)
        if task.id in result:
            raise ValueError(f"duplicate task id: {task.id}")
        result[task.id] = (task, path)
    if not result:
        raise ValueError("no task.yaml definitions found in the selected catalog")
    return result


def summary(project=None, tasks_root=None):
    root = project_root(project)
    tasks = load_catalog(tasks_root or root / "catalog/tasks")
    return {
        "tasks": len(tasks),
        "backends": dict(Counter(t.backend for t, _ in tasks.values())),
        "ids": sorted(tasks),
        "execution_modes": {key: tasks[key][0].execution_mode for key in sorted(tasks)},
    }


def task_name(value):
    """Normalize a native name without changing its simulator selector."""
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", value)
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _task(backend, name, kwargs, instruction, *, robot=None):
    return Task(
        id=f"{backend}/{task_name(name)}",
        backend_kwargs=dict(kwargs),
        robot=robot or ROBOTS[backend],
        budget_mode="practice"
        if backend in {"libero", "robotwin", "robodojo", "behavior", "simple"}
        else "official",
        instruction=instruction,
    )


def _libero(root, issues):
    from roboenv.tasks.libero import entries

    for entry in entries(root / "third_party/LIBERO/libero/libero"):
        short = entry["suite_name"].removeprefix("libero_")
        yield (
            _task(
                "libero",
                f"{short}-{entry['task_id']}",
                {
                    "suite_name": entry["suite_name"],
                    "task_id": entry["task_id"],
                    "init_state_index": 0,
                },
                entry["instruction"],
                robot=entry["robot"],
            ),
            entry["bddl"],
        )


def _robocasa(root, issues):
    base = root / "third_party/robocasa"
    source = base / "robocasa/__init__.py"
    seen = set()
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(
            ("robocasa.environments.kitchen.atomic.", "robocasa.environments.kitchen.composite.")
        ):
            continue
        path = base / (node.module.replace(".", "/") + ".py")
        classes = {
            n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        }
        for alias in node.names:
            if alias.name in seen or alias.name not in classes:
                continue
            seen.add(alias.name)
            instruction = (ast.get_docstring(classes[alias.name]) or "").split("\n\n")[0].strip()
            instruction = instruction or robocasa_instruction(classes, alias.name)
            yield (
                _task("robocasa", alias.name, {"task_name": alias.name}, instruction),
                path,
            )


def _robotwin(root, issues):
    base = root / "third_party/RoboTwin"
    names = yaml.safe_load((base / "env_cfg/eval/all_tasks.yml").read_text())["tasks"]
    for name in names:
        source = base / "envs" / (name + ".py")
        if not source.is_file():
            raise ValueError(f"RoboTwin task source missing: {name}")
        description = base / "description/task_instruction" / (name + ".json")
        data = json.loads(description.read_text()) if description.is_file() else {}
        instruction = data.get("full_description")
        if not isinstance(instruction, str) or not instruction.strip():
            issues.append(
                {
                    "backend": "robotwin",
                    "task": name,
                    "reason": "task lacks the upstream full_description instruction",
                }
            )
            continue
        instruction = instruction.strip()
        for suffix, robot, embodiment in (
            ("", ROBOTS["robotwin"], "piper"),
            ("-aloha-agilex", "robotwin/robotwin-aloha-agilex", "aloha_agilex"),
        ):
            yield (
                _task(
                    "robotwin",
                    name + suffix,
                    {"task_name": name, "embodiment": embodiment},
                    instruction,
                    robot=robot,
                ),
                source,
            )


def _robodojo(root, issues):
    base = root / "third_party/RoboDojo/task/RoboDojo"
    for config in sorted((base / "config").glob("*.yml")):
        name = config.stem
        source = base / "tasks" / (name + ".py")
        if not source.is_file() or name.startswith("_"):
            continue
        tree = ast.parse(source.read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)
        if cls is None:
            continue
        description = robodojo_instruction(tree, name)
        yield (
            _task(
                "robodojo",
                name,
                {"task_name": name, "env_cfg_type": "arx_x5"},
                description,
            ),
            source,
        )


def _behavior(root, issues):
    source = root / "third_party/BEHAVIOR-1K/docs/challenge/task_data.json"
    for entry in json.loads(source.read_text())["tasks"]:
        if not entry.get("scene_model") or not entry.get("instruction"):
            issues.append(
                {
                    "id": "behavior/" + task_name(entry["id"]),
                    "reason": "challenge task lacks an explicit scene or instruction",
                }
            )
            continue
        yield (
            _task(
                "behavior",
                entry["id"],
                {"activity_name": entry["id"], "scene_model": entry["scene_model"]},
                entry["instruction"],
            ),
            source,
        )


def _isaaclab(root, issues):
    for source in sorted((root / "catalog/tasks/isaaclab").glob("*/task.yaml")):
        yield Task.load(source), source


def _abc(root, issues):
    from roboenv.tasks.abc import discover_tasks

    yield from discover_tasks(root)


def _simple(root, issues):
    from roboenv.tasks.simple import discover_tasks

    yield from discover_tasks(root)


DISCOVERERS = {
    "abc": _abc,
    "simple": _simple,
    "isaaclab": _isaaclab,
    "libero": _libero,
    "robocasa": _robocasa,
    "robotwin": _robotwin,
    "robodojo": _robodojo,
    "behavior": _behavior,
}


def native_identity(task):
    """Identify a native task and embodiment independently of its public name."""
    selectors = {
        "simple": ("env_id",),
        "libero": ("suite_name", "task_id"),
        "behavior": ("activity_name",),
    }.get(task.backend, ("task_name",))
    return (task.backend, task.robot, *(task.backend_kwargs[key] for key in selectors))


def discover(project, destination, backends=BACKENDS):
    root, lock = load_lock(project)
    if set(backends) - set(DISCOVERERS):
        raise ValueError("unknown simulator discovery backend")
    source_status = status(root, backends)
    if not source_status["ready"]:
        raise ValueError("upstream pins or patches are not ready; run roboenv sources sync first")
    examples = root / "catalog/tasks"
    existing = load_catalog(examples) if examples.exists() else {}
    by_native = {}
    for task, _ in existing.values():
        identity = native_identity(task)
        if identity in by_native:
            raise ValueError(f"ambiguous catalog native task: {task.id}")
        by_native[identity] = task
    bundle = Bundle(destination)
    records = {}
    issues = []
    for backend in backends:
        for task, source in DISCOVERERS[backend](root, issues):
            old = by_native.get(native_identity(task))
            if old is not None:
                task = replace(task, id=old.id, instruction=old.instruction)
            task = Task.from_dict(task.as_dict())
            if task.id in records:
                raise ValueError(f"duplicate discovered task: {task.id}")
            bundle.add_yaml(f"tasks/{task.id}/task.yaml", task.as_dict())
            records[task.id] = {
                "backend": backend,
                "source": str(source.relative_to(root)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
    metadata = {
        "schema": "roboenv/discovered-tasks/1",
        "tasks": records,
        "source_revisions": {item["name"]: item["actual"] for item in source_status["sources"]},
        "patches": {
            item["name"]: lock["sources"][item["name"]].get("patches", [])
            for item in source_status["sources"]
        },
        "execution_verified": False,
        "excluded": issues,
    }
    bundle.add("inventory.json", json.dumps(metadata, indent=2) + "\n")
    bundle.commit(
        {
            "kind": "discovered-environment-tasks",
            "task_count": len(records),
            "execution_verified": False,
        }
    )
    return {
        "tasks": len(records),
        "backends": dict(Counter(r["backend"] for r in records.values())),
        "execution_verified": False,
        "excluded": len(issues),
    }
