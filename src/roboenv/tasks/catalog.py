"""Read and discover environment tasks without importing simulator engines."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from roboenv.project import project_root
from roboenv.provisioning.sources import load_lock, status
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.model import Task

BACKENDS = ("libero", "robocasa", "robotwin", "robodojo", "behavior", "isaaclab", "simple")
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


def _literal(path, variable):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == variable for n in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"upstream metadata variable not found: {variable}")


def _task(backend, identifier, kwargs, instruction, *, robot=None, device_bindings=None):
    task_name = identifier.removeprefix(backend + ".")
    return Task(
        id=f"{backend}/{task_name}",
        backend=backend,
        backend_task=identifier,
        backend_kwargs={
            **kwargs,
        },
        robot=robot or ROBOTS[backend],
        device_bindings=device_bindings,
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
                f"libero.{short}_{entry['task_id']}",
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
            instruction = (
                instruction
                or f"Complete the benchmark task {alias.name}; inspect the live environment goal."
            )
            yield (
                _task("robocasa", "robocasa." + alias.name, {"task_name": alias.name}, instruction),
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
        for group, robot, embodiment in (
            ("robotwin", ROBOTS["robotwin"], "piper"),
            ("robotwin-aloha", "robotwin/robotwin-aloha-agilex", "aloha_agilex"),
        ):
            yield (
                _task(
                    "robotwin",
                    f"{group}.{name}",
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
        description = (ast.get_docstring(cls) or "").split("\n\n")[0].strip()
        description = description or "Complete the benchmark task: " + name.replace("_", " ") + "."
        yield (
            _task(
                "robodojo",
                "robodojo." + name,
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
                    "id": "behavior." + entry["id"],
                    "reason": "challenge task lacks an explicit scene or instruction",
                }
            )
            continue
        yield (
            _task(
                "behavior",
                "behavior." + entry["id"],
                {"activity_name": entry["id"], "scene_model": entry["scene_model"]},
                entry["instruction"],
            ),
            source,
        )


def _isaaclab(root, issues):
    for source in sorted((root / "catalog/tasks/isaaclab").glob("*/task.yaml")):
        yield Task.load(source), source


def _simple(root, issues):
    from roboenv.tasks.simple import discover_tasks

    yield from discover_tasks(root)


DISCOVERERS = {
    "simple": _simple,
    "isaaclab": _isaaclab,
    "libero": _libero,
    "robocasa": _robocasa,
    "robotwin": _robotwin,
    "robodojo": _robodojo,
    "behavior": _behavior,
}


def discover(project, destination, backends=BACKENDS):
    root, lock = load_lock(project)
    if set(backends) - set(DISCOVERERS):
        raise ValueError("unknown simulator discovery backend")
    source_status = status(root, backends)
    if not source_status["ready"]:
        raise ValueError("upstream pins or patches are not ready; run roboenv sources sync first")
    examples = root / "catalog/tasks"
    existing = load_catalog(examples) if examples.exists() else {}
    bundle = Bundle(destination)
    records = {}
    issues = []
    for backend in backends:
        for task, source in DISCOVERERS[backend](root, issues):
            if task.id in records:
                raise ValueError(f"duplicate discovered task: {task.id}")
            if task.id in existing:
                old, _ = existing[task.id]
                task = replace(task, instruction=old.instruction)
            bundle.add_yaml(f"tasks/{task.id}/task.yaml", asdict(task))
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
