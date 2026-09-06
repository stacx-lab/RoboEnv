"""Read and discover environment tasks without importing simulator engines."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

from .harbor import Bundle
from .project import project_root
from .task import Task
from .upstreams import load_lock, status

BACKENDS = ("libero", "robocasa", "robotwin", "robodojo", "behavior")
ROBOTS = {
    "libero": "libero-panda",
    "robocasa": "robocasa-panda-omron",
    "robotwin": "robotwin-dual-piper",
    "robodojo": "robodojo-arx-x5",
    "behavior": "behavior-r1pro",
}


def load_catalog(root):
    result = {}
    for path in sorted(Path(root).glob("*/task.yaml")):
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
    }


def _literal(path, variable):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == variable for n in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"upstream metadata variable not found: {variable}")


def _task(backend, identifier, kwargs, instruction, *, robot=None):
    return Task(
        id=identifier,
        backend=backend,
        backend_task=identifier,
        backend_kwargs=kwargs,
        robot=robot or ROBOTS[backend],
        instruction=instruction,
        action_abstraction="symbolic-primitives"
        if backend == "behavior"
        else "cartesian-primitives",
    )


def _libero(root, issues):
    base = root / "third_party/cap-x/capx/third_party/LIBERO-PRO/libero/libero"
    mapping = _literal(base / "benchmark/libero_suite_task_map.py", "libero_task_map")
    module = ast.parse((base / "benchmark/__init__.py").read_text())
    registered = set()
    for cls in module.body:
        if isinstance(cls, ast.ClassDef) and any(
            isinstance(d, ast.Name) and d.id == "register_benchmark" for d in cls.decorator_list
        ):
            for node in ast.walk(cls):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if any(isinstance(n, ast.Attribute) and n.attr == "name" for n in node.targets):
                        registered.add(node.value.value)
    for suite in sorted(registered):
        if suite not in mapping:
            issues.append(
                {"id": suite, "reason": "no static task map; composite suite not expanded"}
            )
            continue
        for index, name in enumerate(mapping[suite]):
            bddl = base / "bddl_files" / suite / (name + ".bddl")
            if not bddl.is_file():
                issues.append({"id": f"{suite}:{index}", "reason": "missing BDDL definition"})
                continue
            match = re.search(r"\(:language\s+(.*?)\)", bddl.read_text(), re.S)
            instruction = (
                " ".join(match.group(1).split()).strip('"') if match else name.replace("_", " ")
            )
            short = suite.removeprefix("libero_")
            group = (
                "libero"
                if suite
                in {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
                else "libero-pro"
            )
            yield (
                _task(
                    "libero",
                    f"{group}.{short}_{index}",
                    {"suite_name": suite, "task_id": index},
                    instruction,
                ),
                bddl,
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
        instruction = (
            "Complete the benchmark task: "
            + name.replace("_", " ")
            + ". Inspect the live task description."
        )
        description = base / "description/task_instruction" / (name + ".json")
        if description.is_file():
            data = json.loads(description.read_text())
            if isinstance(data.get("description"), str):
                instruction = data["description"]
        for group, robot, embodiment in (
            ("robotwin", ROBOTS["robotwin"], "piper"),
            ("robotwin-aloha", "robotwin-aloha-agilex", "aloha_agilex"),
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
        yield _task("robodojo", "robodojo." + name, {"task_name": name}, description), source


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


DISCOVERERS = {
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
                task = _task(
                    backend, task.id, task.backend_kwargs, old.instruction, robot=task.robot
                )
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
