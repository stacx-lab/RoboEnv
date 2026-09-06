"""Portable, preflighted batch generation and execution of standard Harbor tasks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .catalogue import BACKENDS, discover, load_catalog
from .deployment import deployment_contract, load_deployment
from .harbor import Bundle, export_task, registry_name
from .project import project_root
from .validation import json_loads


def deployment_map(path):
    path = Path(path).resolve()
    data = json_loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("deployment map must map backend IDs to manifest paths")
    result = {}
    for backend, value in data.items():
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", backend)
            or not isinstance(value, str)
            or not value
        ):
            raise ValueError("deployment map needs backend IDs and nonempty manifest paths")
        source = Path(value)
        result[backend] = (source if source.is_absolute() else path.parent / source).resolve()
    return result


def deployment_environment(mapping):
    """Resolve local depot paths in memory; never serialize them into public tasks."""
    environment = {}
    for backend, path in mapping.items():
        spec = load_deployment(path, backend)
        for index, mount in enumerate(spec.get("mounts", [])):
            environment[f"ROBOENV_DEPOT_{backend.upper()}_{index}"] = mount["source"]
    return environment


def validate_harbor_task(path):
    """Use Harbor's actual parser/loader, not a parallel imitation of its schema."""
    from harbor.models.task.task import Task as HarborTask

    if not HarborTask.is_valid_dir(path):
        raise ValueError("Harbor rejected the task directory or required verifier files")
    task = HarborTask(Path(path))
    if task.config.schema_version != "1.4":
        raise ValueError("Expected Harbor task schema 1.4")
    return {"name": task.config.task.name, "schema_version": task.config.schema_version}


def export_taskset(
    project,
    source,
    deployments,
    destination,
    wheel,
    *,
    backends=BACKENDS,
    identifiers=None,
    seeds=None,
    include_references=False,
    check_images=False,
    validate=True,
):
    root = project_root(project)
    if source is None:
        with tempfile.TemporaryDirectory(prefix="roboenv-discover-") as temporary:
            discover(root, temporary, backends)
            return export_taskset(
                root,
                Path(temporary) / "tasks",
                deployments,
                destination,
                wheel,
                backends=backends,
                identifiers=identifiers,
                seeds=seeds,
                include_references=include_references,
                check_images=check_images,
                validate=validate,
            )
    inventory = load_catalog(source)
    mapping = deployment_map(deployments)
    destination = Path(destination).resolve()
    if destination == root or destination.is_relative_to(root / "src"):
        raise ValueError("task outputs must not overwrite project source")
    selected = [
        (task, path)
        for task, path in inventory.values()
        if task.backend in backends and (not identifiers or task.id in identifiers)
    ]
    if not selected:
        raise ValueError("no tasks matched the selection")
    if identifiers and set(identifiers) - {t.id for t, _ in selected}:
        raise ValueError("some requested task IDs were not found in the selected catalog")
    missing = {t.backend for t, _ in selected} - set(mapping)
    if missing:
        raise ValueError("missing deployment manifests for: " + ", ".join(sorted(missing)))
    if check_images:
        for backend in {t.backend for t, _ in selected}:
            image = json.loads(mapping[backend].read_text())["image"]
            subprocess.run(
                ["docker", "image", "inspect", image], check=True, capture_output=True, timeout=30
            )
    plans, records, names = [], [], set()
    for task, path in selected:
        for seed in seeds if seeds is not None else [task.seed]:
            name = registry_name(task.id, seed)
            if name in names:
                raise ValueError("duplicate generated task name/seed")
            names.add(name)
            folder = name.split("/", 1)[1]
            bundle, metadata = export_task(
                path,
                root / "catalog",
                destination / "tasks" / folder,
                None,
                wheel,
                seed=seed,
                deployment_path=mapping[task.backend],
                include_reference=include_references,
                portable=True,
                prepare_only=True,
            )
            plans.append((bundle, metadata))
            records.append(
                {
                    "id": task.id,
                    "name": name,
                    "seed": seed,
                    "backend": task.backend,
                    "path": "tasks/" + folder,
                    "has_reference": "solution/solve.sh" in bundle.contents,
                }
            )
    # All task inputs and output conflicts are checked before writing any task.
    if validate:
        with tempfile.TemporaryDirectory(prefix="roboenv-format-") as temporary:
            for index, (bundle, metadata) in enumerate(plans):
                probe = Bundle(Path(temporary) / str(index))
                probe.contents = bundle.contents
                probe.commit(metadata)
                validate_harbor_task(probe.root)
    output = Bundle(destination)
    output.add(
        "job.json",
        json.dumps(
            {
                "tasks": [{"path": item["path"]} for item in records],
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "retry": {"max_retries": 0},
            },
            indent=2,
        )
        + "\n",
    )
    report = {
        "schema": "roboenv/taskset/1",
        "tasks": records,
        "harbor_schema": "1.4",
        "harbor_loader_validated": validate,
        "image_presence_checked": check_images,
        "execution_verified": False,
        "reference_policy": "explicit-opt-in; project-authored unless upstream provenance supplied",
        "deployments": {
            backend: deployment_contract(load_deployment(mapping[backend], backend))
            for backend in sorted({task.backend for task, _ in selected})
        },
        "required_env": sorted(
            deployment_environment({b: mapping[b] for b in {t.backend for t, _ in selected}})
        ),
    }
    source_inventory = Path(source).parent / "inventory.json"
    if source_inventory.is_file():
        output.add("source-inventory.json", source_inventory.read_bytes())
    output.add("taskset.json", json.dumps(report, indent=2) + "\n")
    output.preflight()
    for bundle, metadata in plans:
        bundle.commit(metadata)
    output.commit({"kind": "generated-harbor-taskset", "task_count": len(records)})
    return {
        "tasks": len(records),
        "backends": dict(Counter(r["backend"] for r in records)),
        "harbor_loader_validated": validate,
        "references": sum(r["has_reference"] for r in records),
        "execution_verified": False,
    }


def run_taskset(path, deployments, *, agent, model=None, extra_args=()):
    root = Path(path).resolve()
    manifest = json_loads((root / "taskset.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "roboenv/taskset/1"
        or not isinstance(manifest.get("tasks"), list)
        or not manifest["tasks"]
    ):
        raise ValueError("unsupported or empty taskset manifest")
    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("an explicit agent is required")
    if agent == "oracle" and not all(t["has_reference"] for t in manifest["tasks"]):
        raise ValueError("Oracle requires an explicit reference for every selected task")
    executable = shutil.which("harbor")
    if not executable:
        raise ValueError("Install Harbor before running tasks")
    mapping = deployment_map(deployments)
    backends = {task["backend"] for task in manifest["tasks"]}
    if backends - set(mapping):
        raise ValueError("missing deployment manifests for selected task backends")
    contracts = manifest.get("deployments")
    if not isinstance(contracts, dict) or set(contracts) != backends:
        raise ValueError("taskset lacks pinned deployment identities; regenerate the taskset")
    for backend in backends:
        current = deployment_contract(load_deployment(mapping[backend], backend))
        if contracts[backend] != current:
            raise ValueError(
                "deployment image, mount targets or environment changed; regenerate the taskset"
            )
    depots = deployment_environment({backend: mapping[backend] for backend in backends})
    if set(manifest["required_env"]) - set(depots):
        raise ValueError("deployment map does not supply this taskset's required asset depots")
    environment = {**os.environ, **depots}
    command = [executable, "run", "-c", "job.json", "-a", agent]
    if model:
        command += ["-m", model]
    command += list(extra_args)
    # Keep host-side Harbor scratch on the explicitly chosen artifact filesystem.
    # Docker's separate data filesystem must also have space (checked by the builder).
    with tempfile.TemporaryDirectory(prefix=".harbor-", dir=root) as temporary:
        environment["TMPDIR"] = temporary
        return subprocess.run(command, cwd=root, env=environment).returncode
