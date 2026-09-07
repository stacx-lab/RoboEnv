"""Prepare, or explicitly run, standard Harbor GPU validation using only Oracle.

Connectivity probes and solved reference trials are separate evidence categories.
No simulator import, model client, asset download, or GPU execution happens while
preparing. Raw Harbor logs stay private; only allowlisted receipt fields are public.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from roboenv.devices.bindings.spec import binding_path
from roboenv.harbor.deployment import load_deployment
from roboenv.harbor.taskset import deployment_map, export_taskset, validate_harbor_task
from roboenv.privacy import validate_public_text
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.catalog import load_catalog
from roboenv.validation import json_loads

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = {
    "libero-panda": ("libero", {}),
    "robocasa-panda-omron": ("robocasa", {}),
    "robotwin-dual-piper": ("robotwin", {"embodiment": "piper"}),
    "robotwin-aloha-agilex": ("robotwin", {"embodiment": "aloha_agilex"}),
    "robodojo-arx-x5": ("robodojo", {"env_cfg_type": "arx_x5"}),
    "robodojo-yam": ("robodojo", {"env_cfg_type": "yam"}),
    "behavior-r1pro": ("behavior", {}),
}
DEFAULT_BACKENDS = ("libero", "robocasa", "robotwin", "robodojo")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_receipt(path, receipt):
    content = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    validate_public_text(content, "validation-receipt.json")
    Path(path).write_text(content)


def _update_bundle(root, updates):
    saved = json_loads((root / "bundle.json").read_text())
    bundle = Bundle(root)
    for name, expected in saved["files"].items():
        path = root / name
        if digest(path) != expected:
            raise ValueError("generated validation task was modified before execution")
        bundle.add(name, path.read_bytes(), path.stat().st_mode & 0o777)
    for name, value in updates.items():
        bundle.add(name, value[0], value[1])
    bundle.commit({key: value for key, value in saved.items() if key != "files"})


def _verify_prepared_bundle(root):
    saved = json_loads((root / "bundle.json").read_text())
    for name, expected in saved["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("prepared validation artifact escapes its task bundle")
        if digest(path) != expected:
            raise ValueError("prepared validation bundle changed; generate a new output")


def _probe_program(names):
    return (
        "# Project-authored read-only device connectivity probe; not a benchmark solution.\n"
        "from roboenv.client import Client\n"
        "client = Client()\n"
        "inventory = client.devices()\n"
        "assert inventory['schema_version'] == 'roboenv/devices/1'\n"
        f"assert sorted(device['name'] for device in inventory['devices']) == {names!r}\n"
        "for descriptor in inventory['devices']:\n"
        "    action = descriptor['actions']['read']\n"
        "    assert action['supported'] is True and action['read_only'] is True\n"
        "    result = client.device(descriptor['name']).read()\n"
        "    assert result['read_only'] is True\n"
    )


def select_cases(
    *,
    project=ROOT,
    source=None,
    backends=DEFAULT_BACKENDS,
    cases=None,
    robots=None,
    modes=("reference",),
):
    project = Path(project)
    if not modes or set(modes) - {"reference", "probe"}:
        raise ValueError("select reference and/or probe validation")
    if not backends or set(backends) - {value[0] for value in ROBOTS.values()}:
        raise ValueError("unknown validation backend")
    if robots and set(robots) - set(ROBOTS):
        raise ValueError("unknown validation robot")
    inventory = load_catalog(project / "catalog/tasks")
    if source is not None:
        extra = load_catalog(source)
        if set(extra) & set(inventory):
            raise ValueError("additional task source must not replace built-in example IDs")
        inventory.update(extra)
    selected = [
        (task, path)
        for task, path in inventory.values()
        if task.backend in backends and (not cases or task.id in cases)
    ]
    if cases and set(cases) - {task.id for task, _ in selected}:
        raise ValueError("some selected case IDs were not found for these backends")
    if set(backends) - {task.backend for task, _ in inventory.values()}:
        raise ValueError("selected backend needs an explicit task source; no scene is inferred")
    plans = []
    if "reference" in modes:
        for task, source_path in selected:
            if robots and task.robot not in robots:
                continue
            reference = source_path.with_name("solution.py")
            if not reference.is_file() or reference.is_symlink():
                raise ValueError("reference validation requires an explicit regular solution.py")
            note = reference.with_name("solution-source.json")
            if note.exists() and json_loads(note.read_text()).get("official") is True:
                raise ValueError("this validation lane accepts project-authored references only")
            plans.append(
                (
                    "reference",
                    replace(task, id=task.backend + "/reference-" + task.id.split("/")[-1]),
                    source_path,
                    reference.read_text(),
                )
            )
    if "probe" in modes:
        for robot, (backend, overrides) in ROBOTS.items():
            if backend not in backends or (robots and robot not in robots):
                continue
            candidates = [(task, path) for task, path in selected if task.backend == backend]
            if not candidates:
                continue
            task, source_path = next(
                ((task, path) for task, path in candidates if task.robot == robot), candidates[0]
            )
            if backend == "behavior" and not task.backend_kwargs.get("scene_model"):
                raise ValueError("BEHAVIOR validation needs an explicit scene_model in task source")
            task = replace(
                task,
                id=f"{task.backend}/probe-{robot}-{task.id.split('/')[-1]}",
                robot=robot,
                device_bindings=None,
                backend_kwargs={**task.backend_kwargs, **overrides},
                action_abstraction="symbolic-primitives"
                if backend == "behavior"
                else task.action_abstraction,
            )
            binding = yaml.safe_load(binding_path(project / "catalog/robots" / robot).read_text())
            plans.append(("probe", task, source_path, _probe_program(sorted(binding["devices"]))))
    if not plans:
        raise ValueError("no validation cases matched")
    return plans


def prepare(
    deployments,
    sdk_wheel,
    out,
    *,
    project=ROOT,
    source=None,
    backends=DEFAULT_BACKENDS,
    cases=None,
    robots=None,
    modes=("reference",),
    seeds=(0,),
):
    project, sdk_wheel, out = (
        Path(project).resolve(),
        Path(sdk_wheel).resolve(),
        Path(out).resolve(),
    )
    if out.exists():
        raise ValueError("validation output must be new; old evidence is never overwritten")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("validation seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate validation seeds")
    plans = select_cases(
        project=project, source=source, backends=backends, cases=cases, robots=robots, modes=modes
    )
    backends = sorted({task.backend for _, task, _, _ in plans})
    mapping = deployment_map(deployments)
    if set(backends) - set(mapping):
        raise ValueError("missing validation deployments")
    provenance = {}
    for backend in backends:
        spec = load_deployment(mapping[backend], backend)
        provenance[backend] = {
            "image": spec["image"],
            "image_presence_verified": False,
            "runtime_wheel_sha256": spec.get("runtime_wheel_sha256"),
            "source_revisions": spec.get("source_revisions", {}),
            "upstream_lock_sha256": spec.get("upstream_lock_sha256"),
            "executable_depot_manifests": spec.get("executable_depot_manifests", {}),
            "provenance_basis": "build-deployment declaration; image inspection occurs only on run",
        }
        provenance[backend]["build_provenance_complete"] = bool(
            re.fullmatch(r"[0-9a-f]{64}", str(spec.get("runtime_wheel_sha256", "")))
            and re.fullmatch(r"[0-9a-f]{64}", str(spec.get("upstream_lock_sha256", "")))
            and isinstance(spec.get("source_revisions"), dict)
            and spec["source_revisions"]
            and all(
                re.fullmatch(r"[0-9a-f]{40}", str(value))
                for value in spec["source_revisions"].values()
            )
        )
    normalized = out / "inputs/tasks"
    by_id = {}
    for mode, task, source_path, program in plans:
        task_source = Bundle(normalized / task.id)
        task_source.add_yaml("task.yaml", asdict(task))
        task_source.add("solution.py", program)
        task_source.add(
            "solution-source.json",
            json.dumps(
                {
                    "origin": "project-reference"
                    if mode == "reference"
                    else "project-connectivity-probe",
                    "official": False,
                    "replay_verified": False,
                }
            )
            + "\n",
        )
        task_source.commit({"kind": "validation-input", "mode": mode})
        by_id[task.id] = {
            "mode": mode,
            "source_task_sha256": digest(source_path),
            "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
            "backend_task": task.backend_task,
            "robot": task.robot,
            "bindings_sha256": digest(binding_path(project / "catalog/robots" / task.robot)),
            "expected_devices": sorted(
                yaml.safe_load(binding_path(project / "catalog/robots" / task.robot).read_text())[
                    "devices"
                ]
            ),
        }
    bundle = out / "taskset"
    export_taskset(
        project,
        normalized,
        deployments,
        bundle,
        sdk_wheel,
        backends=backends,
        seeds=seeds,
        include_references=True,
    )
    taskset = json_loads((bundle / "taskset.json").read_text())
    records = []
    for record in taskset["tasks"]:
        details = by_id[record["id"]]
        marker = {
            "schema": "roboenv/validation-program/1",
            "case_id": record["id"],
            "seed": record["seed"],
            "program_sha256": details["program_sha256"],
        }
        recorder = (
            "import json\nfrom pathlib import Path\n"
            f"Path('/logs/agent/validation-program.json').write_text(json.dumps({marker!r}))\n"
        )
        _update_bundle(
            bundle / record["path"],
            {
                "solution/record-completion.py": (recorder, 0o644),
                "solution/solve.sh": (
                    "#!/bin/sh\nset -eu\nroboenv replay /solution/solution.py\n"
                    "/usr/bin/python3 -I /solution/record-completion.py\n",
                    0o755,
                ),
            },
        )
        validate_harbor_task(bundle / record["path"])
        records.append(
            {
                **record,
                **details,
                "execution_attempted": False,
                "boot_verified": False,
                "device_read_probe_passed": False,
                "reference_program_completed": False,
                "reference_replay_passed": False,
                "final_verifier_success": False,
            }
        )
    job = json_loads((bundle / "job.json").read_text())
    job.update(job_name="gpu-validation", jobs_dir="jobs")
    _update_bundle(bundle, {"job.json": (json.dumps(job, indent=2) + "\n", 0o644)})
    receipt = {
        "schema": "roboenv/gpu-validation/1",
        "phase": "prepared",
        "harbor_schema": "1.4",
        "harbor_loader_validated": True,
        "agent": "oracle",
        "model_calls": 0,
        "reference_official": False,
        "sdk_wheel_sha256": digest(sdk_wheel),
        "deployments": provenance,
        "cases": records,
        "all_selected_checks_passed": False,
        "selected_evidence_complete": False,
        "fresh_process_replay_certificate": False,
    }
    write_receipt(out / "receipt.json", receipt)
    return receipt


def _read_json(path):
    return json_loads(path.read_text()) if path.is_file() else {}


def _trial_evidence(record, result_path):
    directory = result_path.parent
    result = _read_json(result_path)
    report = _read_json(directory / "verifier/submit.json")
    reward = _read_json(directory / "verifier/reward.json").get("reward")
    marker = _read_json(directory / "agent/validation-program.json")
    if not all(isinstance(value, dict) for value in (result, report, marker)):
        raise ValueError("validation evidence must contain objects")
    journal = directory / "verifier/actions.jsonl"
    identity = (report.get("task"), report.get("seed")) == (record["backend_task"], record["seed"])
    valid = bool(
        identity
        and report.get("valid") is True
        and journal.is_file()
        and report.get("journal", {}).get("sha256") == digest(journal)
    )
    marker_valid = marker == {
        "schema": "roboenv/validation-program/1",
        "case_id": record["id"],
        "seed": record["seed"],
        "program_sha256": record["program_sha256"],
    }
    program_completed = bool(
        marker_valid
        and not (directory / "agent/exit-code.txt").exists()
        and result.get("exception_info") is None
    )
    boot = bool(identity and report.get("health", {}).get("worker", {}).get("state") == "ready")
    read_devices, failed_read_devices, mutating_probe_calls = set(), set(), False
    if valid:
        for line in journal.read_text().splitlines():
            event = json_loads(line)
            request, response = event.get("request", {}), event.get("response", {})
            arguments = request.get("args", {})
            if (
                event.get("phase") == "error"
                and response.get("ok") is False
                and request.get("op") == "devices.call"
                and arguments.get("action") == "read"
            ):
                name = arguments.get("device")
                failed_read_devices.add(
                    name if name in record["expected_devices"] else "unknown-device"
                )
            if event.get("phase") != "commit" or response.get("ok") is not True:
                continue
            if request.get("op") == "devices.call":
                args = request.get("args", {})
                if (
                    args.get("action") == "read"
                    and response.get("result", {}).get("read_only") is True
                ):
                    read_devices.add(args.get("device"))
                else:
                    mutating_probe_calls = True
            elif request.get("op") not in {
                "health",
                "api",
                "observe",
                "score.peek",
                "environment.describe",
                "devices.list",
                "devices.describe",
                "operations.get",
                "submit",
            }:
                mutating_probe_calls = True
    final_success = bool(
        valid
        and type(reward) in {int, float}
        and reward == 1.0
        and report.get("trial_success") is True
        and report.get("benchmark_eligible") is True
        and result.get("verifier_result", {}).get("rewards", {}).get("reward") == 1.0
    )
    probe_success = bool(
        record["mode"] == "probe"
        and boot
        and program_completed
        and valid
        and read_devices == set(record["expected_devices"])
        and not mutating_probe_calls
    )
    failure = "harbor-exception" if result.get("exception_info") else None
    if failure is None and record["mode"] == "probe" and not probe_success:
        if failed_read_devices:
            failure = "device-read-failed"
        elif not program_completed:
            failure = "oracle-program-failed"
        elif not valid:
            failure = "invalid-verifier-evidence"
        elif mutating_probe_calls:
            failure = "probe-mutated-environment"
        else:
            failure = "incomplete-device-read-evidence"
    return {
        "boot_verified": boot,
        "oracle_program_completed": program_completed,
        "device_read_probe_passed": probe_success,
        "reference_program_completed": record["mode"] == "reference" and program_completed,
        "reference_replay_passed": record["mode"] == "reference"
        and program_completed
        and boot
        and final_success,
        "final_verifier_success": final_success,
        "verifier_evidence_valid": valid,
        "reward": reward if type(reward) in {int, float} and reward in {0, 1} else None,
        "journal_sha256": digest(journal) if journal.is_file() else None,
        "trial_result_sha256": digest(result_path),
        "failure_category": failure,
        "failed_device_reads": sorted(failed_read_devices),
    }


def run_validation(out, deployments):
    out = Path(out).resolve()
    receipt = _read_json(out / "receipt.json")
    if receipt.get("schema") != "roboenv/gpu-validation/1" or receipt.get("phase") != "prepared":
        raise ValueError("execution requires a fresh prepared validation receipt")
    _verify_prepared_bundle(out / "taskset")
    for record in receipt["cases"]:
        task = out / "taskset" / record["path"]
        _verify_prepared_bundle(task)
        if digest(task / "solution/solution.py") != record["program_sha256"]:
            raise ValueError("reference program differs from its validation receipt")
    for provenance in receipt["deployments"].values():
        try:
            actual = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", provenance["image"]],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            receipt.update(phase="preflight-failed", failure_category="image-unavailable")
            write_receipt(out / "receipt.json", receipt)
            return receipt
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", actual):
            raise ValueError("Docker did not return a valid immutable image ID")
        if provenance["image"].startswith("sha256:") and actual != provenance["image"]:
            raise ValueError("Docker image identity does not match deployment")
        provenance.update(image_presence_verified=True, inspected_image_id=actual)
    receipt["phase"] = "running"
    write_receipt(out / "receipt.json", receipt)
    code = (
        "import sys; from roboenv.harbor.taskset import run_taskset; "
        "raise SystemExit(run_taskset(sys.argv[1], sys.argv[2], agent='oracle'))"
    )
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    with (out / "harbor.private.log").open("w") as log:
        (out / "harbor.private.log").chmod(0o600)
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(out / "taskset"),
                str(Path(deployments).resolve()),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    trials, malformed_results = {}, 0
    for path in (out / "taskset/jobs").rglob("result.json"):
        try:
            value = _read_json(path)
        except (ValueError, OSError):
            malformed_results += 1
            continue
        if isinstance(value, dict) and "task_name" in value:
            trials.setdefault(value["task_name"], []).append(path)
    for record in receipt["cases"]:
        record["execution_attempted"] = True
        matches = trials.get(record["name"], [])
        if len(matches) != 1:
            record["failure_category"] = "missing-or-duplicate-trial-result"
            continue
        try:
            record.update(_trial_evidence(record, matches[0]))
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            record["failure_category"] = "malformed-evidence"
    receipt.update(
        phase="completed",
        harbor_exit_code=process.returncode,
        malformed_trial_results=malformed_results,
        all_selected_checks_passed=process.returncode == 0
        and malformed_results == 0
        and all(
            record["device_read_probe_passed"]
            if record["mode"] == "probe"
            else record["reference_replay_passed"]
            for record in receipt["cases"]
        ),
    )
    receipt["selected_evidence_complete"] = receipt["all_selected_checks_passed"] and all(
        item["build_provenance_complete"] and item["image_presence_verified"]
        for item in receipt["deployments"].values()
    )
    write_receipt(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployments", required=True, type=Path)
    parser.add_argument("--sdk-wheel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Additional explicit normalized tasks")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--robots", nargs="+", choices=sorted(ROBOTS))
    parser.add_argument("--modes", nargs="+", choices=("probe", "reference"), default=["reference"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--run", action="store_true", help="Run model-free Oracle trials; may use GPUs"
    )
    args = parser.parse_args()
    try:
        report = prepare(
            args.deployments,
            args.sdk_wheel,
            args.out,
            source=args.source,
            backends=args.backends,
            cases=args.cases,
            robots=args.robots,
            modes=args.modes,
            seeds=args.seeds,
        )
        if args.run:
            report = run_validation(args.out, args.deployments)
    except ValueError as error:
        message = str(error)
        try:
            validate_public_text(message, "validation-error")
        except ValueError:
            message = "Validation input rejected; inspect private local artifacts."
        parser.exit(1, message + "\n")
    except (OSError, subprocess.SubprocessError):
        parser.exit(1, "Validation setup/execution failed; inspect private local artifacts.\n")
    print(json.dumps(report, indent=2))
    return 0 if not args.run or report["all_selected_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
