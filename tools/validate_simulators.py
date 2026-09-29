"""Reproducible, model-free simulator validation on one host: launch, replay, certify.

For every selected task and seed the tool starts the world as an owner does
(``roboenv launch``), runs one program through ``roboenv replay`` as an agent would,
certifies the run with ``roboenv verify`` and stops the world. A probe replays a
project-authored read-only device connectivity program; a reference replays the task's
checked-in ``solution.py``. A probe passes when every declared readable device answered
a read and nothing mutated the world; a reference passes only with a valid, fully
successful certificate. No model is called, no asset is downloaded and no simulator is
imported into this process. Raw logs stay in a private file; the public receipt carries
allowlisted fields only.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlsplit

import roboenv
from roboenv.devices.bindings.spec import BindingPlan, binding_resource
from roboenv.privacy import validate_public_text
from roboenv.tasks.artifacts import Bundle
from roboenv.tasks.catalog import load_catalog
from roboenv.tasks.libero import ROBOTS as LIBERO_ROBOTS
from roboenv.validation import json_loads
from roboenv.verify import READ_ONLY_ACTIONS, READ_ONLY_OPS, strict_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "roboenv/simulator-validation/1"
ROBOTS = {
    **{robot: ("libero", {}) for robot in LIBERO_ROBOTS.values()},
    "robocasa/robocasa-panda-omron": ("robocasa", {}),
    "robotwin/robotwin-dual-piper": ("robotwin", {"embodiment": "piper"}),
    "robotwin/robotwin-aloha-agilex": ("robotwin", {"embodiment": "aloha_agilex"}),
    "robodojo/robodojo-arx-x5": ("robodojo", {"env_cfg_type": "arx_x5"}),
    "behavior/behavior-r1pro": ("behavior", {}),
}
DEFAULT_BACKENDS = ("libero", "robocasa", "robotwin", "robodojo")
# A probe journal holds the reference read-only operations (the status query `roboenv
# replay` makes on completion is one of them) and the verifier's own submit.
PROBE_OPS = READ_ONLY_OPS | {"submit"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_receipt(path, receipt):
    content = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    validate_public_text(content, "receipt.json")
    Path(path).write_text(content)


def _verify_prepared_bundle(root):
    saved = json_loads((root / "bundle.json").read_text())
    for name, expected in saved["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("prepared validation artifact escapes its case directory")
        if digest(path) != expected:
            raise ValueError("prepared validation case changed; generate a new output")


def _probe_inventory(catalog, task):
    root = Path(catalog) / "robots" / task.robot
    plan = BindingPlan.load(root, backend=task.backend, devices=task.device_bindings)
    readable = []
    for spec in plan.devices:
        profile = spec.resolved_profile(root)
        actions = profile.get("access", {}).get("driver", {}).get("actions", [])
        if profile["type"] != "world" or "read" in actions:
            readable.append(spec.name)
    return {"expected_devices": [s.name for s in plan.devices], "expected_reads": readable}


def _probe_program(names, readable):
    """A replay program using only the public ``env`` handle: reads, never writes."""
    return (
        "# Project-authored read-only device connectivity probe; not a benchmark solution.\n"
        f"expected = {sorted(names)!r}\n"
        f"readable = {sorted(readable)!r}\n"
        "profiles = {profile['name']: profile for profile in env.devices()}\n"
        "assert sorted(profiles) == expected, sorted(profiles)\n"
        "for name in expected:\n"
        "    read = profiles[name]['actions'].get('read', {})\n"
        "    if name not in readable:\n"
        "        assert profiles[name]['type'] == 'world', name\n"
        "        assert not read.get('supported', False), name\n"
        "        continue\n"
        "    assert read['supported'] is True and read['read_only'] is True, name\n"
        "    assert isinstance(env.device(name).read(), dict), name\n"
    )


def check_reference_provenance(provenance):
    """The catalog's reference policy: typed flags, and an official reference names its source.

    A reference marked ``official: true`` needs a public HTTPS ``upstream_url`` without
    credentials and a ``revision`` pinned to a 40- or 64-digit hexadecimal commit; the
    claim is otherwise unverifiable. Project-authored references carry ``official: false``.
    """
    if not isinstance(provenance, dict):
        raise ValueError("reference provenance must be an object")
    for flag in ("retired", "official"):
        if flag in provenance and type(provenance[flag]) is not bool:
            raise ValueError(f"reference {flag} status must be a boolean")
    if provenance.get("official") is True:
        upstream, revision = provenance.get("upstream_url"), provenance.get("revision")
        if not isinstance(upstream, str) or not isinstance(revision, str):
            raise ValueError("an official reference needs an upstream URL and revision")
        source = urlsplit(upstream)
        if (
            source.scheme != "https"
            or not source.hostname
            or source.username is not None
            or source.password is not None
            or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision)
        ):
            raise ValueError(
                "an official reference needs a public HTTPS upstream URL and a pinned revision"
            )


def _reference(task, source_path, cases):
    """The task's checked-in reference program, or None when there is nothing to replay."""
    reference = source_path.with_name("solution.py")
    if not reference.exists() and not cases:
        return None
    if not reference.is_file() or reference.is_symlink():
        raise ValueError("reference validation requires an explicit regular solution.py")
    note = reference.with_name("solution-source.json")
    provenance = json_loads(note.read_text()) if note.exists() else {}
    check_reference_provenance(provenance)
    if provenance.get("retired") is True:
        if cases:
            raise ValueError("selected reference is retired; use a validated typed controller")
        return None
    if provenance.get("official") is True:
        raise ValueError("this validation lane accepts project-authored references only")
    return reference.read_text()


def select_cases(
    *,
    project=ROOT,
    catalog=None,
    source=None,
    backends=DEFAULT_BACKENDS,
    cases=None,
    robots=None,
    modes=("reference",),
):
    """Plan (mode, task, source path, program) tuples; nothing is launched."""
    catalog = Path(catalog) if catalog else Path(project) / "catalog"
    if not modes or set(modes) - {"reference", "probe"}:
        raise ValueError("select reference and/or probe validation")
    if not backends:
        raise ValueError("select at least one validation backend")
    if robots and set(robots) - set(ROBOTS):
        raise ValueError("unknown validation robot")
    tasks_root = catalog / "tasks"
    inventory = load_catalog(tasks_root) if tasks_root.is_dir() else {}
    explicit_ids = set()
    if source is not None:
        extra = load_catalog(source)
        if set(extra) & set(inventory):
            raise ValueError("additional task source must not replace built-in example IDs")
        inventory.update(extra)
        explicit_ids = set(extra)
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
            program = _reference(task, source_path, cases)
            if program is None:
                continue
            plans.append(
                (
                    "reference",
                    replace(task, id=task.backend + "/reference-" + task.id.split("/")[-1]),
                    source_path,
                    program,
                )
            )
    if "probe" in modes:
        rigs = [(robot, backend, overrides) for robot, (backend, overrides) in ROBOTS.items()]
        known = {backend for _, backend, _ in rigs}
        for task, _ in selected:
            # Backends without a maintained rig list probe each selected task's own robot.
            if task.backend not in known and (task.robot, task.backend, {}) not in rigs:
                rigs.append((task.robot, task.backend, {}))
        for robot, backend, overrides in rigs:
            if backend not in backends or (robots and robot not in robots):
                continue
            candidates = [(task, path) for task, path in selected if task.backend == backend]
            if backend == "libero":
                candidates = [(t, p) for t, p in candidates if t.robot == robot]
            # A supplied task source takes priority over built-in examples for
            # that backend; otherwise adding an example silently changes jobs.
            explicit = [(t, p) for t, p in candidates if t.id in explicit_ids]
            candidates = explicit or candidates
            if not candidates:
                continue
            task, source_path = next(
                ((task, path) for task, path in candidates if task.robot == robot), candidates[0]
            )
            if backend == "behavior" and not task.backend_kwargs.get("scene_model"):
                raise ValueError("BEHAVIOR validation needs an explicit scene_model in task source")
            task = replace(
                task,
                id=f"{task.backend}/probe-{robot.replace('/', '-')}-{task.id.split('/')[-1]}",
                robot=robot,
                device_bindings=task.device_bindings if task.robot == robot else None,
                backend_kwargs={**task.backend_kwargs, **overrides},
            )
            inventory_ = _probe_inventory(catalog, task)
            program = _probe_program(inventory_["expected_devices"], inventory_["expected_reads"])
            plans.append(("probe", task, source_path, program))
    if not plans:
        raise ValueError("no validation cases matched")
    return plans


def _pending(record):
    return {
        **record,
        "execution_attempted": False,
        "boot_verified": False,
        "program_completed": False,
        "world_stopped": None,
        "replay_exit_code": None,
        "verify_exit_code": None,
        "verifier_valid": False,
        "final_verifier_success": False,
        "device_read_probe_passed": False,
        "reference_replay_passed": False,
        "reward": None,
        "budget": None,
        "journal": None,
        "timing_s": {},
        "failure_category": None,
        "failed_device_reads": [],
    }


def prepare(
    out,
    *,
    project=ROOT,
    catalog=None,
    source=None,
    backends=DEFAULT_BACKENDS,
    cases=None,
    robots=None,
    modes=("reference",),
    seeds=(0,),
):
    """Write every selected case (task, program, provenance) and a prepared receipt."""
    project, out = Path(project).resolve(), Path(out).resolve()
    catalog = Path(catalog).resolve() if catalog else project / "catalog"
    if out.exists():
        raise ValueError("validation output must be new; old evidence is never overwritten")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("validation seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate validation seeds")
    plans = select_cases(
        project=project, catalog=catalog, source=source, backends=backends, cases=cases,
        robots=robots, modes=modes,
    )
    records = []
    for index, (mode, task, source_path, program) in enumerate(plans):
        for seed in seeds:
            spec = replace(task, seed=seed)
            slug = task.id.replace("/", "-")
            path = f"cases/{index:03d}-{slug}-s{seed}"
            bundle = Bundle(out / path)
            bundle.add_yaml("task.yaml", asdict(spec))
            bundle.add("program.py", program)
            bundle.add(
                "program-source.json",
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
            bundle.commit({"kind": "validation-case", "mode": mode, "seed": seed})
            records.append(
                _pending(
                    {
                        "id": task.id,
                        "path": path,
                        "mode": mode,
                        "seed": seed,
                        "backend": task.backend,
                        "backend_task": task.backend_task,
                        "robot": task.robot,
                        "execution_mode": spec.execution_mode,
                        "budget_mode": spec.budget_mode,
                        "source_task_sha256": digest(source_path),
                        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                        "bindings_sha256": hashlib.sha256(
                            binding_resource(
                                catalog / "robots" / task.robot, backend=task.backend
                            ).read_bytes()
                        ).hexdigest(),
                        **_probe_inventory(catalog, task),
                    }
                )
            )
    receipt = {
        "schema": SCHEMA,
        "phase": "prepared",
        "roboenv_version": roboenv.__version__,
        "model_calls": 0,
        "reference_official": False,
        "cases": records,
        "all_selected_checks_passed": False,
        "fresh_process_replay_certificate": False,
    }
    write_receipt(out / "receipt.json", receipt)
    return receipt


def _journal_evidence(journal, record):
    """Independent read evidence from the retained, strictly parsed journal."""
    read_devices, failed_read_devices, mutating = set(), set(), False
    for line in journal.read_bytes().splitlines():
        event = strict_json(line)
        request, response = event.get("request", {}), event.get("response", {})
        args = request.get("args", {})
        if event.get("phase") == "error" and request.get("op") == "devices.call":
            if args.get("action") == "read":
                name = args.get("device")
                failed_read_devices.add(
                    name if name in record["expected_devices"] else "unknown-device"
                )
            continue
        if event.get("phase") != "commit" or response.get("ok") is not True:
            continue
        if request.get("op") == "devices.call":
            result = response.get("result", {})
            if args.get("action") == "read" and result.get("read_only") is True:
                read_devices.add(args.get("device"))
            elif args.get("action") not in READ_ONLY_ACTIONS:
                mutating = True
        elif request.get("op") not in PROBE_OPS:
            mutating = True
    return read_devices, failed_read_devices, mutating


def _evidence(record, report, verdict):
    """Allowlisted verdict fields plus the probe/reference classification of one run."""
    evidence = {"failure_category": None}
    if not isinstance(report, dict):
        raise ValueError("verifier report must be an object")
    valid = report.get("valid") is True
    reward = json_loads((verdict / "reward.json").read_text()).get("reward")
    journal = verdict / "actions.jsonl"
    artifact = report.get("journal_artifact", {})
    evidence["verifier_valid"] = valid
    evidence["reward"] = reward if type(reward) in {int, float} and reward in {0, 1} else None
    if isinstance(report.get("error"), str):
        # The failure type only: the message may name local paths or identities.
        evidence["verifier_error_type"] = report["error"].split(":", 1)[0]
    if journal.is_file() and artifact.get("sha256") == digest(journal):
        evidence["journal"] = {
            "sha256": artifact["sha256"],
            "bytes": artifact.get("bytes"),
            "requests": report.get("journal", {}).get("requests"),
            "validated": artifact.get("validated") is True,
        }
    else:
        valid = False
    submit = report.get("submit", {}) if isinstance(report.get("submit"), dict) else {}
    if isinstance(submit.get("budget"), dict):
        budget = submit["budget"]
        evidence["budget"] = {key: budget.get(key) for key in ("used", "max", "unit")}
    # The certificate names the task the world ran under: the case ID and its seed.
    identity = (report.get("task"), report.get("seed")) == (record["id"], record["seed"])
    boot = bool(report.get("health", {}).get("worker", {}).get("state") == "ready")
    evidence["boot_verified"] = boot and identity
    final_success = bool(
        valid and identity and report.get("trial_success") is True and reward == 1.0
    )
    evidence["final_verifier_success"] = final_success
    if record["mode"] == "probe":
        if valid and identity:
            reads, failed, mutating = _journal_evidence(journal, record)
        else:
            reads, failed, mutating = set(), set(), False
        evidence["failed_device_reads"] = sorted(failed)
        passed = bool(
            evidence["boot_verified"]
            and record["program_completed"]
            and valid
            and reads == set(record["expected_reads"])
            and not mutating
        )
        evidence["device_read_probe_passed"] = passed
        if not passed:
            if not record["program_completed"]:
                evidence["failure_category"] = "probe-program-failed"
            elif failed:
                evidence["failure_category"] = "device-read-failed"
            elif not valid:
                evidence["failure_category"] = "invalid-verifier-evidence"
            elif mutating:
                evidence["failure_category"] = "probe-mutated-environment"
            else:
                evidence["failure_category"] = "incomplete-device-read-evidence"
    else:
        passed = bool(record["program_completed"] and evidence["boot_verified"] and final_success)
        evidence["reference_replay_passed"] = passed
        if not passed:
            if not record["program_completed"]:
                evidence["failure_category"] = "reference-program-failed"
            elif not valid:
                evidence["failure_category"] = "invalid-verifier-evidence"
            else:
                evidence["failure_category"] = "reference-not-successful"
    return evidence


def _clean_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("ROBOENV_")}


def _cli(arguments, log, timeout):
    """Run one `roboenv` command as a separate process; raw output goes to the private log."""
    completed = subprocess.run(
        [sys.executable, "-m", "roboenv.cli.main", *arguments],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log.write(f"$ roboenv {' '.join(arguments)}\n{completed.stdout}{completed.stderr}\n")
    log.flush()
    return completed


def _run_case(case, record, log, *, launch_options, program_timeout, verify_timeout):
    from roboenv.environment import launch

    record["execution_attempted"] = True
    timing = record["timing_s"]
    started = time.monotonic()
    try:
        run = launch(case / "task.yaml", run_dir=case / "runs", log_stream=log, **launch_options)
    except Exception as exc:
        log.write(f"launch failed: {type(exc).__name__}: {exc}\n")
        record["failure_category"] = "launch-failed"
        return
    timing["launch_s"] = round(time.monotonic() - started, 3)
    try:
        started = time.monotonic()
        replay = _cli(
            ["replay", str(case / "program.py"), "--endpoint", run.endpoint], log, program_timeout
        )
        timing["replay_s"] = round(time.monotonic() - started, 3)
        record["replay_exit_code"] = replay.returncode
        try:
            record["program_completed"] = (
                replay.returncode == 0 and json_loads(replay.stdout)["program_completed"] is True
            )
        except (ValueError, KeyError, TypeError):
            record["program_completed"] = False
        started = time.monotonic()
        verify = _cli(
            ["verify", "--run", str(run.path), "--out", str(case / "verdict")], log, verify_timeout
        )
        timing["verify_s"] = round(time.monotonic() - started, 3)
        record["verify_exit_code"] = verify.returncode
    except subprocess.TimeoutExpired as exc:
        log.write(f"stage timed out: {exc}\n")
        record["failure_category"] = "stage-timeout"
        return
    finally:
        try:
            run.close()
            record["world_stopped"] = json_loads(run.path.read_text()).get("state") == "stopped"
        except Exception as exc:
            log.write(f"stop failed: {type(exc).__name__}: {exc}\n")
            record["world_stopped"] = False
    try:
        report = json_loads((case / "verdict/submit.json").read_text())
        record.update(_evidence(record, report, case / "verdict"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.write(f"evidence rejected: {type(exc).__name__}: {exc}\n")
        record["failure_category"] = "malformed-evidence"


def run_validation(
    out,
    *,
    project=ROOT,
    catalog=None,
    interpreter=None,
    runtime_root=None,
    python_path=(),
    launch_timeout=900,
    program_timeout=1800,
    verify_timeout=300,
):
    """Execute every prepared case in turn: launch, replay, verify, stop; update the receipt."""
    out = Path(out).resolve()
    receipt = json_loads((out / "receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("phase") != "prepared":
        raise ValueError("execution requires a fresh prepared validation receipt")
    for record in receipt["cases"]:
        case = out / record["path"]
        _verify_prepared_bundle(case)
        if digest(case / "program.py") != record["program_sha256"]:
            raise ValueError("validation program differs from its prepared receipt")
    if runtime_root is not None and (catalog is None or interpreter is None):
        raise ValueError("packaged startup requires an explicit catalog and interpreter")
    launch_options = {
        "project": None if runtime_root else project,
        "runtime_root": runtime_root,
        "catalog": catalog,
        "interpreter": interpreter,
        "timeout": launch_timeout,
        "python_path": list(python_path),
    }
    receipt["phase"] = "running"
    write_receipt(out / "receipt.json", receipt)
    private = out / "validation.private.log"
    with private.open("a") as log:
        private.chmod(0o600)
        for record in receipt["cases"]:
            log.write(f"== case {record['path']}\n")
            _run_case(
                out / record["path"], record, log, launch_options=launch_options,
                program_timeout=program_timeout, verify_timeout=verify_timeout,
            )
            write_receipt(out / "receipt.json", receipt)
    receipt.update(
        phase="completed",
        all_selected_checks_passed=all(
            record["world_stopped"] is True
            and (
                record["device_read_probe_passed"]
                if record["mode"] == "probe"
                else record["reference_replay_passed"]
            )
            for record in receipt["cases"]
        ),
    )
    write_receipt(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument(
        "--catalog", type=Path, help="Robot/task catalog (default: PROJECT/catalog)"
    )
    parser.add_argument("--source", type=Path, help="Additional explicit normalized tasks")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--robots", nargs="+", choices=sorted(ROBOTS))
    parser.add_argument("--modes", nargs="+", choices=("probe", "reference"), default=["reference"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--run", action="store_true", help="Launch the selected worlds; may use a GPU"
    )
    parser.add_argument(
        "--interpreter", type=Path, help="Prepared simulator interpreter for one backend"
    )
    parser.add_argument(
        "--runtime-root", type=Path, help="Packaged startup root; needs --catalog and --interpreter"
    )
    parser.add_argument(
        "--python-path", type=Path, action="append", default=[],
        help="Explicit SDK/runtime import path for the launched world",
    )
    parser.add_argument("--timeout", type=float, default=900, help="World startup deadline")
    args = parser.parse_args()
    try:
        if args.interpreter is not None and len(args.backends) != 1:
            raise ValueError("an explicit interpreter selects exactly one backend")
        report = prepare(
            args.out,
            project=args.project,
            catalog=args.catalog,
            source=args.source,
            backends=args.backends,
            cases=args.cases,
            robots=args.robots,
            modes=args.modes,
            seeds=args.seeds,
        )
        if args.run:
            report = run_validation(
                args.out,
                project=args.project,
                catalog=args.catalog,
                interpreter=args.interpreter,
                runtime_root=args.runtime_root,
                python_path=args.python_path,
                launch_timeout=args.timeout,
            )
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
