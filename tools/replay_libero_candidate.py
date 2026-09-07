"""Fresh-process LIBERO candidate validation over the actual public Unix-socket API.

Run using the candidate interpreter with an explicitly selected GPU. This is a
local worker acceptance probe, not a Docker/Harbor or multi-seed certification.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from packaging.requirements import Requirement
from roboenv.client import Client
from roboenv.privacy import validate_public_text
from roboenv_runtime.runtime.protocol import socket_address


class OwnerClient(Client):
    """Owner-only lifecycle access; never inserted into the saved program globals."""

    allowed_ops = Client.allowed_ops | {"submit", "shutdown"}


def environment(root, out, phase, gpu):
    local = out / phase
    local.mkdir()
    result = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "ROBOENV_SERVER_MANIFEST",
        "ROBOT_DESCRIPTION_COMMIT",
        "SSH_AUTH_SOCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        result.pop(name, None)
    for name in list(result):
        if name.endswith(("_API_KEY", "_TOKEN")):
            result.pop(name)
    result.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_EGL_DEVICE_ID": str(gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "JAX_PLATFORMS": "cpu",
            "ROBOENV_DETERMINISTIC_IK": "1",
            "ROBOENV_ROOT": str(root),
            "CAPX_ROOT": str(root / "third_party/cap-x"),
            "LIBERO_CONFIG_PATH": str(local / "libero-config"),
            "ROBOT_DESCRIPTIONS_CACHE": str(out / "cache/robots"),
            "XDG_CACHE_HOME": str(out / "cache"),
            "MPLCONFIGDIR": str(out / "cache/matplotlib"),
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "ROBOENV_SERVER_ROBOT": str(root / "catalog/robots/libero-panda"),
        }
    )
    return result


def validate_dependencies(root):
    count = 0
    for line in (root / "runtime/locks/libero.candidate.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        distribution = importlib.metadata.distribution(requirement.name)
        if requirement.url:
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            if direct.get("vcs_info", {}).get("commit_id") != requirement.url.rsplit("@", 1)[1]:
                raise ValueError("candidate VCS source identity differs from the lock")
        elif distribution.version not in requirement.specifier:
            raise ValueError("installed candidate dependency differs from the lock")
        count += 1
    return {"locked_requirements_verified": count}


def validate_origins(root):
    prefix = Path(sys.prefix).resolve()
    result = {}
    for name in ("roboenv", "roboenv_runtime", "libero", "robosuite"):
        source = Path(importlib.util.find_spec(name).origin).resolve()
        if name in {"roboenv", "roboenv_runtime"}:
            if not source.is_relative_to(prefix):
                raise ValueError("SDK/runtime must be installed in the candidate environment")
            result[name] = {"origin": "candidate-environment"}
        else:
            expected = root / "third_party/cap-x/capx/third_party"
            if not source.is_relative_to(expected):
                raise ValueError("simulator module does not come from the pinned checkout")
            result[name] = {"origin": source.relative_to(root).as_posix()}
        result[name]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        importlib.metadata.version("capx")
    except importlib.metadata.PackageNotFoundError:
        result["capx"] = {"origin": "locked-source-only", "installed_distribution": False}
    else:
        raise ValueError("candidate must not silently install the full cap-x distribution")
    return result


def run_phase(root, out, gpu, phase):
    env = environment(root, out, phase, gpu)
    local = out / phase
    socket = local / "worker.sock"
    log_path = local / "worker.private.log"
    task = root / "catalog/tasks/libero/drawer-and-bowl/task.yaml"
    reference = task.with_name("solution.py")
    command = [
        sys.executable,
        "-I",
        "-m",
        "roboenv_runtime.runtime.worker",
        "--socket",
        str(socket),
        "--run-dir",
        str(local / "run"),
        "--task",
        str(task),
        "--session",
        "libero-candidate",
        "--seed",
        "0",
        "--budget-mode",
        "official",
    ]
    started = time.monotonic()
    with log_path.open("w") as log:
        log_path.chmod(0o600)
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        client = Client(socket_address(str(socket)), session="libero-candidate")
        owner = OwnerClient(socket_address(str(socket)), session="libero-candidate")
        try:
            deadline = time.monotonic() + 300
            while True:
                if process.poll() is not None:
                    raise RuntimeError("worker exited during boot; inspect private worker log")
                try:
                    health = client.request("health", timeout=2)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("worker boot timed out") from None
            if health["seed"] != 0 or health["budget_mode"] != "official":
                raise RuntimeError("worker task conditions differ from the reference")
            initial = client.request("observe")["budget"]
            inventory = client.devices()["devices"]
            names = sorted(item["name"] for item in inventory)
            if names != ["arm", "camera", "gripper", "world"]:
                raise RuntimeError("expected LIBERO device set is incomplete")
            record = {
                "phase": phase,
                "fresh_worker": True,
                "boot_passed": True,
                "devices": names,
                "initial_budget": initial,
            }
            if phase == "device-probe":
                for item in inventory:
                    if item["actions"].get("read", {}).get("supported"):
                        client.device(item["name"]).read()
                client.device("gripper").call("move", {"target": [1.0]})
                record["device_reads_and_motion_passed"] = True
                record["final_budget"] = client.request("observe")["budget"]
            else:
                namespace = {"__name__": "__main__"}
                for name in client.api()["tools"]:
                    namespace[name] = lambda *args, _name=name, **kwargs: client.call(
                        _name, *args, **kwargs
                    )
                with (local / "program.private.log").open("w") as program_log:
                    import contextlib

                    (local / "program.private.log").chmod(0o600)
                    with (
                        contextlib.redirect_stdout(program_log),
                        contextlib.redirect_stderr(program_log),
                    ):
                        exec(compile(reference.read_text(), "solution.py", "exec"), namespace)
                final = owner.request("submit")
                observed = client.request("observe")
                record["verdict"] = final
                record["post_submit_budget"] = observed["budget"]
                record["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
                record["reference_replay_passed"] = (
                    final.get("completed") is True
                    and final.get("reward") == 1.0
                    and final["budget"]["used"] < final["budget"]["max"]
                    and final["budget"] == observed["budget"]
                )
                if not record["reference_replay_passed"]:
                    raise RuntimeError("reference replay did not satisfy the official verdict")
            record["duration_seconds"] = round(time.monotonic() - started, 2)
            return record
        finally:
            if process.poll() is None:
                try:
                    owner.request("shutdown", timeout=5)
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical device index")
    root, out = args.project.resolve(), args.out.resolve()
    if out.exists():
        parser.error("--out must select a new directory")
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {
        "schema": "roboenv/libero-candidate-replay/1",
        "scope": "local isolated-interpreter worker and public SDK; not Docker/Harbor",
        "task": "libero.90_8",
        "seed": 0,
        "init_order": "capx",
        "budget_mode": "official",
        "model_calls": 0,
        "phases": [],
        "passed": False,
    }
    try:
        report["dependency_lock"] = validate_dependencies(root)
        report["module_sources"] = validate_origins(root)
        for phase in ("device-probe", "reference-replay"):
            report["phases"].append(run_phase(root, out, args.gpu, phase))
        report["passed"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        try:
            validate_public_text(str(error), "")
            report["error"] = str(error)
        except ValueError:
            report["error"] = "Candidate validation failed; inspect private logs."
    rendered = json.dumps(report, indent=2) + "\n"
    validate_public_text(rendered, "receipt.json")
    (out / "receipt.json").write_text(rendered)
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
