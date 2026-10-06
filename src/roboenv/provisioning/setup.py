"""Audit or strictly install a simulator into an explicitly new environment.

Run from the repository's development environment. The low-level installer never repairs an
existing environment, suppresses dependency resolution, fetches assets, accepts
licenses, or claims a GPU replay. Preparation separately acquires public backend assets.
Rejected installs preserve a private failure log.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from roboenv.project import project_root
from roboenv.provisioning.progress import stage
from roboenv.provisioning.sources import status
from roboenv.validation import json_loads

GIT_PIN = re.compile(
    r"(?P<repository>git\+https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"@(?P<revision>[0-9a-f]{40})"
)


def audit_lock(path):
    """Reject ambiguous duplicates, unresolved VCS inputs, and non-pinned lines."""
    issues, names = [], {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text in {
            "--extra-index-url https://download.pytorch.org/whl/cu128",
            "--extra-index-url https://pypi.nvidia.com",
        }:
            continue
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            issues.append({"line": number, "reason": "invalid-requirement"})
            continue
        name = canonicalize_name(requirement.name)
        if name in names:
            issues.append({"line": number, "package": name, "reason": "duplicate-package"})
        names[name] = requirement
        if requirement.url:
            if not GIT_PIN.fullmatch(requirement.url):
                issues.append({"line": number, "package": name, "reason": "unresolved-source"})
        else:
            pins = list(requirement.specifier)
            if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
                issues.append({"line": number, "package": name, "reason": "not-exactly-pinned"})
    return {"passed": not issues, "packages": len(names), "issues": issues}


def _requirements(path):
    result = {}
    for line in path.read_text().splitlines():
        text = line.strip()
        if text and not text.startswith(("#", "--")):
            item = Requirement(text)
            result[canonicalize_name(item.name)] = item
    return result


def audit_source_pins(lock, path):
    """Only unify a floating Git URL with the identical source pinned in the lock."""
    if path is None:
        return {"passed": True, "pins": {}, "issues": []}
    issues, pins = [], {}
    if not path.is_file():
        return {"passed": False, "pins": {}, "issues": ["missing-source-pins"]}
    checked = audit_lock(path)
    if not checked["passed"]:
        return {"passed": False, "pins": {}, "issues": checked["issues"]}
    locked = _requirements(lock)
    for name, item in _requirements(path).items():
        selected = GIT_PIN.fullmatch(item.url or "")
        original = locked.get(name)
        if (
            selected is None
            or item.extras
            or item.marker is not None
            or original is None
            or original.extras
            or original.marker is not None
            or original.url != item.url
        ):
            issues.append({"package": name, "reason": "only-identical-locked-git-source-allowed"})
            continue
        pins[name] = selected.groupdict()
    if not pins and not issues:
        issues.append({"reason": "empty-source-pins"})
    return {"passed": not issues, "pins": pins, "issues": issues}


def _project_path(project, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or "\\" in relative
    ):
        raise ValueError("recipe paths must remain relative to the selected project")
    path = (project / relative).resolve()
    if not path.is_relative_to(project):
        raise ValueError("recipe path escapes the selected project")
    return path


def audit(backend, project=None):
    project = project_root(project)
    recipes = json_loads((project / "runtime/environments.json").read_text())
    if (
        not isinstance(recipes, dict)
        or recipes.get("schema") != "roboenv/environments/1"
        or not isinstance(recipes.get("backends"), dict)
    ):
        raise ValueError("unsupported environment recipe schema")
    if backend not in recipes["backends"]:
        raise ValueError("unknown simulator backend")
    recipe = recipes["backends"][backend]
    source_status = status(project, [backend])
    lock = _project_path(project, recipe["lock"]) if recipe["lock"] else None
    lock_status = (
        audit_lock(lock)
        if lock and lock.is_file()
        else {"passed": False, "issues": ["missing-lock"]}
    )
    pins_path = _project_path(project, recipe["source_pins"]) if recipe.get("source_pins") else None
    source_pins = (
        audit_source_pins(lock, pins_path)
        if lock_status["passed"]
        else {"passed": False, "pins": {}, "issues": ["invalid-dependency-lock"]}
    )
    blockers = list(recipe["known_blockers"])
    if not source_status["ready"]:
        blockers.append("source-check-failed: run roboenv sources status for this backend")
    if not lock_status["passed"]:
        blockers.append("dependency-lock-audit-failed")
    if not source_pins["passed"]:
        blockers.append("source-pin-audit-failed")
    editables, source_only = recipe["editables"], recipe.get("source_only", [])
    if (
        not isinstance(editables, list)
        or not isinstance(source_only, list)
        or any(not isinstance(path, str) for path in editables + source_only)
    ):
        raise ValueError("recipe editable and source-only closures must be arrays")
    if set(editables) & set(source_only):
        raise ValueError("a checkout cannot be both editable and source-only")
    if any(not _project_path(project, path).is_dir() for path in editables + source_only):
        blockers.append("missing-local-package-checkout")
    recorded = recipe.get("recorded_validation", {})
    if not isinstance(recorded, dict):
        raise ValueError("recorded validation must be an object")
    if recorded and not _project_path(project, recorded["evidence"]).is_file():
        blockers.append("missing-recorded-validation-evidence")
    probes = recipe.get("import_probes", [])
    if not isinstance(probes, list) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name)
        for name in probes
    ):
        raise ValueError("import probes must be module names")
    return {
        "schema": "roboenv/setup-audit/1",
        "backend": backend,
        "python": recipe["python"],
        "lock": recipe["lock"],
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock and lock.is_file()
        else None,
        "lock_audit": lock_status,
        "source_pins": recipe.get("source_pins"),
        "source_pins_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest()
        if pins_path and pins_path.is_file()
        else None,
        "source_pins_audit": source_pins,
        "source_only": source_only,
        "sources_ready": source_status["ready"],
        "blockers": blockers,
        "install_preflight_passed": not blockers,
        "clean_install_verified": False,
        "gpu_replay_verified": False,
        "recorded_recipe_validation": recorded,
        "asset_instructions": recipe["asset_instructions"],
    }, recipe


def _component(value, project, name):
    """Select exactly the requested source tree or wheel, never a parent fallback."""
    path = (
        Path(value).resolve()
        if value is not None
        else (project if name == "roboenv" else project / "runtime")
    )
    if path.is_file() and path.suffix == ".whl":
        distribution, _, _, _ = parse_wheel_filename(path.name)
        if canonicalize_name(distribution) != name:
            raise ValueError(f"the selected wheel is not the {name} distribution")
        return {
            "arguments": [str(path)],
            "import_root": None,
            "receipt": {
                "distribution": name,
                "kind": "wheel",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        }
    configuration = path / "pyproject.toml"
    if not path.is_dir() or not configuration.is_file():
        raise ValueError(f"{name} input must be an explicit package source directory or wheel")
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    metadata = tomllib.loads(configuration.read_text()).get("project", {})
    if canonicalize_name(metadata.get("name", "")) != name:
        raise ValueError(f"the selected source is not the {name} distribution")
    return {
        "arguments": ["-e", str(path) + ("[server]" if name == "roboenv" else "")],
        "import_root": str(path),
        "receipt": {
            "distribution": name,
            "kind": "source",
            "configuration_sha256": hashlib.sha256(configuration.read_bytes()).hexdigest(),
        },
    }


LIBERO_LOGGING_CONFIG = """
import importlib.util
from pathlib import Path
import sys

# Locate the wheel without importing it: robosuite initializes logging on import.
spec = importlib.util.find_spec("robosuite")
if spec is None or spec.origin is None:
    raise RuntimeError("LIBERO requires an installed robosuite package")
package = Path(spec.origin).resolve().parent
if not package.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("Refusing to configure robosuite outside the new environment")
with (package / "macros_private.py").open("x") as config:
    config.write(
        "# RoboEnv captures console logs; avoid robosuite 1.4's shared /tmp log.\\n"
        "import robosuite.macros as macros\\n"
        "macros.FILE_LOGGING_LEVEL = None\\n"
    )
"""


PROBE = """
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
from packaging.requirements import Requirement

settings = json.loads(sys.argv[1])
normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
for name, pin in settings["source_pins"].items():
    selected = metadata.distribution(name)
    provenance = json.loads(selected.read_text("direct_url.json") or "{}")
    repository = pin["repository"].removeprefix("git+")
    if provenance.get("url") != repository:
        raise RuntimeError("installed source repository differs from the locked source")
    if provenance.get("vcs_info", {}).get("commit_id") != pin["revision"]:
        raise RuntimeError("installed source revision differs from the locked source")
    for distribution in metadata.distributions():
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if normalize(requirement.name) != name:
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            if requirement.url:
                expected = requirement.url.removeprefix("git+")
                if expected not in (repository, repository + "@" + pin["revision"]):
                    raise RuntimeError("source override would replace a different required source")
            elif not requirement.specifier.contains(selected.version, prereleases=True):
                raise RuntimeError("source pin violates a numeric dependency constraint")
for module_name, expected_root in settings["components"].items():
    module = importlib.import_module(module_name)
    root = Path(expected_root or sys.prefix).resolve()
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("component import came from an unselected package location")
import roboenv_runtime.runtime.worker
if settings["backend"] == "libero":
    from roboenv_runtime.adapters.libero import ensure_libero_config
    ensure_libero_config()
for module_name in settings["import_probes"]:
    importlib.import_module(module_name)
print("Selected packages, source identities and declared import probes passed; no GPU episode ran.")
"""


def install(backend, prefix, *, project=None, sdk=None, runtime=None, wheels=None, progress=None):
    project = project_root(project)
    with stage(progress, "Validating installation prerequisites"):
        report, recipe = audit(backend, project)
        if not report["install_preflight_passed"]:
            raise ValueError(
                f"installation preflight failed; run 'roboenv doctor {backend}' for blockers"
            )
    if not shutil.which("uv"):
        raise ValueError("uv is required")
    project = Path(project).resolve()
    prefix = Path(prefix)
    if prefix.exists() or prefix.is_symlink():
        raise ValueError("refusing to modify an existing environment; select a new prefix")
    prefix = prefix.resolve()
    if wheels is not None:
        if sdk is not None or runtime is not None:
            raise ValueError("--wheels cannot be combined with individual package inputs")
        wheels = Path(wheels).resolve()
        selected = []
        for name in ("roboenv", "roboenv_runtime"):
            matches = list(wheels.glob(f"{name}-[0-9]*.whl"))
            if len(matches) != 1:
                raise ValueError(f"--wheels needs exactly one {name} wheel")
            selected.append(matches[0])
        sdk, runtime = selected
    components = {
        "roboenv": _component(sdk, project, "roboenv"),
        "roboenv_runtime": _component(runtime, project, "roboenv-runtime"),
    }
    lock = _project_path(project, recipe["lock"])
    source_pins = (
        _project_path(project, recipe["source_pins"]) if recipe.get("source_pins") else None
    )
    pin_audit = audit_source_pins(lock, source_pins)
    if not pin_audit["passed"]:
        raise ValueError("source-pin audit failed before installation")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    logs = project / ".cache" / "setup"
    logs.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=backend + "-", dir=logs))
    logfile = private / "install.private.log"
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
    }
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_NO_DEPS",
        "PIP_NO_DEPS",
        "UV_OVERRIDE",
        "UV_CONSTRAINT",
        "UV_BUILD_CONSTRAINT",
        "PIP_CONSTRAINT",
        "PIP_TARGET",
        "PIP_PREFIX",
    ):
        environment.pop(name, None)
    environment.update(
        ROBOENV_ROOT=str(project),
        ROBOTWIN_ROOT=str(project / "third_party/RoboTwin"),
        ROBODOJO_ROOT=str(project / "third_party/RoboDojo"),
        LIBERO_CONFIG_PATH=str(private / "config/libero"),
        XDG_CONFIG_HOME=str(private / "config"),
        XDG_CACHE_HOME=str(private / "cache"),
        TMPDIR=str(private),
        PYTHONDONTWRITEBYTECODE="1",
    )
    python = str(prefix / "bin/python")
    indexes = [
        argument for url in recipe["extra_indexes"] for argument in ("--extra-index-url", url)
    ]
    if recipe.get("index_url"):
        indexes = ["--index-url", recipe["index_url"], *indexes]
    if recipe.get("index_strategy"):
        indexes += ["--index-strategy", recipe["index_strategy"]]
    overrides = ["--override", str(source_pins)] if source_pins else []
    local = [
        argument
        for path in recipe["editables"]
        for argument in ("-e", str(_project_path(project, path)))
    ]
    with logfile.open("w") as log:
        logfile.chmod(0o600)
        if progress:
            progress(f"Detailed installation log: {logfile.relative_to(project).as_posix()}")

        def run(label, command, *, env=None):
            log.write(f"\n--- {label} ---\n")
            log.flush()
            try:
                with stage(progress, label):
                    subprocess.run(
                        command,
                        cwd=project,
                        env=environment if env is None else env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=1800,
                    )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(
                    f"installation failed; inspect {logfile.relative_to(project).as_posix()}; "
                    "the new partial environment was preserved, not certified"
                ) from error

        run(
            "Creating isolated Python environment",
            ["uv", "venv", "--python", recipe["python"], str(prefix)],
        )
        run(
            "Installing dependencies and RoboEnv (downloads may be large)",
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "-r",
                str(lock),
                *indexes,
                *overrides,
                *components["roboenv"]["arguments"],
                *components["roboenv_runtime"]["arguments"],
                *local,
            ],
        )
        run("Checking dependency consistency", ["uv", "pip", "check", "--python", python])
        if backend == "libero":
            run(
                "Configuring LIBERO console logging",
                [python, "-I", "-c", LIBERO_LOGGING_CONFIG],
            )
        probe_settings = {
            "backend": backend,
            "project": str(project),
            "components": {name: spec["import_root"] for name, spec in components.items()},
            "source_pins": pin_audit["pins"],
            "source_only": recipe.get("source_only", []),
            "import_probes": recipe.get("import_probes", []),
        }
        run(
            "Verifying imports and pinned package origins",
            [python, "-I", "-c", PROBE, json.dumps(probe_settings)],
            env={**environment, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
        )
    report["clean_install_verified"] = True
    report["environment_assets_verified"] = False
    report["package_inputs"] = [spec["receipt"] for spec in components.values()]
    report["log"] = logfile.relative_to(project).as_posix()
    registration = project / ".cache/roboenv/installations" / f"{backend}.json"
    registration.parent.mkdir(parents=True, exist_ok=True)
    record = {"backend": backend, "python": python, "report": report}
    temporary = registration.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(registration)
    if progress:
        progress(
            "Interpreter registered; installation checks passed. "
            "Assets and GPU readiness are separate."
        )
    return report


def prepare(backend, *, project=None, install_to=None, progress=None, renderer=None):
    """Prepare a source-backed interpreter and the backend's supported asset depot."""
    if renderer is not None and (backend != "abc" or renderer not in {"mujoco", "blender"}):
        raise ValueError("Renderer preparation supports abc with mujoco or blender")
    root = project_root(project)
    # Validate the backend before using it in a filesystem path or syncing sources.
    with stage(progress, "Checking simulator configuration and existing sources"):
        audit(backend, root)
    record = None
    if install_to is None:
        registration = root / ".cache/roboenv/installations" / f"{backend}.json"
        if registration.is_file():
            candidate = json_loads(registration.read_text())
            if Path(candidate["python"]).is_file():
                record = candidate
        destination = root / ".venv-sim" / backend
    else:
        destination = Path(install_to)
    if record is None and (destination.exists() or destination.is_symlink()):
        raise ValueError(
            f"Environment already exists at {destination}, but no reusable interpreter "
            "was selected; existing environments are preserved. "
            f"Inspect .cache/setup/{backend}-*/install.private.log for earlier failures, "
            f"then run 'roboenv prepare {backend} --prefix NEW_DIRECTORY'."
        )
    from roboenv.provisioning.sources import sync

    with stage(progress, "Refreshing sources: discarding source edits and reapplying patches"):
        sync(root, [backend], reset=True)
    if record is not None:
        if progress:
            progress(
                "Sources refreshed; reusing registered interpreter without dependency reinstall."
            )
        # The old receipt describes its original pins and dependency lock. Re-audit
        # current inputs rather than relabeling that historical installation proof.
        report, _ = audit(backend, root)
        report.update(sources_ready=True, clean_install_verified=False)
    else:
        report = install(backend, destination, project=root, progress=progress)
    report = _prepare_assets(backend, root, report, progress)
    if renderer == "blender":
        from roboenv.provisioning.abc_blender import prepare as prepare_blender

        python = record["python"] if record is not None else str(destination / "bin/python")
        probe = subprocess.run(
            [python, "-c", "from pxr import Usd; from PIL import Image"], capture_output=True
        )
        if probe.returncode:
            raise ValueError(
                "ABC interpreter lacks Blender Python dependencies; existing environments "
                "are preserved. Run roboenv prepare abc --renderer blender --prefix NEW_DIRECTORY"
            )
        report["rendering"] = prepare_blender(root, progress=progress)
    return report


def _prepare_assets(backend, root, report, progress):
    if backend == "abc":
        from roboenv.provisioning.abc_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "simple":
        from roboenv.provisioning.simple_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "behavior":
        from roboenv.provisioning.behavior_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "robodojo":
        from roboenv.provisioning.robodojo_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "robotwin":
        from roboenv.provisioning.robotwin_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "libero":
        from roboenv.provisioning.libero_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    if backend == "robocasa":
        from roboenv.provisioning.robocasa_assets import prepare_assets

        return {**report, **prepare_assets(root, progress=progress)}
    return report


def doctor(backend, *, project=None):
    root = project_root(project)
    report, _ = audit(backend, root)
    registration = root / ".cache/roboenv/installations" / f"{backend}.json"
    installed = json_loads(registration.read_text()) if registration.is_file() else None
    report["installed"] = bool(installed and Path(installed["python"]).is_file())
    report["next_steps"] = []
    if not report["install_preflight_passed"]:
        report["next_steps"].append(f"roboenv sources sync --backends {backend}")
        report["next_steps"].append("Resolve the reported recipe/asset prerequisites")
    if not report["installed"]:
        report["next_steps"].append(f"roboenv prepare {backend}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "backend",
        choices=(
            "abc", "libero", "robocasa", "robotwin", "robodojo", "behavior", "isaaclab", "simple"
        ),
    )
    parser.add_argument(
        "--install", action="store_true", help="May download many GB; assets excluded"
    )
    parser.add_argument(
        "--prefix", type=Path, help="Required new environment location for --install"
    )
    parser.add_argument("--project", type=Path, help="Explicit Roboenv source checkout")
    parser.add_argument("--sdk", type=Path, help="SDK source directory or wheel; default: project")
    parser.add_argument(
        "--runtime", type=Path, help="Runtime source directory or wheel; default: project/runtime"
    )
    parser.add_argument(
        "--wheels", type=Path, help="Directory holding exactly one SDK/runtime wheel"
    )
    args = parser.parse_args()
    if args.install and args.prefix is None:
        parser.error("--install requires an explicit --prefix")
    if not args.install and any(
        value is not None for value in (args.sdk, args.runtime, args.wheels)
    ):
        parser.error("package input options require --install")
    if args.wheels is not None and (args.sdk is not None or args.runtime is not None):
        parser.error("--wheels cannot be combined with --sdk or --runtime")
    try:
        report = (
            install(
                args.backend,
                args.prefix,
                project=args.project,
                sdk=args.sdk,
                runtime=args.runtime,
                wheels=args.wheels,
            )
            if args.install
            else audit(args.backend, args.project)[0]
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    except OSError:
        parser.exit(1, "Cannot access a setup input or destination; check paths and permissions.\n")
    print(json.dumps(report, indent=2))
    return 0 if report["install_preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
