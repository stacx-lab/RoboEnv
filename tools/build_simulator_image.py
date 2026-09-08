#!/usr/bin/env python3
"""Build independent native simulator images from explicitly staged dependency slices.

No whole original working-tree mount, agent code, .env, or model credentials enter
the image. Large asset directories may be staged into a separately movable depot.
Existing source/environments are read only; output must be a fresh directory.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

from roboenv.provisioning.sources import status

WORKSPACE = Path(__file__).resolve().parents[1]
INTERPRETERS = {
    "isaaclab": ".venv-sim/isaaclab",
    "libero": "third_party/cap-x/.venv-libero",
    "robotwin": "third_party/cap-x/.venv-robotwin",
    "robocasa": "miniconda3/envs/robocasa",
    "robodojo": "miniconda3/envs/RoboDojo",
    "behavior": ".cache/miniconda3/envs/behavior",
}
SOURCES = {
    "isaaclab": ["third_party/IsaacLab"],
    "libero": [
        "third_party/cap-x",
        "third_party/cap-x/capx/third_party/LIBERO-PRO",
        "third_party/cap-x/capx/third_party/libero_dependencies/robosuite",
    ],
    "robotwin": ["third_party/cap-x", "third_party/RoboTwin"],
    "robocasa": ["third_party/robocasa", "third_party/robosuite"],
    "robodojo": [
        "third_party/cap-x",
        "third_party/RoboDojo",
        "third_party/RoboDojo/third_party/IsaacLab",
        "third_party/RoboDojo/third_party/curobo",
    ],
    "behavior": ["third_party/BEHAVIOR-1K"],
}
EXTERNAL = {
    "isaaclab": [".cache/isaaclab/assets"],
    "libero": [],
    "robotwin": ["third_party/RoboTwin/assets"],
    "robocasa": ["third_party/robocasa/robocasa/models/assets"],
    "robodojo": ["third_party/RoboDojo/Assets"],
    "behavior": ["third_party/BEHAVIOR-1K/datasets"],
}


def ignore_interpreter_state(directory, names):
    """Kit runtime documents are generated user state, not simulator dependencies."""
    ignored = set(
        shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".git",
            ".env",
            ".env.*",
            ".ssh",
            ".aws",
            ".azure",
            ".config/gcloud",
            ".netrc",
            ".pypirc",
            ".npmrc",
            ".git-credentials",
            ".docker",
        )(directory, names)
    )
    if Path(directory).parts[-3:] == ("isaacsim", "kit", "data"):
        ignored.add("documents")
    return ignored & set(names)


def interpreter_size(environment):
    total = 0
    for directory, children, names in os.walk(environment, followlinks=False):
        ignored = ignore_interpreter_state(directory, children + names)
        children[:] = [name for name in children if name not in ignored]
        for name in names:
            path = Path(directory) / name
            if name not in ignored and not path.is_symlink():
                total += path.stat().st_size
    return total


def bundle_kit_extensions(staged, prepared, dependency_root):
    """Materialize only Kit's already-installed extension links into the staging copy."""
    staged, prepared, dependency_root = map(Path, (staged, prepared, dependency_root))
    allowed = (dependency_root / "omni-home/.local/share/ov/data/exts").resolve()
    receipts = []
    copied = {}
    for link in sorted(staged.glob("lib/python*/site-packages/isaacsim/kit/data/Kit/*/*/exts/*/*")):
        if not link.is_symlink():
            continue
        relative = link.relative_to(staged)
        original = prepared / relative
        source = original.resolve(strict=True)
        if not source.is_relative_to(allowed) or not (source / "config/extension.toml").is_file():
            raise ValueError("Kit extension link is not in the declared prepared extension store")
        destination = staged / "_roboenv_extensions" / source.name
        if source not in copied:
            if destination.exists():
                raise ValueError("conflicting staged Kit extension identity")
            shutil.copytree(source, destination, symlinks=True, ignore=ignore_interpreter_state)
            copied[source] = destination
            receipts.append(
                {
                    "extension": source.name,
                    "manifest_sha256": hashlib.sha256(
                        (source / "config/extension.toml").read_bytes()
                    ).hexdigest(),
                    "staged_path": destination.relative_to(staged).as_posix(),
                }
            )
        link.unlink()
        link.symlink_to(os.path.relpath(destination, link.parent), target_is_directory=True)
    return receipts


def check_docker_storage(size):
    """Docker storage may be on a different filesystem from the staging directory."""
    try:
        root = subprocess.check_output(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        ).strip()
        free = shutil.disk_usage(root).free
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify local Docker storage; check Docker availability") from exc
    # Source snapshot, installed image layer and export can coexist temporarily.
    if free < 3 * size + 4 * 1024**3:
        raise RuntimeError("Insufficient Docker storage for staging, layers and export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=INTERPRETERS)
    parser.add_argument("--dependencies", required=True, type=Path)
    parser.add_argument(
        "--interpreter",
        type=Path,
        help="Explicit simulator bin/python; independent of the asset dependency root",
    )
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--asset-mode", choices=("reference", "copy"), default="reference")
    parser.add_argument("--interpreter-mode", choices=("image", "depot"), default="image")
    args = parser.parse_args()
    source_status = status(WORKSPACE, [args.backend])
    if not source_status["ready"]:
        parser.error("pinned upstream sources are not ready; run roboenv sources sync first")
    root, out = args.dependencies.resolve(), args.out.resolve()
    if out.exists():
        parser.error("output exists; use a fresh directory (never overwrite an image/depot)")
    interpreter = (
        args.interpreter.absolute()
        if args.interpreter is not None
        else root / INTERPRETERS[args.backend] / "bin/python"
    )
    if interpreter.parent.name != "bin":
        parser.error("interpreter must be the bin/python executable of an isolated environment")
    environment = interpreter.parent.parent
    if not interpreter.is_file() or not args.runtime_wheel.is_file():
        parser.error("interpreter and built runtime wheel are required")
    size = interpreter_size(environment)
    staging_parent = next(parent for parent in out.parents if parent.exists())
    if shutil.disk_usage(staging_parent).free < size + 4 * 1024**3:
        parser.error("insufficient free staging space; choose another output filesystem")
    if not args.stage_only:
        check_docker_storage(size if args.interpreter_mode == "image" else 0)
    probe = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import sys,json; print(json.dumps([sys.version_info[:2],sys.base_prefix]))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    version, base = json.loads(probe.stdout.strip())
    context = out / "context"
    context.mkdir(parents=True)
    replacements = {
        str(root): "/opt/roboenv/runtime",
        str(environment): "/opt/sim-env",
        str(root / "RoboDojo"): "/opt/roboenv/runtime/third_party/RoboDojo",
        str(root / "cap-x"): "/opt/roboenv/runtime/third_party/cap-x",
        str(
            root / "third_party/cap-x/capx/third_party/RoboTwin"
        ): "/opt/roboenv/runtime/third_party/RoboTwin",
        str(WORKSPACE): "/opt/roboenv/runtime",
    }
    # Fresh copies: never hardlink mutable environments/bridge source to the old checkout.
    shutil.copytree(
        environment,
        context / "sim-env",
        symlinks=True,
        ignore=ignore_interpreter_state,
    )
    (context / "sim-env/.roboenv-staged-environment.json").write_text(
        '{"schema":"roboenv/staged-environment/1"}\n'
    )
    extension_receipts = bundle_kit_extensions(context / "sim-env", environment, root)
    python_base = Path(base).resolve() != environment.resolve()
    if python_base:
        shutil.copytree(
            base,
            context / "python-base",
            symlinks=True,
            ignore=ignore_interpreter_state,
        )
        replacements[str(Path(base))] = "/opt/python"
        replacements[str(Path(base).resolve())] = "/opt/python"
        (context / "python-base/.roboenv-staged-environment.json").write_text(
            '{"schema":"roboenv/staged-environment/1"}\n'
        )
    else:
        (context / "python-base").mkdir()
    exclusions = {str(WORKSPACE / p) for paths in EXTERNAL.values() for p in paths}
    mounts = []
    for relative in EXTERNAL[args.backend]:
        source = root / relative
        if args.backend == "robotwin" and not source.exists():
            source = root / "third_party/cap-x/capx/third_party/RoboTwin/assets"
        if source.is_dir():
            destination = source
            if args.asset_mode == "copy":
                destination = out / "depot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True
                )
            mounts.append(
                {
                    "source": os.path.relpath(destination, out),
                    "target": "/opt/roboenv/runtime/" + relative,
                    "read_only": True,
                }
            )
        else:
            raise FileNotFoundError(f"required asset depot missing for {args.backend}")

    def ignore(directory, names):
        blocked = {".git", "__pycache__", "node_modules", ".cache", "XPolicyLab"}
        blocked |= {name for name in names if name.startswith((".venv", ".env"))}
        if Path(directory).name == "capx":
            blocked.add("third_party")
        if Path(directory) == WORKSPACE / "third_party/RoboDojo":
            blocked.add("third_party")
        blocked |= {name for name in names if str(Path(directory) / name) in exclusions}
        return blocked & set(names)

    for relative in SOURCES[args.backend]:
        source = WORKSPACE / relative
        if not source.is_dir():
            raise FileNotFoundError(f"required dependency missing: {source}")
        target = context / "runtime-data" / relative
        shutil.copytree(source, target, symlinks=True, ignore=ignore, dirs_exist_ok=True)
    descriptions = root / ".cache/robot_descriptions"
    if descriptions.is_dir():
        shutil.copytree(
            descriptions, context / "runtime-data/.cache/robot_descriptions", symlinks=True
        )
    for folder in (WORKSPACE / "runtime/data").iterdir():
        shutil.copytree(folder, context / "runtime-data" / folder.name, dirs_exist_ok=True)
    for name in ("Dockerfile", "relocate.py", "nvidia_icd.json", "10_nvidia.json"):
        shutil.copy2(WORKSPACE / "runtime/packaging" / name, context / name)
    (context / "wheels").mkdir()
    shutil.copy2(args.runtime_wheel, context / "wheels" / args.runtime_wheel.name)
    prepare_dependencies = runpy.run_path(str(WORKSPACE / "tools/contract_dependencies.py"))[
        "prepare_contract_dependencies"
    ]
    contract_receipt = prepare_dependencies(interpreter, context / "contract-deps")
    # Host locations are a build secret, not a Docker layer or public task artifact.
    relocation_secret = out / ".relocation.private.json"
    relocation_secret.write_text(
        json.dumps(
            {
                "replacements": replacements,
                "python_base": python_base,
                "python_version": ".".join(map(str, version)),
            },
            indent=2,
        )
    )
    relocation_secret.chmod(0o600)
    dependency_locks = {}
    if args.interpreter_mode == "depot":
        relocate = runpy.run_path(str(WORKSPACE / "runtime/packaging/relocate.py"))["relocate"]
        relocation = json.loads(relocation_secret.read_text())
        relocate(context / "sim-env", relocation)
        if python_base:
            relocate(context / "python-base", {**relocation, "python_base": False})
        snapshot = runpy.run_path(str(WORKSPACE / "runtime/packaging/verify_dependencies.py"))[
            "snapshot"
        ]
        (out / "depot").mkdir(exist_ok=True)
        (context / "dependency-locks").mkdir()
        depots = {}
        for name in ("sim-env", "python-base"):
            if name == "python-base" and not python_base:
                continue
            depot = out / "depot" / name
            (context / name).rename(depot)
            target = "/opt/roboenv/runtime/dependencies/" + name
            depots[target] = depot
        for target, depot in depots.items():
            name = depot.name
            content = (
                json.dumps(snapshot(depot, target, peers=depots), sort_keys=True, indent=2) + "\n"
            )
            (context / "dependency-locks" / (name + ".json")).write_text(content)
            dependency_locks[name] = hashlib.sha256(content.encode()).hexdigest()
            mounts.append({"source": "depot/" + name, "target": target, "read_only": True})
        shutil.copy2(WORKSPACE / "runtime/packaging/Dockerfile.depot", context / "Dockerfile")
        shutil.copy2(
            WORKSPACE / "runtime/packaging/verify_dependencies.py",
            context / "verify_dependencies.py",
        )
    metadata = {
        "schema": "roboenv/deployment/1",
        "backend": args.backend,
        "image": args.tag,
        "mounts": mounts,
        "runtime_wheel_sha256": hashlib.sha256(args.runtime_wheel.read_bytes()).hexdigest(),
        "interpreter": "/opt/sim-env/bin/python",
        "legacy_tree_required": False,
        "source_revisions": {item["name"]: item["actual"] for item in source_status["sources"]},
        "interpreter_mode": args.interpreter_mode,
        "executable_depot_manifests": dependency_locks,
        "contract_dependencies": contract_receipt,
        "bundled_kit_extensions": extension_receipts,
        "upstream_lock_sha256": hashlib.sha256(
            (WORKSPACE / "upstreams.lock.json").read_bytes()
        ).hexdigest(),
    }
    (out / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.stage_only:
        check_docker_storage(interpreter_size(context))
        subprocess.run(
            [
                "docker",
                "build",
                "--secret",
                f"id=relocation,src={relocation_secret}",
                "-t",
                args.tag,
                str(context),
            ],
            check=True,
        )
        metadata["image"] = subprocess.run(
            ["docker", "image", "inspect", args.tag, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    (out / "deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "deployment": os.path.relpath(out / "deployment.json", Path.cwd()),
                "built": not args.stage_only,
            }
        )
    )


if __name__ == "__main__":
    main()
