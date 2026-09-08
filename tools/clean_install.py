"""Install wheels in a new environment and probe them outside the source tree.

This is a CPU packaging check, never simulator or GPU replay evidence. A private
log is retained on failure; console output and the JSON receipt contain no host
locations. Dependencies are resolved normally rather than bypassed with --no-deps.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def verify(wheels, *, harbor=True, offline=False, out=None):
    wheels = Path(wheels).resolve()
    if not shutil.which("uv"):
        raise ValueError("uv is required for clean-install verification")
    selected = []
    for distribution in ("roboenv", "roboenv_runtime"):
        matches = list(wheels.glob(f"{distribution}-[0-9]*.whl"))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {distribution} wheel")
        selected.append(matches[0])
    project = Path(__file__).resolve().parents[1]
    scratch = project / ".cache" / "clean-install"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    environment["PYTHONNOUSERSITE"] = "1"
    logfile = scratch / "last-run.private.log"
    result = {
        "schema": "roboenv/clean-install/1",
        "wheels": [
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in selected
        ],
        "checks": [],
        "gpu_replay_verified": False,
    }
    with tempfile.TemporaryDirectory(prefix="run-", dir=scratch) as temporary:
        directory = Path(temporary)
        venv = directory / "venv"
        python = str(venv / "bin/python")
        with logfile.open("w") as log:
            logfile.chmod(0o600)

            def run(label, command):
                try:
                    subprocess.run(
                        command,
                        cwd=directory,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    raise RuntimeError(
                        f"{label} failed; inspect .cache/clean-install/last-run.private.log"
                    ) from error
                result["checks"].append(label)

            flags = ["--offline"] if offline else []
            run("fresh-python-3.12", ["uv", *flags, "venv", "--python", "3.12", str(venv)])
            run(
                "resolved-wheel-install",
                [
                    "uv",
                    *flags,
                    "pip",
                    "install",
                    "--python",
                    python,
                    "--find-links",
                    str(wheels),
                    *map(str, selected),
                    *(["harbor==0.22.0"] if harbor else []),
                ],
            )
            run("dependency-consistency", ["uv", "pip", "check", "--python", python])
            code = (
                "import importlib.metadata as m, pathlib, sys; "
                "import roboenv, roboenv_runtime.runtime.worker as worker; "
                "root = pathlib.Path(sys.prefix).resolve(); "
                "assert pathlib.Path(roboenv.__file__).resolve().is_relative_to(root); "
                "assert pathlib.Path(worker.__file__).resolve().is_relative_to(root); "
                "assert not {'torch', 'mujoco', 'roborl', 'complex'} & sys.modules.keys(); "
                "assert any(item.replace(' ', '').startswith('roboenv[server]==') "
                "for item in m.requires('roboenv-runtime')); "
                "assert m.version('roboenv') == m.version('roboenv-runtime')"
            )
            run("isolated-wheel-imports", [python, "-I", "-c", code])
            run("sdk-cli", [str(venv / "bin/roboenv"), "--help"])
            run("worker-cli", [str(venv / "bin/roboenv-worker"), "--help"])
            run("supervisor-cli", [str(venv / "bin/roboenv-supervise"), "--help"])
            if harbor:
                run("harbor-cli", [str(venv / "bin/harbor"), "--help"])
                run(
                    "harbor-task-schema",
                    [
                        python,
                        "-I",
                        "-c",
                        "from harbor.models.task.config import TaskConfig; "
                        "assert TaskConfig().schema_version == '1.4'",
                    ],
                )
    result["passed"] = True
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path, default=Path("build/wheels"))
    parser.add_argument("--without-harbor", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = verify(
            args.wheels, harbor=not args.without_harbor, offline=args.offline, out=args.out
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
