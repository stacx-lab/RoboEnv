"""Resolve and export a task using prepared deployment artifacts."""

import subprocess
import sys
import tempfile
from pathlib import Path

from roboenv.project import project_root
from roboenv.tasks.resolve import resolve_task


def export(reference, *, project=None, catalog=None, out, deployment=None, wheel=None):
    from roboenv.harbor.deployment import load_deployment
    from roboenv.harbor.export import export_task
    from roboenv.harbor.taskset import validate_harbor_task

    root = project_root(project)
    resolved = resolve_task(reference, project=root, catalog=catalog)
    deployment = deployment or root / ".cache/images" / resolved.task.backend / "deployment.json"
    if not Path(deployment).is_file():
        raise ValueError(
            "No prepared image deployment; supply --deployment deployment.json. "
            "See docs/harbor-tasks.md for image preparation."
        )
    load_deployment(deployment, resolved.task.backend)
    if wheel is None:
        staging = root / ".cache/roboenv/wheels"
        staging.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(dir=staging))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--outdir",
                str(destination),
                str(root),
            ],
            check=True,
            stdout=sys.stderr,
        )
        (wheel,) = destination.glob("roboenv-*.whl")
    result = export_task(
        resolved.path,
        resolved.catalog,
        out,
        wheel,
        deployment_path=deployment,
    )
    validate_harbor_task(out)
    return result
