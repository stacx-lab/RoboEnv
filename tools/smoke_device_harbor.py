"""Run a CPU-only native Harbor device trial using an existing SDK-only base image.

No model calls, downloads, GPU, or source-tree mounts. The explicitly supplied
base must contain Python, venv, pip, numpy and PyYAML (the normal main image does).
Only newly created smoke artifacts/images are written; prior trials are untouched.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from harbor.models.job.config import JobConfig
from roboenv.harbor.export import export_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True, help="Existing SDK-only image ID or tag")
    parser.add_argument("--sdk-wheel", required=True, type=Path)
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="device-harbor-", dir=cache))
    base = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.base_image], text=True
    ).strip()
    image_tag = "roboenv-device-smoke:" + uuid.uuid4().hex[:12]
    base_tag = "roboenv-device-base:" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "tag", base, base_tag], check=True)
    context = out / "image"
    context.mkdir()
    shutil.copy2(args.runtime_wheel, context / "runtime.whl")
    (context / "Dockerfile").write_text(
        f"FROM {base_tag}\n"
        "COPY runtime.whl /opt/roboenv-runtime.whl\n"
        "RUN python3 -m venv --system-site-packages /opt/sim-env "
        "&& mkdir -p /opt/roboenv/runtime\n"
        "ENV PYTHONPATH=/opt/roboenv-runtime.whl\n"
        "RUN /opt/sim-env/bin/python -c 'import roboenv_runtime.runtime.worker'\n"
    )
    print(f"Artifacts: {out.relative_to(ROOT)}", flush=True)
    with (out / "image-build.log").open("w") as log:
        subprocess.run(
            ["docker", "build", "--network=none", "--pull=false", "-t", image_tag, str(context)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=180,
        )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image_tag], text=True
    ).strip()
    deployment = out / "deployment.json"
    deployment.write_text(
        json.dumps(
            {
                "schema": "roboenv/deployment/1",
                "backend": "fakeshell",
                "legacy_tree_required": False,
                "image": image_id,
                "environment": {"PYTHONPATH": "/opt/roboenv-runtime.whl"},
            },
            indent=2,
        )
    )
    task = out / "task"
    export_task(
        ROOT / "tests/fixtures/native-toy/task.yaml",
        ROOT / "tests/fixtures/native-toy/catalog",
        task,
        args.sdk_wheel,
        deployment_path=deployment,
    )
    # The production main image recipe is unchanged. This smoke reuses its
    # already-installed OS dependencies and upgrades only the SDK, offline.
    (task / "environment/Dockerfile").write_text(
        f"FROM {base_tag}\n"
        "COPY sdk/ /tmp/device-sdk/\n"
        "RUN /opt/roboenv-sdk/bin/pip install --no-deps --force-reinstall /tmp/device-sdk/*.whl\n"
        'COPY workspace/ /app/\nWORKDIR /app\nCMD ["sleep", "infinity"]\n'
    )
    (task / "solution").mkdir(exist_ok=True)
    (task / "solution/solve.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "roboenv describe\nroboenv devices --name controller\n"
        "roboenv device controller invoke "
        '\'{"tool_name":"push","kwargs":{"dx":0.3}}\' --request-id smoke-push\n'
        "roboenv operation smoke-push\nroboenv device controller read '{}'\n"
    )
    (task / "solution/source.json").write_text(
        json.dumps(
            {
                "origin": "synthetic-test-fixture",
                "official": False,
                "description": "Project-authored CPU connectivity oracle; not a benchmark expert",
            },
            indent=2,
        )
        + "\n"
    )
    manifest_path = task / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("environment/Dockerfile", "solution/solve.sh"):
        manifest["files"][name] = hashlib.sha256((task / name).read_bytes()).hexdigest()
    manifest["smoke"] = {"oracle": "device-cli", "base_image": base}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config = JobConfig(
        job_name="device-contract-smoke",
        jobs_dir=out / "jobs",
        n_attempts=1,
        agents=[{"name": "oracle"}],
        tasks=[{"path": task}],
        retry={"max_retries": 0},
    )
    job = out / "job.json"
    job.write_text(config.model_dump_json(indent=2))
    with (out / "harbor.log").open("w") as log:
        result = subprocess.run(
            [shutil.which("harbor") or str(ROOT / ".venv/bin/harbor"), "run", "-c", str(job)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=360,
        )
    rewards = [json.loads(path.read_text()) for path in (out / "jobs").rglob("reward.json")]
    passed = result.returncode == 0 and rewards == [{"reward": 1.0}]
    report = {
        "passed": passed,
        "harbor_exit_code": result.returncode,
        "rewards": rewards,
        "artifacts": str(out.relative_to(ROOT)),
        "image": image_id,
        "image_tag": image_tag,
        "model_calls": 0,
        "physical_replay_certificate": False,
    }
    (out / "smoke-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
