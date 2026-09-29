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
import sys
import tempfile
import uuid
from pathlib import Path

from contract_dependencies import prepare_contract_dependencies
from harbor.models.job.config import JobConfig
from roboenv.harbor.export import export_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True, help="Existing SDK-only image ID or tag")
    parser.add_argument("--sdk-wheel", required=True, type=Path)
    parser.add_argument("--runtime-wheel", required=True, type=Path)
    parser.add_argument(
        "--agent-bundle", type=Path, help="Use the real no-model Kinex Harbor agent"
    )
    parser.add_argument("--out", type=Path, help="Fresh artifact directory")
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    out = (
        args.out.resolve()
        if args.out
        else Path(tempfile.mkdtemp(prefix="device-harbor-", dir=cache))
    )
    if args.out:
        out.mkdir(parents=True, exist_ok=False)
    base = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.base_image], text=True
    ).strip()
    image_tag = "roboenv-device-smoke:" + uuid.uuid4().hex[:12]
    base_tag = "roboenv-device-base:" + uuid.uuid4().hex[:12]
    subprocess.run(["docker", "tag", base, base_tag], check=True)
    context = out / "image"
    context.mkdir()
    shutil.copy2(args.runtime_wheel, context / "runtime.whl")
    shutil.copy2(args.sdk_wheel, context / "sdk.whl")
    prepare_contract_dependencies(sys.executable, context / "contract-deps")
    (context / "Dockerfile").write_text(
        f"FROM {base_tag}\n"
        "COPY runtime.whl /opt/roboenv-runtime.whl\n"
        "COPY sdk.whl /opt/roboenv-sdk.whl\n"
        "COPY contract-deps/ /opt/roboenv-contract-deps/\n"
        "RUN python3 -m venv --system-site-packages /opt/sim-env "
        "&& mkdir -p /opt/roboenv/runtime\n"
        "ENV PYTHONPATH=/opt/roboenv-runtime.whl:/opt/roboenv-sdk.whl:"
        "/opt/roboenv-contract-deps\n"
        "RUN /opt/sim-env/bin/python -c 'import roboenv_runtime.runtime.worker'\n"
    )
    print(f"Artifacts: {out}", flush=True)
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
                "environment": {
                    "PYTHONPATH": "/opt/roboenv-runtime.whl:/opt/roboenv-contract-deps"
                },
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
    solve = task / "solution/solve.sh"
    solve.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "roboenv describe\nroboenv devices --name controller\n"
        "roboenv device controller write '{\"target\":[0.3]}' --request-id smoke-push\n"
        "roboenv operation smoke-push\nroboenv device controller read '{}'\n"
    )
    # Uploads keep this mode; the main container's root has no capability to add it.
    solve.chmod(0o755)
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
    if args.agent_bundle:
        tool = out / "tool"
        tool.mkdir()
        # Kinex commands only actuated device types. The probe reads the synthetic
        # "other" puck through normal Kinex dispatch and moves it with the main
        # image's RoboEnv SDK, which the tool runs as its interpreter.
        (tool / "tool.yaml").write_text(
            "name: harbor_probe\nkind: controller\n"
            "description: Exercise the synthetic device through normal Kinex dispatch.\n"
            "timeout_s: 30\ndevices: [other]\ninterpreter: /opt/roboenv-sdk/bin/python\n"
            "parameters: {type: object, properties: {}, additionalProperties: false}\n"
        )
        (tool / "tool.py").write_text(
            "import os\nimport roboenv\nfrom kinex import device, finish\n"
            "before = device('controller')['data']\n"
            "with roboenv.connect(os.environ['ROBOENV_ENDPOINT']) as env:\n"
            "    env.device('controller').write(target=[0.3])\n"
            "after = device('controller')['data']\n"
            "finish({'completed': True, 'before': before, 'after': after})\n"
        )
        agents = [
            {
                "import_path": "roboenv.harbor.kinex:KinexAgent",
                "kwargs": {
                    "bundle": str(args.agent_bundle.resolve()),
                    "mode": "no-model",
                    "tool": "harbor_probe",
                    "tool_directory": str(tool),
                    "require_completion": True,
                },
            }
        ]
        config = JobConfig.model_validate({**config.model_dump(), "agents": agents})
    job.write_text(config.model_dump_json(indent=2))
    with (out / "harbor.log").open("w") as log:
        result = subprocess.run(
            [shutil.which("harbor") or str(ROOT / ".venv/bin/harbor"), "run", "-c", str(job)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900 if args.agent_bundle else 360,
        )
    rewards = [json.loads(path.read_text()) for path in (out / "jobs").rglob("reward.json")]
    passed = result.returncode == 0 and rewards == [{"reward": 1.0}]
    report = {
        "passed": passed,
        "harbor_exit_code": result.returncode,
        "rewards": rewards,
        "artifacts": str(out),
        "image": image_id,
        "image_tag": image_tag,
        "model_calls": 0,
        "physical_replay_certificate": False,
    }
    if args.agent_bundle:
        executions = [
            json.loads(path.read_text()) for path in (out / "jobs").rglob("execution.json")
        ]
        integrity = [
            json.loads(path.read_text()) for path in (out / "jobs").rglob("integrity.json")
        ]
        report["kinex_execution"] = executions
        report["integrity"] = integrity
        passed = passed and len(executions) == len(integrity) == 1
        passed = passed and all(item.get("passed") is True for item in executions + integrity)
        report["passed"] = passed
    (out / "smoke-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
