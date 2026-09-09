"""Export a catalog task without importing or packaging a particular agent."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path

from roboenv.devices.bindings.spec import BINDING_CONTRACT
from roboenv.devices.contract import DEVICE_CONTRACT
from roboenv.privacy import validate_public_text
from roboenv.tasks.artifacts import Bundle, sha
from roboenv.validation import json_loads, yaml_load


def registry_name(identifier, seed):
    slug = re.sub(r"[^a-z0-9-]+", "-", identifier.lower()).strip("-")[:80]
    return f"roboenv/{slug}-s{seed}-{sha(identifier.encode())[:8]}"


def export_task(
    task_path,
    catalog,
    destination,
    wheel,
    *,
    seed=None,
    deployment_path=None,
    practice=False,
    include_reference=False,
    portable=True,
    prepare_only=False,
):
    """Create a standard Harbor task using a pinned, standalone simulator image."""
    if deployment_path is None:
        raise ValueError("a native deployment is required")
    task_path, catalog = Path(task_path).resolve(), Path(catalog).resolve()
    destination, wheel = (Path(value).resolve() for value in (destination, wheel))
    if destination == catalog or catalog in destination.parents:
        raise ValueError("build outputs must be outside the source catalog")
    from roboenv.tasks.resolve import resolve_task

    resolved = resolve_task(task_path, catalog=catalog)
    task = resolved.task
    if practice:
        task = replace(task, budget_mode="practice")
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed override must be a nonnegative integer")
        task = replace(task, seed=seed)
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("--wheel must be a built roboenv wheel")
    robot_dir = resolved.robot
    if robot_dir.is_symlink() or not robot_dir.resolve().is_relative_to(catalog):
        raise ValueError("robot definition must remain inside the source catalog")
    robot = yaml_load((robot_dir / "workspace.yaml").read_text())
    plan = resolved.bindings
    out = Bundle(destination)
    assets = files("roboenv").joinpath("assets")
    for source in sorted(robot_dir.rglob("*")):
        if source.is_symlink():
            raise ValueError("robot definition must not contain linked source files")
        if source.is_file() and source.suffix in (".py", ".yaml", ".md", ".json"):
            relative = source.relative_to(robot_dir)
            content = source.read_bytes()
            out.add("environment/sim/robot/" + str(relative), content)
            if relative.parts[0] == "devices" or relative.name == "bindings.yaml":
                continue  # binding implementation/config remains server-owned
            out.add(
                "environment/definition/robot/" + str(relative),
                content,
            )
    if plan:
        out.add_yaml("environment/sim/robot/bindings.yaml", plan.as_dict())
        for spec in plan.devices:
            profile = spec.resolved_profile(robot_dir)
            access = profile.setdefault("access", {})
            driver = access.setdefault("driver", {})
            actions = driver.get("actions", ["read"])
            driver.update(
                entry="driver.py", interpreter="python3", persistent=True, actions=actions
            )
            prefix = f"environment/definition/robot/devices/{spec.name}/"
            if task.backend not in {"robocasa", "isaaclab", "libero"}:
                out.add_yaml(prefix + "profile.yaml", profile)
                out.add(prefix + "driver.py", assets.joinpath("device-client.py").read_bytes())
                out.add(prefix + "_roboenv.py", assets.joinpath("_roboenv.py").read_bytes())
        out.add(
            "environment/definition/robot/lib/README.md",
            "Device implementations run on the server; clients use the public SDK.\n",
        )
    names = (
        [spec.name for spec in plan.devices]
        if plan
        else sorted(path.parent.name for path in (robot_dir / "devices").glob("*/profile.yaml"))
    )
    primary = next((name for name in ("arm", "left_arm", "controller") if name in names), None)
    manifest = {
        "schema_version": "roboenv/environment/1",
        "task": asdict(task),
        "robot": {"id": robot["identity"]["id"], "directory": "/env/robot"},
        "endpoint": {
            "protocol": "roboenv/1",
            "socket": "/run/roboenv/sim.sock",
            "session": "harbor",
        },
        "capabilities": ["device-driver/1", "roboenv-public-primitives/1"],
        "observation_access": "sensors-only"
        if task.backend in {"robocasa", "isaaclab", "libero"}
        else "benchmark-ground-truth",
        "action_abstraction": task.action_abstraction,
        "compatibility": {
            "backend_runtime": "roboenv-runtime",
            "dispatch_device": primary,
        },
    }
    manifest["capabilities"].append(DEVICE_CONTRACT)
    manifest["device_contract"] = DEVICE_CONTRACT
    manifest["endpoint"]["generation"] = 0
    manifest["binding_contract"] = BINDING_CONTRACT
    manifest["binding_mode"] = "explicit"
    if task.backend in {"robocasa", "isaaclab", "libero"}:
        out.add("environment/sim/environment.json", json.dumps(manifest, indent=2) + "\n")
        manifest = {
            "task": {"id": task.id, "instruction": task.instruction, "context": {}},
            "devices": names,
            "device_contract": DEVICE_CONTRACT,
        }
    out.add("environment/definition/environment.json", json.dumps(manifest, indent=2) + "\n")
    out.add("environment/sdk/" + wheel.name, wheel.read_bytes())
    dockerfile = assets.joinpath("main.Dockerfile").read_text()
    if task.backend in {"robocasa", "isaaclab", "libero"}:
        dockerfile += (
            "\nRUN useradd --create-home --uid 1000 agent "
            "&& chown -R agent:agent /app\nUSER agent\n"
        )
    out.add("environment/Dockerfile", dockerfile)
    out.add("environment/.dockerignore", "logs/\n**/__pycache__/\n")
    out.add_yaml(
        f"environment/sim/tasks/{task.backend_task}.yaml",
        {**resolved.runtime_config(), "seeds": [task.seed], "budget": {"mode": task.budget_mode}},
    )
    out.add(
        "environment/sim/serve-native.sh", assets.joinpath("serve-native.sh").read_text(), 0o755
    )
    out.add("environment/sim/healthcheck.py", assets.joinpath("healthcheck.py").read_text())
    public_env = {
        "ROBOENV_SOCKET": "/run/roboenv/sim.sock",
        "ROBOENV_SESSION": "harbor",
        "ROBOENV_RUN_DIR": "/run/roboenv/run",
    }
    from roboenv.harbor.deployment import native_service

    sim = native_service(deployment_path, task, portable=portable)
    sim["volumes"] += ["./definition:/env:ro", "./sdk:/opt/roboenv/sdk:ro"]
    sim["environment"].update(
        ROBOENV_SERVER_ROBOT="/opt/serve/robot",
        ROBOENV_SERVER_MANIFEST="/env/environment.json",
        ROBOENV_SDK_PATH="/opt/roboenv/sdk/" + wheel.name,
    )
    main = {
        "pull_policy": "build",
        "depends_on": {"sim": {"condition": "service_healthy"}},
        "volumes": ["roboenv-run:/run/roboenv:ro", "./definition:/env:ro"],
        "environment": public_env,
    }
    if task.backend in {"robocasa", "isaaclab", "libero"}:
        sim["user"] = "0:0"
        public_env.update(ROBOENV_ENDPOINT="http://sim:8123", ROBOENV_ROBOT="/run/roboenv/robot")
        sim["environment"].update(
            ROBOENV_FRESH_SESSION="1",
            ROBOENV_ENDPOINT="http://sim:8123",
            ROBOENV_HTTP_HOST="0.0.0.0",
            ROBOENV_HTTP_PORT="8123",
            ROBOENV_PUBLISH_DEVICES="/run/roboenv/robot/devices",
            ROBOENV_SERVER_MANIFEST="/opt/serve/environment.json",
        )
        main.update(cap_drop=["ALL"], security_opt=["no-new-privileges:true"])
    compose = {"services": {"main": main, "sim": sim}, "volumes": {"roboenv-run": {}}}
    out.add_yaml("environment/docker-compose.yaml", compose)
    smoke = json.loads(json.dumps(compose))
    smoke["services"]["main"].update(build={"context": "."}, command=["sleep", "infinity"])
    smoke["services"]["main"]["volumes"] += [
        "../tests:/tests:ro",
        "./logs:/logs",
    ]
    if include_reference and task_path.with_name("solution.py").is_file():
        smoke["services"]["main"]["volumes"].append("../solution:/solution:ro")
    out.add_yaml("environment/smoke-compose.yaml", smoke)
    observation_access = (
        "sensors-only"
        if task.backend in {"robocasa", "isaaclab", "libero"}
        else "benchmark-ground-truth"
    )
    out.add(
        "task.toml",
        f'''schema_version = "1.4"
[task]
name = "{registry_name(task.id, task.seed)}"
version = "0.1.0"
description = {json.dumps(task.instruction, ensure_ascii=False)}
[metadata]
backend = "{task.backend}"
backend_task = "{task.backend_task}"
seed = {task.seed}
environment_contract = "roboenv/environment/1"
observation_access = "{observation_access}"
action_abstraction = "{task.action_abstraction}"
[agent]
timeout_sec = 3600.0
{'user = "agent"' if task.backend in {"robocasa", "isaaclab", "libero"} else ""}
[verifier]
timeout_sec = 600.0
{'user = "root"' if task.backend in {"robocasa", "isaaclab", "libero"} else ""}
[environment]
build_timeout_sec = 1800.0
cpus = 4
memory_mb = 8192
storage_mb = 20480
gpus = 0
''',
    )
    instruction = (
        task.instruction
        + """

Work in /app. The environment definition is /env/environment.json; each device
has /env/robot/devices/NAME/profile.yaml and driver.py. Use `roboenv describe`,
`roboenv devices`, `roboenv scene`, and `roboenv api` to discover the environment.
`roboenv device NAME ACTION 'JSON'` uses a device; `roboenv call NAME 'JSON'` calls
a documented public primitive. All commands are also available via the roboenv
Python SDK. Poses and action units are documented by the device/primitive.

The existing native Cartesian primitives are available. Generic joint-space IK/FK
and unknown calibration are not claimed. Your deliverable is the live world state.
The independent verifier submits after you finish and checks full reward,
completed status, unexhausted official budget and the supervisor action journal.
Do not reset, restore, checkpoint, submit, shut down or access simulator internals.
Save candidate code under /app. The environment definition and evidence are read-only.
"""
    )
    instruction += """
The environment server is already running; you do not need to install or initialize
a simulator, or implement environment drivers. Server device discovery is authoritative:
`roboenv devices --name NAME` describes supported actions, parameters, and limitations.
Device responses contain `data`, a budget snapshot, and execution metadata. A successful
device call is not a benchmark verdict. For commands whose delivery becomes uncertain,
inspect `roboenv operation REQUEST_ID`; never blindly resend a motion under a new id.
"""
    if task.budget_mode == "practice":
        instruction = instruction.replace("unexhausted official budget", "the authoring budget")
        instruction = instruction.replace(
            "Do not reset, restore, checkpoint, submit, shut down or access simulator internals.",
            (
                "This is an AUTHORING-ONLY trial: supported reset/checkpoint/restore "
                "controls are allowed. "
                "Do not submit, shut down or access simulator internals. "
                "Results are not benchmark eligible."
            ),
        )
    if task.backend in {"robocasa", "isaaclab", "libero"}:
        clock_instruction = (
            "Physics continues while you think. Writes update persistent commands; "
            "stop stops a device, not the world. "
            if task.backend_kwargs.get("realtime", task.backend in {"robocasa", "libero"})
            else "Physics pauses between actions. Each write advances one native control "
            "interval and then pauses again; reads and stop do not advance time. "
        )
        instruction = (
            task.instruction
            + "\n\n"
            + (
                "The simulation service is already running. In Python, use "
                "`env = roboenv.connect(os.environ['ROBOENV_ENDPOINT'])`, then inspect "
                "`env.task`, `env.devices()`, and `env.device(name).profile`. "
                "Read sensors and command devices through their declared basic interfaces. "
                + clock_instruction
                + (
                    "There is no simulation horizon. "
                    if task.backend in {"robocasa", "libero"}
                    and task.backend_kwargs.get("max_steps") is None
                    else "The owner enforces the task control-interval budget. "
                )
                + "Reads and IK/FK do not request extra steps. "
                "Camera reads return client-accessible image files. Live profiles and thin drivers "
                "are in /run/roboenv/robot/devices. Build perception and control tools "
                "in /app. The verifier checks final state and budget after you finish.\n"
            )
        )
    out.add("instruction.md", instruction)
    out.add("environment/workspace/AGENTS.md", "# Environment workspace\n\n" + instruction)
    reference = task_path.with_name("solution.py")
    if include_reference and reference.is_file():
        if reference.is_symlink():
            raise ValueError("reference program must not be a linked source file")
        out.add("solution/solution.py", reference.read_text())
        out.add(
            "solution/solve.sh", "#!/bin/sh\nset -eu\nroboenv replay /solution/solution.py\n", 0o755
        )
        source_note = reference.with_name("solution-source.json")
        if source_note.is_symlink():
            raise ValueError("reference provenance must not be a linked source file")
        provenance = (
            json_loads(source_note.read_text())
            if source_note.is_file()
            else {
                "origin": "project-reference",
                "official": False,
                "replay_verified": False,
            }
        )
        if not isinstance(provenance, dict):
            raise ValueError("reference provenance must be an object")
        if "official" in provenance and type(provenance["official"]) is not bool:
            raise ValueError("reference official status must be a boolean")
        if provenance.get("official") is True:
            from urllib.parse import urlsplit

            upstream, revision = provenance.get("upstream_url"), provenance.get("revision")
            if not isinstance(upstream, str) or not isinstance(revision, str):
                raise ValueError("An official reference needs an upstream URL and revision")
            source_url = urlsplit(upstream)
            if (
                source_url.scheme != "https"
                or not source_url.hostname
                or source_url.username is not None
                or source_url.password is not None
                or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision)
            ):
                raise ValueError(
                    "An official reference needs a public HTTPS upstream URL and pinned revision"
                )
        provenance.setdefault("origin", "project-reference")
        provenance.setdefault("official", False)
        # Attribution is not a certificate for a newly generated task.
        provenance["replay_verified"] = False
        provenance["program_sha256"] = sha(reference.read_bytes())
        out.add("solution/source.json", json.dumps(provenance, indent=2) + "\n")
    out.add("tests/verify.py", assets.joinpath("verify.py").read_text())
    out.add(
        "tests/task.json",
        json.dumps(
            {
                "task_id": task.backend_task,
                "backend": task.backend,
                "seed": task.seed,
                "session": "harbor",
                "budget_mode": task.budget_mode,
            }
        )
        + "\n",
    )
    out.add(
        "tests/test.sh",
        "#!/bin/sh\nset -eu\n/usr/bin/python3 -I /tests/verify.py\n",
        0o755,
    )
    metadata = {
        "schema_version": 1,
        "task": asdict(task),
        "task_sha256": sha(task_path.read_bytes()),
        "sdk_sha256": sha(wheel.read_bytes()),
        "deployment": "native-image",
        "deployment_manifest": Path(deployment_path).name if deployment_path else None,
        "deployment_sha256": sha(Path(deployment_path).read_bytes()) if deployment_path else None,
        "certified": False,
    }
    validate_public_text(json.dumps(metadata), "bundle.json")
    if prepare_only:
        out.preflight()
        return out, metadata
    result = out.commit(metadata)
    return {"output": str(destination), "files": len(result["files"]), "task": task.id}
