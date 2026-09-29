"""Optional Harbor agent bridge; the Kinex checkout and agent loop stay external."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import tarfile
from pathlib import Path, PurePosixPath

from harbor.agents.base import BaseAgent
from roboenv.validation import json_loads

# Pi providers that development accepts, and Kinex's thinking levels except off:
# development always states an explicit effort.
PROVIDERS = ("openai",)
EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def inspect_bundle(path):
    """Check the complete immutable input before uploading or executing anything."""
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or "bundle.json" not in names:
            raise ValueError("Invalid Kinex bundle inventory")
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError("Kinex bundles may contain only regular relative files")
        manifest = json_loads(archive.extractfile("bundle.json").read())
        if manifest.get("schema") != "roboenv/kinex-agent-bundle/1":
            raise ValueError("Unsupported Kinex bundle schema")
        hashes = {
            name: hashlib.sha256(archive.extractfile(name).read()).hexdigest()
            for name in names
            if name != "bundle.json"
        }
        identity = {
            "kinex_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        }
        if hashes != manifest["files"] or identity != manifest["identity"]:
            raise ValueError("Kinex bundle contents differ from the manifest")
        required = {
            "bin/node",
            "bin/uv",
            "kinex/dist/application.js",
            "kinex/node_modules/@kinex/core/dist/index.js",
            "kinex/integrations/roboenv/run.ts",
        }
        if not required <= hashes.keys():
            raise ValueError("Kinex bundle lacks its runtime entrypoints")
        return identity


class KinexAgent(BaseAgent):
    """Let Harbor own trials and verifiers; compose the normal Kinex APIs in main."""

    def __init__(
        self,
        logs_dir,
        model_name=None,
        *,
        bundle,
        mode="no-model",
        effort=None,
        recording=None,
        tool="integration_probe",
        args=None,
        tool_directory=None,
        artifact=None,
        require_completion=False,
        agent_timeout_sec=None,
        setup_timeout_sec=600,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if mode not in {"no-model", "develop"}:
            raise ValueError("Kinex mode must be no-model or develop")
        if mode == "no-model" and (model_name or effort):
            raise ValueError("No-model execution must not select a model or effort")
        provider, _, model = (model_name or "").partition("/")
        if mode == "develop" and (provider not in PROVIDERS or not model or effort not in EFFORTS):
            raise ValueError("Development requires an explicit openai/MODEL and effort")
        for timeout in (agent_timeout_sec or 300, setup_timeout_sec):
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Kinex timeouts must be positive finite seconds")
        if (
            not isinstance(tool, str)
            or not tool
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in tool
            )
        ):
            raise ValueError("Tool must be a simple tool identifier")
        if artifact and tool_directory:
            raise ValueError("Choose a frozen artifact or a development tool directory")
        self.bundle = Path(bundle).expanduser().resolve(strict=True)
        self.identity = inspect_bundle(self.bundle)
        self.bundle_sha256 = hashlib.sha256(self.bundle.read_bytes()).hexdigest()
        self.mode, self.effort = mode, effort
        self.recording, self.tool_args = recording or {}, args or {}
        self.tool, self.require_completion = tool, require_completion or bool(artifact)
        self.tool_directory = Path(tool_directory).resolve(strict=True) if tool_directory else None
        self.artifact = Path(artifact).resolve(strict=True) if artifact else None
        if self.artifact:
            selected = json_loads((self.artifact / "manifest.json").read_text())
            self.tool, self.tool_args = selected["tool"], selected["args"]
        self.timeout = agent_timeout_sec or 300
        self.setup_timeout = setup_timeout_sec
        self.root = "/app/.roboenv-agent"
        self.bridge = self.root + "/kinex/integrations/roboenv"
        self.workspace = "/app/workspace"

    @staticmethod
    def name():
        return "kinex"

    def version(self):
        return self.identity["kinex_sha256"][:16]

    @property
    def environment(self):
        return {
            "PATH": self.root + "/bin:/opt/roboenv-sdk/bin:/usr/local/bin:/usr/bin:/bin",
            "KINEX_CONFIG_DIR": self.root + "/config",
            "KINEX_DATA_DIR": self.root + "/data",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_LINK_MODE": "copy",
            "UV_LOCKED": "true" if self.artifact and self.mode == "no-model" else "false",
        }

    async def checked(self, environment, command, *, timeout):
        result = await environment.exec(
            command, cwd="/app", env=self.environment, timeout_sec=math.ceil(timeout)
        )
        if result.return_code:
            raise RuntimeError(
                f"Kinex command exited {result.return_code}: {result.stdout}\n{result.stderr}"
            )
        return result

    async def upload_json(self, environment, name, content):
        path = self.logs_dir / name
        path.write_text(json.dumps(content, indent=2) + "\n")
        await environment.upload_file(path, self.root + "/" + name)

    async def receipt(self, environment, name):
        target = self.logs_dir / name
        await environment.download_file(str(self.environment_logs_dir / name), target)
        report = json_loads(target.read_text())
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise ValueError(f"Kinex did not produce passing {name}")
        return report

    async def setup(self, environment):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if hashlib.sha256(self.bundle.read_bytes()).hexdigest() != self.bundle_sha256:
            raise ValueError("Kinex bundle changed after preflight")
        await self.checked(environment, "mkdir -p " + self.root + " " + self.workspace, timeout=30)
        await environment.upload_file(self.bundle, self.root + "/agent.tar.gz")
        await self.checked(
            environment, f"tar -xzf {self.root}/agent.tar.gz -C {self.root}", timeout=120
        )
        for source, name in ((self.tool_directory, "tool"), (self.artifact, "artifact")):
            if source:
                await environment.upload_dir(source, self.root + "/" + name)
        (self.logs_dir / "bundle-identity.json").write_text(
            json.dumps({**self.identity, "bundle_sha256": self.bundle_sha256}, indent=2) + "\n"
        )

    async def run(self, instruction, environment, context):
        output = str(self.environment_logs_dir)
        config = {
            "workspace": self.workspace,
            "instruction": instruction,
            "identity": self.identity,
            "mode": self.mode,
            "tool": self.tool,
            "before": self.root + "/before.json",
            "integrity": output + "/integrity.json",
            "artifact": self.root + "/artifact" if self.artifact else None,
            "tool_directory": self.root + "/tool" if self.tool_directory else None,
        }
        await self.upload_json(environment, "prepare.json", config)
        await self.checked(
            environment,
            f"python3 {self.bridge}/prepare.py prepare {self.root}/prepare.json",
            timeout=30,
        )
        await self.checked(
            environment,
            f"{self.root}/bin/node {self.bridge}/run.ts --prepare-python "
            f"{self.workspace} > {output}/python.log 2>&1",
            timeout=self.setup_timeout,
        )
        runner = {
            "mode": self.mode,
            "workspace": self.workspace,
            "output": output + "/execution.json",
            "tool": self.tool,
            "args": self.tool_args,
            "timeout_s": self.timeout,
            "recording": self.recording,
            "require_completion": self.require_completion,
        }
        if self.mode == "develop":
            provider, model = self.model_name.split("/", 1)
            runner.update(
                prompt=instruction,
                model={"provider": provider, "id": model, "effort": self.effort},
            )
        await self.upload_json(environment, "runner.json", runner)
        context.metadata = {
            "mode": self.mode,
            "identity": self.identity,
            "task_success_authority": "Harbor verifier",
            "completed": False,
        }
        try:
            await self.checked(
                environment,
                f"{self.root}/bin/node {self.bridge}/run.ts "
                f"{self.root}/runner.json > {output}/kinex.log 2>&1",
                timeout=self.timeout + 60,
            )
            execution = await self.receipt(environment, "execution.json")
            if execution.get("mode") != self.mode:
                raise ValueError("Kinex execution receipt mode differs from the configured mode")
            if self.recording:
                await self.checked(
                    environment,
                    f"{self.workspace}/.kinex/python/.venv/bin/python "
                    f"{self.bridge}/verify_media.py {self.workspace} "
                    f"{self.root}/runner.json {output}/media.json",
                    timeout=120,
                )
                await self.receipt(environment, "media.json")
            context.metadata["completed"] = True
        finally:
            # Preserve agent code, recordings and the Python lock for review/freeze.
            # Never export auth.
            await self.checked(
                environment,
                "tar --exclude=./.kinex/python/.venv -czf "
                f"{shlex.quote(output)}/workspace.tar.gz "
                f"-C {self.workspace} .",
                timeout=120,
            )
            await self.checked(
                environment,
                f"python3 {self.bridge}/prepare.py verify {self.root}/prepare.json",
                timeout=30,
            )
            await self.receipt(environment, "integrity.json")
