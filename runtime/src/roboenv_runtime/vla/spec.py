"""Policy specs: one YAML per (pretrained policy, simulator) pair.

A spec says how to *serve* a policy (interpreter, working directory, argv,
environment, port) and what the policy *expects* (cameras, image size, state
layout, action dimension, replanning horizon).  Specs ship with the package
under ``roboenv/vla/policies/`` and a directory named by ``$ROBOENV_VLA_POLICIES``
is searched first, so a deployment can add or override policies without
editing the package.  ``${REPO}`` in any string expands to the repository root
and ``{port}`` in ``argv`` to the port the server is started on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from roboenv_runtime.paths import repo_root

__all__ = ["PolicySpec", "list_policies", "load_policy_spec", "policies_root"]

_PACKAGE_DIR = Path(__file__).resolve().parent / "policies"


def policies_root() -> list[Path]:
    """Directories searched for ``<name>.yaml``, most specific first."""
    dirs: list[Path] = []
    override = os.environ.get("ROBOENV_VLA_POLICIES")
    if override:
        dirs.append(Path(override).expanduser())
    dirs.append(_PACKAGE_DIR)
    return dirs


@dataclass
class PolicySpec:
    """Everything needed to serve one policy and feed it the right observation."""

    name: str
    simulator: str
    protocol: str = "openpi-ws"
    description: str = ""
    interpreter: str = ""
    cwd: str = ""
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    port_base: int = 8800
    checkpoint: str = ""
    download: str = ""
    boot_timeout_s: float = 900.0
    gpu_mib: int = 12000
    obs: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    default: bool = False
    deterministic: bool = True
    rng_seed: int = 0
    tool_doc: str = ""
    source: str = ""

    @property
    def action_dim(self) -> int:
        """Width of one action vector."""
        return int(self.action.get("dim", 7))

    @property
    def replan_steps(self) -> int:
        """Actions executed from each predicted chunk before inferring again."""
        return int(self.action.get("replan_steps", 5))

    def checkpoint_present(self) -> bool:
        """Whether the checkpoint path exists locally."""
        return bool(self.checkpoint) and Path(self.checkpoint).exists()

    def command(self, port: int) -> list[str]:
        """The server command line for ``port``."""
        return [self.interpreter, *[a.replace("{port}", str(port)) for a in self.argv]]


def _expand(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${REPO}", str(root)))
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    return value


def load_policy_spec(name: str) -> PolicySpec:
    """Load ``<name>.yaml`` (or a path to one) and expand ``${REPO}``."""
    path = Path(name)
    if not (path.suffix == ".yaml" and path.exists()):
        for d in policies_root():
            candidate = d / f"{name}.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no policy spec named {name!r} under {policies_root()}")
    data = _expand(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, repo_root())
    known = set(PolicySpec.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    spec = PolicySpec(**data)
    spec.source = str(path)
    spec.env = {k: str(v) for k, v in spec.env.items()}
    spec.argv = [str(a) for a in spec.argv]
    return spec


def list_policies(simulator: str | None = None) -> list[PolicySpec]:
    """Every spec found (deployment overrides shadow package specs by name)."""
    seen: dict[str, PolicySpec] = {}
    for d in policies_root():
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.yaml")):
            if path.stem in seen:
                continue
            spec = load_policy_spec(str(path))
            if simulator is None or spec.simulator == simulator:
                seen[path.stem] = spec
    return list(seen.values())
