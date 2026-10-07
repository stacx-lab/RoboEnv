"""Native OmniGibson scene construction and owner-only BDDL evaluation."""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_CACHE_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME")


def _writable(directory: Path) -> bool:
    try:
        probe = directory / ".roboenv-probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def isaac_apps() -> Path | None:
    """Isaac Sim's ``apps`` directory as Kit will see it, or None without Isaac Sim."""
    if os.environ.get("EXP_PATH"):
        return Path(os.environ["EXP_PATH"])
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent / "apps"


def writable_apps(real: Path, parent: Path) -> Path:
    """A writable stand-in for a read-only Isaac Sim ``apps`` directory.

    OmniGibson copies its experience file into ``apps`` at every launch, and Kit then resolves
    the extension folders relative to that file (``${app}/../exts`` and its siblings), so a
    prepared interpreter that is mounted read-only cannot start it. The stand-in is a directory
    of symlinks, unique to this process: the siblings of ``apps`` at the top, and inside
    ``apps`` every file of the original except OmniGibson's own, which it writes itself (a link
    would write through to the read-only original).
    """
    real = real.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    shadow = Path(tempfile.mkdtemp(prefix="isaacsim-", dir=parent))
    atexit.register(shutil.rmtree, shadow, ignore_errors=True)
    apps = shadow / "apps"
    apps.mkdir()
    for entry in real.parent.iterdir():
        if entry.name != "apps":
            (shadow / entry.name).symlink_to(entry)
    for entry in real.iterdir():
        if not entry.name.lower().startswith("omnigibson"):
            (apps / entry.name).symlink_to(entry)
    return apps


def fix_cache_env() -> None:
    """Re-point what OmniGibson and Isaac Sim write at launch, when it is not writable, at
    ``<repo>/.cache``: the caches named by ``_CACHE_VARS`` and Isaac Sim's ``apps`` directory."""
    from roboenv_runtime.paths import repo_root

    base = repo_root() / ".cache"
    for var in _CACHE_VARS:
        current = os.environ.get(var)
        target = Path(current).expanduser() if current else None
        if target is not None:
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            else:
                if _writable(target):
                    continue  # writable — leave it alone
        fallback = base / var.lower()
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(fallback)
    apps = isaac_apps()
    if apps is not None and not _writable(apps):
        os.environ["EXP_PATH"] = str(writable_apps(apps, base / "behavior"))
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    depot = repo_root() / "third_party/BEHAVIOR-1K/OmniGibson/datasets"
    if depot.is_dir():
        os.environ.setdefault("OMNIGIBSON_DATA_PATH", str(depot))


def load_challenge_index() -> dict[str, dict[str, Any]]:
    """The 2026 challenge task index (id -> instruction/scene/rooms), or ``{}``.

    Source: ``third_party/BEHAVIOR-1K/docs/challenge/task_data.json``.
    """
    import json

    from roboenv_runtime.paths import repo_root

    path = repo_root() / "third_party/BEHAVIOR-1K/docs/challenge/task_data.json"
    if not path.is_file():
        return {}
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    except (OSError, ValueError, KeyError):
        return {}
    return {t["id"]: t for t in tasks if isinstance(t, dict) and "id" in t}


class BehaviorBackend:
    """Construct one native world; BehaviorDevices owns stepping and shutdown."""

    def __init__(
        self,
        activity_name: str,
        scene_model: str | None = None,
        activity_instance_id: int = 0,
        max_steps: int = 6000,
    ) -> None:
        self.activity_name = activity_name
        self.activity_instance_id = int(activity_instance_id)
        self.max_steps = int(max_steps)
        self.challenge = load_challenge_index().get(activity_name, {})
        self.scene_model = scene_model or self.challenge.get("scene_model")
        if not self.scene_model:
            raise ValueError(
                f"activity {activity_name!r} is not in the challenge index and no "
                "scene_model was given (adapter_kwargs: {scene_model: ...})"
            )
        self.env: Any = None
        self.robot: Any = None

    def build_config(self) -> dict[str, Any]:
        """The roboenv-owned R1Pro native scene config, specialised to this activity."""
        import yaml

        cfg = yaml.safe_load((Path(__file__).parent / "r1pro.yaml").read_text())
        cfg["env"]["num_envs"] = 1
        cfg["task"]["activity_name"] = self.activity_name
        cfg["task"]["activity_instance_id"] = self.activity_instance_id
        cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        cfg["scene"]["scene_model"] = self.scene_model
        # loading only the task's rooms keeps boot time sane; whole scene when unknown
        cfg["scene"]["load_room_types"] = self.challenge.get("rooms") or None
        return cfg

    def boot(self) -> None:
        """Import omnigibson and create the environment (idempotent)."""
        if self.env is not None:
            return
        from roboenv.provisioning.acceptance import acceptance_environment
        from roboenv.provisioning.gpu import configure_gpu

        os.environ.update(acceptance_environment())
        selection = configure_gpu("behavior")
        os.environ["OMNIGIBSON_GPU_ID"] = "0"
        os.environ["OMNIGIBSON_RENDER_GPU"] = str(selection["vulkan_index"])
        fix_cache_env()
        import omnigibson as og

        self.env = og.Environment(configs=self.build_config())
        if self.env.num_envs != 1 or len(self.env.robots[0]) != 1:
            raise ValueError("BEHAVIOR requires one scene with one robot")
        self.robot = self.env.robots[0][0]
        log.info("BEHAVIOR env up: %s on %s", self.activity_name, self.scene_model)

    def goal_report(self) -> dict[str, Any]:
        """Fresh goal evaluation without stepping the simulator."""
        task = self.env.task
        predicate = task._termination_conditions.get("predicate")
        if predicate is not None and getattr(predicate, "_check_goal_fn", None):
            done, status = predicate._check_goal_fn(0)
        else:  # fall back to the last stepped status
            done, status = bool(task.success[0]), {"satisfied": [], "unsatisfied": []}
        return {
            "instruction": self.challenge.get("instruction", ""),
            "conditions": list(task.activity_natural_language_goal_conditions),
            "satisfied": list(status.get("satisfied", [])),
            "unsatisfied": list(status.get("unsatisfied", [])),
            "success": bool(done),
        }
