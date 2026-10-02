"""Native OmniGibson scene construction and owner-only BDDL evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
_CACHE_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME")


def fix_cache_env() -> None:
    """Re-point unwritable cache env vars at ``<repo>/.cache`` (see module doc)."""
    from roboenv_runtime.paths import repo_root

    base = repo_root() / ".cache"
    for var in _CACHE_VARS:
        current = os.environ.get(var)
        target = Path(current).expanduser() if current else None
        if target is not None:
            try:
                target.mkdir(parents=True, exist_ok=True)
                probe = target / ".roboenv-probe"
                probe.touch()
                probe.unlink()
                continue  # writable — leave it alone
            except OSError:
                pass
        fallback = base / var.lower()
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(fallback)
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
        from omnigibson.eval.utils.eval_utils import TASK_NAMES_TO_ROOMS

        cfg = yaml.safe_load((Path(__file__).parent / "r1pro.yaml").read_text())
        cfg["env"]["num_envs"] = 1
        cfg["task"]["activity_name"] = self.activity_name
        cfg["task"]["activity_instance_id"] = self.activity_instance_id
        cfg["task"]["termination_config"]["max_steps"] = self.max_steps
        cfg["scene"]["scene_model"] = self.scene_model
        # Native evaluation includes transit rooms; display categories omit their floors.
        cfg["scene"]["load_room_types"] = None
        cfg["scene"]["load_room_instances"] = TASK_NAMES_TO_ROOMS.get(self.activity_name)
        return cfg

    def boot(self) -> None:
        """Import omnigibson and create the environment (idempotent)."""
        if self.env is not None:
            return
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
