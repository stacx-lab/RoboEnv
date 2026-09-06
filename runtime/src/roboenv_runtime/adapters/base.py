"""Bridging cap-x task environments into the adapter contract.

cap-x already knows how to build a low-level simulator plus a privileged API
from a YAML node; :class:`CapxBridge` instantiates that stack and re-exposes
the API's functions as tools (their docstrings come along and become the
prompt's ``api`` section).  Nothing in cap-x is modified.

Every tool passes through one wrapper that

* counts calls (a budget for platforms with no simulator step counter),
* coerces numpy arguments to plain Python (upstream string comparisons choke
  on arrays),
* pins the working directory back to the simulator's (RoboTwin resolves
  assets relative to its cwd while agent code chdirs into a workspace),
* trips a **planning circuit breaker** after repeated identical IK/planning
  failures on the same arm, with a message telling the agent to change
  approach — escalating retries near an IK wall is how objects get flung.

The wrapper deliberately does **not** set ``__wrapped__``: the bound method
would leak ``__self__`` → API → raw simulator, which agents exploited to
teleport objects into goal states.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from roboenv_runtime.core.adapter import AdapterBase
from roboenv_runtime.core.types import Budget, Capability, ScoreReport
from roboenv_runtime.paths import capx_root

__all__ = ["CapxBridge", "apply_overrides", "plain"]

log = logging.getLogger(__name__)

PLANNING_FAILURE_MARKERS = ("planning failed", "IK Failed")
CIRCUIT_BREAKER_AFTER = 4


def apply_overrides(node: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply ``{"a.b.c": value}`` dotted-path overrides to a nested dict, in place."""
    for dotted, value in (overrides or {}).items():
        cursor: Any = node
        *parents, leaf = dotted.split(".")
        for key in parents:
            cursor = cursor[key]
        cursor[leaf] = value


def plain(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return type(value)(plain(x) for x in value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def _arm_key(name: str, kwargs: Mapping[str, Any]) -> str:
    arm = kwargs.get("arm") or kwargs.get("side")
    if isinstance(arm, str):
        return arm
    for suffix in ("left", "right"):
        if name.endswith(f"_{suffix}"):
            return suffix
    return "any"


def ensure_capx_importable() -> Path:
    """Put the cap-x checkout on ``sys.path`` (idempotent) and return it."""
    root = capx_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


class CapxBridge(AdapterBase):
    """Any cap-x YAML-configured task env becomes an adapter.

    Subclasses provide identity (``name``, ``embodiment``), prompt text, and
    usually a constructor that synthesises the YAML node; they may add
    platform-specific tools by overriding :meth:`extra_tools`.
    """

    name: ClassVar[str] = "capx-bridge"
    embodiment: ClassVar[str] = "general"
    capabilities: ClassVar[frozenset[str]] = frozenset({Capability.PRIVILEGED, Capability.VIDEO})
    fallback_call_budget: ClassVar[int] = 60_000

    def __init__(self, config_path: str | Path, overrides: Mapping[str, Any] | None = None) -> None:
        root = ensure_capx_importable()
        import yaml
        from capx.envs.configs.instantiate import instantiate

        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = root / cfg_file
        node = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["env"]
        if isinstance(node.get("cfg"), dict):
            node["cfg"].pop("multi_turn_prompt", None)
        apply_overrides(node, overrides or {})
        self._env = instantiate(node)
        self._lle = self._env.low_level_env
        # platform code (e.g. RoboTwin) chdirs at construction and resolves
        # assets relative to that cwd; remember it so tool calls can restore
        # it even when agent code has chdir'ed into its own workspace
        self._env_cwd = os.getcwd()
        self._calls = 0
        self._plan_fails: dict[str, int] = {}
        self._tool_cache: dict[str, Callable[..., Any]] | None = None

    # ------------------------------------------------------------ tools

    def api_functions(self) -> dict[str, Callable[..., Any]]:
        """The raw functions of every cap-x API attached to the env."""
        fns: dict[str, Callable[..., Any]] = {}
        for api in getattr(self._env, "_apis", {}).values():
            fns.update(api.functions())
        return fns

    def extra_tools(self) -> dict[str, Callable[..., Any]]:
        """Platform-specific additions; subclasses override.  Never shadow API names."""
        return {}

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        """All tools, each wrapped exactly once (cached)."""
        if self._tool_cache is None:
            fns = self.api_functions()
            for name, fn in self.extra_tools().items():
                fns.setdefault(name, fn)
            self._tool_cache = {name: self._counted(name, fn) for name, fn in fns.items()}
        return dict(self._tool_cache)

    def _counted(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        bridge = self

        def inner(*args: Any, **kwargs: Any) -> Any:
            bridge._calls += 1
            args = tuple(plain(a) for a in args)
            kwargs = {k: plain(v) for k, v in kwargs.items()}
            key = _arm_key(name, kwargs)
            agent_cwd = os.getcwd()
            pinned = agent_cwd != bridge._env_cwd
            if pinned:
                os.chdir(bridge._env_cwd)
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:
                message = str(exc)
                if any(marker in message for marker in PLANNING_FAILURE_MARKERS):
                    count = bridge._plan_fails.get(key, 0) + 1
                    bridge._plan_fails[key] = count
                    if count >= CIRCUIT_BREAKER_AFTER:
                        raise type(exc)(
                            f"{message}\n[circuit breaker] {count} consecutive planning failures "
                            f"for the {key} arm — this region is likely unreachable for it. STOP "
                            "retrying variations here: switch arm, use goto_pose_flex_*, "
                            "transfer_object, or move the object closer instead."
                        ) from exc
                raise
            finally:
                if pinned:
                    os.chdir(agent_cwd)
            bridge._plan_fails[key] = 0
            return out

        inner.__name__ = getattr(fn, "__name__", name)
        inner.__qualname__ = inner.__name__
        inner.__doc__ = fn.__doc__
        try:  # keep the real signature for documentation without exposing __wrapped__
            import inspect

            inner.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            pass
        return inner

    # ------------------------------------------------------------ contract

    def goal(self) -> str:
        """Benchmark-provided language goal, when the simulator carries one."""
        handle = getattr(self._lle, "handle", None)
        return str(getattr(handle, "task_language", "") or "")

    def reset(self, seed: int) -> None:
        """Reset the episode and (re)start video capture."""
        self._env.reset(seed=seed)
        self._plan_fails.clear()
        try:
            self._env.enable_video_capture(True, clear=True)
        except TypeError:
            self._env.enable_video_capture(True)
        except Exception:
            pass

    def frames(self) -> list[Any]:
        """Captured video frames for the trial (empty if capture unsupported)."""
        try:
            return list(self._env.get_video_frames())
        except Exception:
            return []

    def score(self) -> ScoreReport:
        """``compute_reward`` / ``task_completed`` of the low-level env."""
        try:
            reward = float(self._lle.compute_reward())
        except Exception:
            reward = 0.0
        try:
            completed = bool(self._lle.task_completed())
        except Exception:
            completed = False
        return ScoreReport(reward=reward, raw_score=reward, completed=completed)

    def budget(self) -> Budget:
        """Simulator steps when the platform counts them, else API calls."""
        used = int(
            getattr(self._lle, "_sim_step_count", 0) or getattr(self._lle, "step_count", 0) or 0
        )
        cap = int(getattr(self._lle, "max_steps", 0) or 0)
        if used == 0:  # platform keeps no sim counter: report action calls
            return Budget(used=self._calls, max=cap or self.fallback_call_budget, unit="api_calls")
        return Budget(used=used, max=cap or max(used, 1), unit="sim_steps")

    def close(self) -> None:
        """Close the low-level env if it can be closed."""
        closer = getattr(self._lle, "close", None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()
