"""Single-owner reward clock for the stateful RoboDojo worker.

RoboDojo has two superficially similar scoring surfaces with very different
semantics:

* task-level ``EvalEnv.run_reward`` / ``EvalEnv.get_score`` *register* scorer
  programs and must run exactly once after reset;
* ``EvalEnv.reward_manager`` evaluates that already-registered program.

The legacy Cap-X score adapter mixes the two and pumps the state machine on
every read.  ``ScoreClock`` instead seals the registrars, advances the reward
manager at the benchmark's action-frame cadence, and provides a monotonic
server-side success latch.  It intentionally never calls ``adapter.score``.
"""

# Moved from codex_dojo_alltasks/harness/score_clock.py on 2026-08-26 (refactor P3).

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import Any, Iterator

_MISSING = object()


def _as_list(value: Any) -> list[Any] | None:
    """Best-effort conversion that also handles numpy arrays without importing it."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return [value]


def _first_float(value: Any, default: float = 0.0) -> float:
    values = _as_list(value)
    if not values:
        return default
    try:
        result = float(values[0])
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first_int(value: Any, default: int = 0) -> int:
    values = _as_list(value)
    if not values:
        return default
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return default


def pump_registered_reward_manager(
    reward_manager: Any,
    *,
    env_idx_list: tuple[int, ...] | list[int] = (0,),
    ticks: int = 5,
) -> dict[str, Any]:
    """Advance one already-registered RoboDojo reward program.

    Task-level EvalEnv.run_reward and EvalEnv.get_score are registrars, not
    score reads. Reset owns those calls. A final score pump may only advance the
    existing RewardManager and invalidate its gated-score cache; re-registering
    here appends fresh check/query queues after a program has completed them.
    """
    count = int(ticks)
    if count < 0:
        raise ValueError("ticks must be non-negative")
    indices = [int(index) for index in env_idx_list]
    if not indices:
        raise ValueError("env_idx_list must not be empty")
    step = getattr(reward_manager, "step", None)
    if not callable(step):
        raise TypeError("RewardManager.step is not callable")
    for _ in range(count):
        step(env_idx_list=list(indices))
    if hasattr(reward_manager, "_gated_score_lst"):
        reward_manager._gated_score_lst = None
    return {"ticks": count, "env_idx_list": indices}


class ScoreClock:
    """Own reward evaluation for one already-reset RoboDojo adapter.

    Args:
        adapter: Refactor ``RoboDojoAdapter`` (or a compatible fake).  The
            actual EvalEnv must be reachable as ``adapter._env._env``.
        collect_interval: Number of raw EvalEnv ``step`` calls in one official
            action frame.  Defaults to ``EvalEnv.obs_manager.collect_interval``.
        env_idx_list: RewardManager environments to evaluate.  The campaign is
            single-environment and therefore defaults to ``(0,)``.

    Install this only *after* ``adapter.reset(seed)`` has invoked the two task
    registrars.  Call :meth:`close` before resetting the adapter again.
    """

    SUCCESS_THRESHOLD = 1.0 - 1e-3
    _OWNER_ATTR = "_codex_dojo_score_clock"

    def __init__(
        self,
        adapter: Any,
        collect_interval: int | None = None,
        env_idx_list: tuple[int, ...] | list[int] = (0,),
    ) -> None:
        try:
            eval_env = adapter._env._env
        except AttributeError as exc:
            raise TypeError("ScoreClock requires adapter._env._env (the RoboDojo EvalEnv)") from exc
        reward_manager = getattr(eval_env, "reward_manager", None)
        if reward_manager is None:
            raise TypeError("adapter._env._env has no reward_manager")

        if collect_interval is None:
            obs_manager = getattr(eval_env, "obs_manager", None)
            collect_interval = getattr(obs_manager, "collect_interval", None)
        try:
            collect_interval = int(collect_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("collect_interval must be a positive integer") from exc
        if collect_interval <= 0:
            raise ValueError("collect_interval must be a positive integer")
        indices = [int(index) for index in env_idx_list]
        if not indices:
            raise ValueError("env_idx_list must not be empty")

        existing = getattr(eval_env, self._OWNER_ATTR, None)
        if existing is not None:
            raise RuntimeError("another ScoreClock already owns this EvalEnv")

        self.adapter = adapter
        self.eval_env = eval_env
        self.reward_manager = reward_manager
        self.collect_interval = collect_interval
        self.env_idx_list = indices
        self._lock = threading.RLock()
        self._native_local = threading.local()
        self._saved_instance_attrs: dict[str, Any] = {}
        self._closed = False
        self._latched = False
        self._submitted = False
        self._submit_result: dict[str, Any] | None = None
        self._submit_error: Exception | None = None
        self._last_rewards: list[Any] | None = None
        self._last_scores: list[Any] | None = None

        self.programmatic_raw_calls = 0
        self.suppressed_raw_calls = 0
        self.programmatic_ticks = 0
        self.native_ticks = 0
        self.blocked_registrar_calls = 0
        self.blocked_registrar_calls_by_name = {
            "run_reward": 0,
            "get_score": 0,
        }

        setattr(eval_env, self._OWNER_ATTR, self)
        try:
            self._seal_registrar("run_reward")
            self._seal_registrar("get_score")
            self._patch_raw_step()
            # RewardManager.get_score() treats an absent gate as success and
            # may otherwise expose the terminal tier on the first peek.
            # Establish the non-final gate once, without a reward tick.
            self._refresh_reward(final_check=False)
        except Exception:
            self._restore_patches()
            if getattr(eval_env, self._OWNER_ATTR, None) is self:
                delattr(eval_env, self._OWNER_ATTR)
            raise

    # ---------------------------------------------------------------- patches

    def _save_instance_attr(self, name: str) -> None:
        namespace = getattr(self.eval_env, "__dict__", {})
        self._saved_instance_attrs[name] = namespace.get(name, _MISSING)

    def _seal_registrar(self, name: str) -> None:
        original = getattr(self.eval_env, name, None)
        if not callable(original):
            return
        self._save_instance_attr(name)

        def sealed_registrar(*_args: Any, **_kwargs: Any) -> None:
            with self._lock:
                if self._closed:
                    return None
                self.blocked_registrar_calls += 1
                self.blocked_registrar_calls_by_name[name] += 1
            return None

        sealed_registrar.__name__ = f"sealed_{name}"
        sealed_registrar.__doc__ = (
            f"ScoreClock seal for EvalEnv.{name}; registration already happened at reset."
        )
        setattr(self.eval_env, name, sealed_registrar)

    def _patch_raw_step(self) -> None:
        original = getattr(self.eval_env, "step", None)
        if not callable(original):
            raise TypeError("adapter._env._env.step is not callable")
        self._save_instance_attr("step")

        def clocked_step(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._observe_raw_step()
            return result

        clocked_step.__name__ = "score_clock_step"
        clocked_step.__doc__ = "EvalEnv.step wrapped by ScoreClock."
        setattr(self.eval_env, "step", clocked_step)

    def _restore_patches(self) -> None:
        for name, original_instance_value in reversed(list(self._saved_instance_attrs.items())):
            if original_instance_value is _MISSING:
                try:
                    delattr(self.eval_env, name)
                except AttributeError:
                    pass
            else:
                setattr(self.eval_env, name, original_instance_value)
        self._saved_instance_attrs.clear()

    # ------------------------------------------------------------- clock logic

    def _native_depth(self) -> int:
        return int(getattr(self._native_local, "depth", 0))

    def _observe_raw_step(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._native_depth() > 0:
                self.suppressed_raw_calls += 1
                return
            if self._submitted:
                # Submission freezes the official result.  The worker also
                # rejects subsequent actions, but keep this guard local.
                return
            self.programmatic_raw_calls += 1
            if self.programmatic_raw_calls % self.collect_interval != 0:
                return
            self.reward_manager.step(env_idx_list=list(self.env_idx_list))
            self.programmatic_ticks += 1
            self._refresh_reward(final_check=False)

    def _read_score(self) -> list[Any] | None:
        reader = getattr(self.reward_manager, "get_score", None)
        if not callable(reader):
            return self._last_scores
        scores = _as_list(reader())
        self._last_scores = scores
        return scores

    def _refresh_reward(self, *, final_check: bool) -> None:
        reader = getattr(self.reward_manager, "get_reward", None)
        if not callable(reader):
            raise TypeError("RewardManager.get_reward is not callable")
        rewards = _as_list(reader(final_check=final_check))
        self._last_rewards = rewards
        self._read_score()
        if _first_float(rewards) >= self.SUCCESS_THRESHOLD:
            self._latched = True

    @contextmanager
    def native_action(self) -> Iterator[None]:
        """Suppress custom raw-step ticks around one native ``take_action``.

        RoboDojo's native path already performs one ``RewardManager.step`` and
        one reward read.  On a successful context exit we only refresh our
        local latch/counters via :meth:`observe_native_tick`; no additional
        ``RewardManager.step`` is issued.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("ScoreClock is closed")
            if self._submitted:
                raise RuntimeError("cannot take a native action after submit")
            depth = self._native_depth()
            self._native_local.depth = depth + 1
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            with self._lock:
                current = self._native_depth()
                self._native_local.depth = max(current - 1, 0)
                outermost = current == 1
            if succeeded and outermost:
                self.observe_native_tick()

    def observe_native_tick(self) -> dict[str, Any]:
        """Refresh state after a native action without calling ``rm.step``."""

        with self._lock:
            if self._closed:
                raise RuntimeError("ScoreClock is closed")
            if self._submitted:
                return self._submit_result or self._snapshot()
            self.native_ticks += 1
            self._refresh_reward(final_check=False)
            return self._snapshot()

    # ---------------------------------------------------------------- reports

    @property
    def latched(self) -> bool:
        with self._lock:
            return self._latched

    @property
    def submitted(self) -> bool:
        with self._lock:
            return self._submitted

    def _snapshot(self) -> dict[str, Any]:
        reward = _first_float(self._last_rewards)
        score = _first_float(self._last_scores)
        completed = reward >= self.SUCCESS_THRESHOLD
        return {
            "reward": reward,
            "reward_vector": list(self._last_rewards or []),
            "raw_score": score,
            "score_vector": list(self._last_scores or []),
            "tier": _first_int(getattr(self.reward_manager, "score_completed_count", None)),
            "final_tier": _first_int(
                getattr(self.reward_manager, "final_score_completed_count", None)
            ),
            "completed": completed,
            "latched": bool(self._latched),
            "submitted": bool(self._submitted),
            "collect_interval": self.collect_interval,
            "programmatic_raw_calls": self.programmatic_raw_calls,
            "suppressed_raw_calls": self.suppressed_raw_calls,
            "programmatic_ticks": self.programmatic_ticks,
            "native_ticks": self.native_ticks,
            "blocked_registrar_calls": self.blocked_registrar_calls,
        }

    def peek(self) -> dict[str, Any]:
        """Read current RewardManager state without advancing reward checks.

        This calls only ``RewardManager.get_score``.  It never calls
        ``adapter.score``, ``EvalEnv.run_reward``, ``EvalEnv.get_score``,
        ``RewardManager.step``, or ``RewardManager.get_reward``.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("ScoreClock is closed")
            if self._submit_result is not None:
                return self._submit_result
            self._read_score()
            return self._snapshot()

    def submit(self) -> dict[str, Any]:
        """Run the final-only checks once and cache the official result."""

        with self._lock:
            if self._closed:
                raise RuntimeError("ScoreClock is closed")
            if self._submit_result is not None:
                return self._submit_result
            if self._submit_error is not None:
                raise RuntimeError(
                    "the one-shot final submit previously failed"
                ) from self._submit_error
            self._submitted = True
            try:
                self._refresh_reward(final_check=True)
                self._submit_result = self._snapshot()
            except Exception as exc:
                self._submit_error = exc
                raise
            return self._submit_result

    # --------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Restore EvalEnv methods so the same adapter can be reset safely."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._restore_patches()
            if getattr(self.eval_env, self._OWNER_ATTR, None) is self:
                delattr(self.eval_env, self._OWNER_ATTR)

    def __enter__(self) -> "ScoreClock":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


__all__ = ["ScoreClock", "pump_registered_reward_manager"]
