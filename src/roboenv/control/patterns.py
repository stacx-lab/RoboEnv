"""Ways to combine a pretrained policy with code, as functions instead of prose.

`vla_run` is one call that spends a chunk of the episode.  Every useful thing we know
about using it is about *when to stop it and look* — the policy cannot see that it has
failed, cannot be told where, and will happily keep moving after the task is won.  The
helpers here are the shapes that came out of the RoboDojo campaign (three hybrid wins)
and the 2026-08 measurements, written so a solution program is three lines rather than
thirty, and so the campaign can compare patterns instead of re-implementing them.

**Why a chunk boundary changes the outcome, and when it must not be used.**
``run_policy`` restores the policy's sampling key at the *first inference of every*
``vla_run`` call, which is what makes a program replay (``roboenv/vla/runner.py``).  A
500-frame call is therefore one sampling chain; five 100-frame calls are five chains,
each restarted from the same key but started from a *different observation* — so the
action sequences genuinely differ.  **A chunk boundary restarts the plan.**  That is a
gift when the plan was going nowhere and a disaster when it was working, and both were
measured on 2026-08-28:

* LIBERO, short horizons and one leg — chunking at 100 flips 6 of 13 one-shot failures
  to wins (spatial_6, spatial_8, goal_5, 10_5, 10_6, 90_30), 11 of 13 on LIBERO-PRO, and
  costs 2-4x fewer simulator steps on tasks one shot already won.
* RoboCasa composites, one long transport — ``MoveToCounter`` certifies at a single
  ``vla_run(900)`` and scores **0.0** when the same 900 frames are spent as six calls of
  150, the object travelling 2.393 m cumulatively without ever settling.

So: chunk when you expect to lose, or when the win is a short reach the policy either
finds or does not.  Do not chunk a transport that has to converge.  ``vla-chunked`` is
also not a safe *screen* on RoboCasa for the same reason — it can report a failure the
one-shot call would not have.

They take the tools as arguments — a solution program has them as globals, a test has an
adapter — so nothing here imports a simulator::

    from roboenv.control.patterns import supervised_chunks, stalled

    out = supervised_chunks(vla_run, chunk=100, budget=500,
                            observe=lambda: get_object_pose("bowl")[0],
                            solved=lambda r: r["reward_after"] >= 1.0)
    if not out.solved:                      # the policy got it close; finish by hand
        place_gently("bowl", ...)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ChunkedRun",
    "moved",
    "stalled",
    "supervised_chunks",
    "total_moved",
    "with_fallback",
]


def moved(report: dict, threshold_m: float = 0.01) -> float:
    """The largest object displacement a ``vla_run`` report saw (0.0 when nothing moved)."""
    values = [float(v) for v in (report.get("objects_moved_m") or {}).values()]
    biggest = max(values, default=0.0)
    return biggest if biggest >= threshold_m else 0.0


def total_moved(reports: Sequence[dict], threshold_m: float = 0.01) -> dict[str, float]:
    """How far each object moved across the whole run, summed over the chunks.

    A per-call report only says what the last chunk did.  The signature worth reacting to
    is *confident transport with a flat reward* — libero.90_80 moved an object
    0.29 + 0.18 + 0.33 + 0.49 + 0.57 m over five chunks and never scored — and that is
    only visible cumulatively.
    """
    out: dict[str, float] = {}
    for report in reports:
        for name, value in (report.get("objects_moved_m") or {}).items():
            distance = float(value)
            if distance >= threshold_m:
                out[name] = round(out.get(name, 0.0) + distance, 4)
    return out


def stalled(reports: Sequence[dict], patience: int = 2, threshold_m: float = 0.01) -> bool:
    """Whether the last ``patience`` calls moved nothing and gained no reward.

    This is the measured stopping rule: two consecutive chunks that move nothing while
    the reward stays flat mean the policy is not going to recover on its own (RoboDojo
    2026-08-18, and every zero-scoring probe since).

    It says nothing about the *start* of a run.  A policy often spends its first chunks
    reaching — or, on RoboCasa, driving the base for 300-600 frames — with nothing but
    the robot moving, so :func:`supervised_chunks` refuses to consult it before
    ``warmup_frames``.  Without that guard it aborted libero.10_2 at frame 120 on an
    episode that wins at frame 300, and stopped 10 of 16 RoboCasa runs at exactly 200.

    It is also only as good as what ``objects_moved_m`` reports: a simulator whose
    adapter lists free objects but not the fixture joints a task scores will look stalled
    while a door swings open.  Check the report before trusting the rule.
    """
    if len(reports) < patience:
        return False
    tail = reports[-patience:]
    return all(
        moved(r, threshold_m) == 0.0 and r.get("reward_after", 0.0) <= r.get("reward_before", 0.0)
        for r in tail
    )


def _one_or_two(fn: Callable[..., bool]) -> Callable[[Any, Any], bool]:
    """Adapt a predicate to the two-argument call site, by what it actually requires.

    Only *required positional* parameters count: :func:`stalled` takes tuning keywords
    with defaults and must still be called with one argument.
    """
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    try:
        required = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in kinds and p.default is inspect.Parameter.empty
        ]
        takes_two = len(required) >= 2
    except (TypeError, ValueError):  # builtins and C callables
        takes_two = False
    return (lambda a, b: bool(fn(a, b))) if takes_two else (lambda a, _b: bool(fn(a)))


@dataclass
class ChunkedRun:
    """What :func:`supervised_chunks` did: the reports, and why it stopped."""

    reports: list[dict] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    solved: bool = False
    stopped_because: str = ""

    @property
    def frames(self) -> int:
        """Policy frames actually spent."""
        return sum(int(r.get("actions_taken", 0)) for r in self.reports)

    @property
    def reward(self) -> float:
        """The reward after the last chunk."""
        return float(self.reports[-1].get("reward_after", 0.0)) if self.reports else 0.0

    def as_dict(self) -> dict[str, Any]:
        """A compact summary for a solution's printout or a findings row."""
        return {
            "chunks": len(self.reports),
            "frames": self.frames,
            "reward": self.reward,
            "solved": self.solved,
            "stopped_because": self.stopped_because,
        }


def supervised_chunks(
    vla_run: Callable[..., dict],
    *,
    chunk: int = 100,
    budget: int = 500,
    observe: Callable[[], Any] | None = None,
    solved: Callable[..., bool] | None = None,
    give_up: Callable[..., bool] | None = None,
    warmup_frames: int = 600,
    instruction: str | None = None,
    on_chunk: Callable[[int, dict, Any], None] | None = None,
) -> ChunkedRun:
    """Run the policy in short calls, looking at the world between them.

    One long call spends the episode whether or not it is helping; a chunk is a decision
    point.  Stops at the first of: ``solved`` (default: reward 1.0), ``give_up``
    (default: :func:`stalled` over two chunks), or the frame ``budget``.

    Args:
        vla_run: the adapter's tool.
        chunk: frames per call — 60-150 is the measured useful range; below ~40 the
            policy has barely started an approach.
        budget: total frames the policy may spend.
        observe: called after every chunk; its value is kept in ``observations`` so the
            program (and the findings row) can say what the policy actually changed.
        solved: verdict on a report; default ``reward_after >= 1.0``.  It may take the
            observation as a second argument (``solved(report, observation)``) when the
            benchmark's own predicate, not the reward, is what says "done".
        give_up: verdict on the reports so far, optionally also the observations
            (``give_up(reports, observations)``); default :func:`stalled`.
        warmup_frames: frames the policy may spend before ``give_up`` is consulted at
            all.  Reaching moves nothing and the default stall rule would call that a
            stall: 200 is the measured floor on LIBERO, but a RoboCasa policy drives the
            base for 300-600 frames before it touches anything and 10 of 16 runs stopped
            at exactly frame 200 with the old default.  600 is therefore the default;
            lower it when you know the policy acts immediately.  Wasted frames are
            cheap, a falsely abandoned win is not.
        instruction: passed through to every call (a sub-goal phrased like the
            benchmark's own language).
        on_chunk: ``(index, report, observation)`` after each call, for printing.

    Returns:
        :class:`ChunkedRun`.
    """
    if chunk <= 0 or budget <= 0:
        raise ValueError("chunk and budget must be positive")
    is_solved = _one_or_two(solved or (lambda r: float(r.get("reward_after", 0.0)) >= 1.0))
    is_done = _one_or_two(give_up or stalled)
    out = ChunkedRun()
    spent = 0
    while spent < budget:
        n = min(chunk, budget - spent)
        report = (
            vla_run(n_actions=n)
            if instruction is None
            else vla_run(n_actions=n, instruction=instruction)
        )
        out.reports.append(report)
        spent += int(report.get("actions_taken", n))
        observation = observe() if observe is not None else None
        out.observations.append(observation)
        if on_chunk is not None:
            on_chunk(len(out.reports) - 1, report, observation)
        if is_solved(report, observation):
            out.solved = True
            out.stopped_because = "solved"
            return out
        if report.get("note") == "the task ended":
            out.stopped_because = "the episode ended"
            return out
        if int(report.get("actions_taken", 0)) == 0:
            out.stopped_because = "the policy returned no actions"
            return out
        if spent >= warmup_frames and is_done(out.reports, out.observations):
            out.stopped_because = "stalled"
            return out
    out.stopped_because = "frame budget spent"
    return out


def with_fallback(
    stage: Callable[[], Any],
    fallback: Callable[[], Any],
    *,
    attempts: int = 2,
    ok: Callable[[Any], bool] | None = None,
) -> tuple[Any, str]:
    """Try ``stage`` up to ``attempts`` times, then ``fallback`` once.

    The cheap way to pair the two halves when neither is reliably better: a coded stage
    that sometimes fails, with the policy (or the other way round) behind it.  Returns
    ``(result, "stage" | "fallback")``; an exception from ``stage`` counts as a failure,
    an exception from ``fallback`` propagates.
    """
    accept = ok or (lambda value: bool(value))
    for _ in range(max(1, attempts)):
        try:
            result = stage()
        except Exception:  # a failed stage is what the fallback is for
            continue
        if accept(result):
            return result, "stage"
    return fallback(), "fallback"
