"""The shared VLA-as-a-tool loop: observe, infer, execute a few actions, repeat.

Every adapter that offers a pretrained policy as a tool implements the small
:class:`VlaHooks` surface (how to build the policy's observation, how to apply
one action, how many control steps are left) and calls :func:`run_policy`; the
before/after report, the budget guard and the replanning cadence are the same
for every simulator, so what a model learns about ``vla_run`` on one benchmark
carries to the others.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from roboenv_runtime.vla.spec import PolicySpec

if TYPE_CHECKING:
    from roboenv_runtime.vla.client import WebsocketPolicyClient

__all__ = ["VlaHooks", "run_policy"]

log = logging.getLogger(__name__)


class VlaHooks(Protocol):
    """What an adapter provides to drive one policy."""

    def begin(self) -> None:
        """Prepare the simulator for policy control (e.g. switch controllers)."""

    def end(self) -> None:
        """Undo :meth:`begin` — always called, even after an error."""

    def element(self) -> dict[str, Any]:
        """The policy's observation: images already preprocessed, state, prompt."""

    def step(self, action: np.ndarray) -> None:
        """Apply ONE action for one control step (charging the budget)."""

    def steps_left(self) -> int:
        """Control steps still allowed by the episode's budget/horizon."""


def iterate_policy(
    hooks: VlaHooks,
    client: WebsocketPolicyClient,
    spec: PolicySpec,
    *,
    n_actions: int,
    reward_fn: Callable[[], float],
    poses_fn: Callable[[], Mapping[str, Any]],
    budget_fn: Callable[[], Any],
    replan_steps: int | None = None,
    moved_threshold_m: float = 0.01,
) -> dict[str, Any]:
    """Run ``spec``'s policy for up to ``n_actions`` control steps and report.

    Executes ``replan_steps`` actions of every predicted chunk before asking
    again (openpi's evaluation cadence).  For a ``deterministic`` policy the first
    inference of the run resets the server's sampling key, so replaying the same
    program against the same episode calls the policy identically.  Stops early when
    the budget runs out, the policy returns nothing, or ``until()`` — a caller-supplied
    check evaluated after every executed action — returns true: the policy owns the
    motion it was trained for, the program decides when that motion is done.  Returns
    ``{"policy", "actions_taken", "reward_before", "reward_after", "objects_moved_m",
    "budget", "note"}``; ``note`` says ``"until condition met"`` when the check stopped it.
    """
    n_actions = int(n_actions)
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    replan = int(replan_steps or spec.replan_steps)
    r0 = float(reward_fn())
    p0 = {k: np.asarray(v, dtype=float).reshape(-1)[:3].copy() for k, v in poses_fn().items()}
    taken = 0
    note = ""
    plan: deque[np.ndarray] = deque()
    hooks.begin()
    try:
        while taken < n_actions:
            if hooks.steps_left() <= 0:
                note = "no steps left"  # resolved below: solved, or out of budget
                break
            if not plan:
                # the first inference of the run restores the policy's sampling key, so
                # the same program on the same episode calls the policy identically
                reset = spec.rng_seed if (taken == 0 and spec.deterministic) else None
                out = client.infer(hooks.element(), reset_rng=reset)
                chunk = np.asarray(out["actions"], dtype=float)
                if chunk.ndim != 2 or chunk.shape[0] == 0:
                    note = f"policy returned an empty chunk (shape {chunk.shape})"
                    break
                plan.extend(chunk[:replan])
            hooks.step(plan.popleft())
            taken += 1
            if (yield {"actions_taken": taken}):
                note = "until condition met"
                break
    finally:
        hooks.end()
    r1 = float(reward_fn())
    if note == "no steps left":
        # the hooks stop the run both when the task ends and when the budget runs out;
        # saying "budget exhausted" over a win reads as a failure (measured on
        # robotwin.click_bell, which is solved at frame 70 of 600)
        note = "the task ended" if r1 >= 1.0 else "episode budget exhausted"
    moved: dict[str, float] = {}
    for name, pose in poses_fn().items():
        before = p0.get(name)
        if before is None:
            continue
        d = float(np.linalg.norm(np.asarray(pose, dtype=float).reshape(-1)[:3] - before))
        if d > moved_threshold_m:
            moved[name] = round(d, 3)
    out: dict[str, Any] = {
        "policy": spec.name,
        "actions_taken": taken,
        "reward_before": round(r0, 3),
        "reward_after": round(r1, 3),
        "objects_moved_m": moved,
        "budget": budget_fn(),
    }
    if note:
        out["note"] = note
    return out


def drive_policy(iterator, until=None):
    try:
        next(iterator)
        while True:
            iterator.send(bool(until()) if until is not None else False)
    except StopIteration as stopped:
        return stopped.value
    finally:
        iterator.close()


def run_policy(hooks, client, spec, *, until=None, **kwargs):
    """Run the same iterator synchronously; native callbacks resume it on the same thread."""
    return drive_policy(iterate_policy(hooks, client, spec, **kwargs), until)
