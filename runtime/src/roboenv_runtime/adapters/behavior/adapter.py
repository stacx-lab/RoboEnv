"""BEHAVIOR-1K (OmniGibson / Isaac Sim, R1Pro in a house) — the adapter.

The 2026 challenge tasks: full-length household activities in multi-room
scenes, scored by BDDL predicate goals — the simulator checks every goal
condition, the system never grades itself.  This adapter drives the robot at
the SYMBOLIC primitive tier (grasp/open/place teleport-style, physics settles
the result); certificates earned here must say so.  Runs in the ``behavior``
conda env; the first reset boots Isaac Sim (~95 s warm).

Episode instances are presampled files shipped with the benchmark
(``activity_instance_id``), so ``reset(seed)`` is reproducible by
construction — the seed is recorded but selects nothing.
"""

from __future__ import annotations

from typing import Any, ClassVar

from roboenv_runtime.adapters.behavior.backend import PRIMITIVES, STATES, BehaviorBackend
from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.tools import tool
from roboenv_runtime.core.types import Budget, Capability, ScoreReport
from roboenv_runtime.embodiments import load_embodiment

__all__ = ["BehaviorAdapter"]


@register_adapter
class BehaviorAdapter(AdapterBase):
    """BEHAVIOR activity ``activity_name`` on the ``r1pro`` embodiment."""

    name: ClassVar[str] = "behavior"
    embodiment: ClassVar[str] = "r1pro"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {Capability.PRIVILEGED, Capability.VIDEO, Capability.BASE_MOTION}
    )

    def __init__(
        self,
        activity_name: str = "picking_up_trash",
        scene_model: str | None = None,
        activity_instance_id: int = 0,
        max_steps: int = 6000,
        **_: Any,
    ) -> None:
        self.backend = BehaviorBackend(
            activity_name,
            scene_model=scene_model,
            activity_instance_id=activity_instance_id,
            max_steps=max_steps,
        )

    # ------------------------------------------------------------ context

    @property
    def profile(self):
        """The ``r1pro`` embodiment profile."""
        return load_embodiment(self.embodiment)

    def situation(self) -> str:
        """Profile situation plus the activity's instruction."""
        base = self.profile.render_situation().rstrip()
        instruction = self.backend.challenge.get("instruction", "")
        goal = f"\nBenchmark goal: {instruction}" if instruction else ""
        return base + goal + "\n"

    def facts(self) -> str:
        """From the embodiment profile."""
        return self.profile.render_facts()

    def goal(self) -> str:
        """The challenge's language instruction (empty for non-challenge activities)."""
        return self.backend.challenge.get("instruction", "")

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int) -> None:
        """Fresh episode of the presampled instance. ``seed`` selects no instance,
        but it seeds the RNGs the symbolic primitives sample poses from — that is
        what lets a saved program replay the same way in a fresh process."""
        self._seed = int(seed)
        self.backend.reset(self._seed)

    def close(self) -> None:
        self.backend.close()

    # ------------------------------------------------------------ evaluation

    def score(self) -> ScoreReport:
        """BDDL goal evaluation: reward 1.0 iff every goal condition holds."""
        report = self.backend.goal_report()
        satisfied, unsatisfied = len(report["satisfied"]), len(report["unsatisfied"])
        return ScoreReport(
            reward=1.0 if report["success"] else 0.0,
            raw_score=satisfied / max(1, satisfied + unsatisfied),
            tier=satisfied,
            completed=bool(report["success"]),
            detail={
                "goal": report,
                "activity": self.backend.activity_name,
                "instance": self.backend.activity_instance_id,
                "abstraction": "symbolic-primitives",
            },
        )

    def budget(self) -> Budget:
        return Budget(
            used=int(self.backend.steps_used),
            max=int(self.backend.max_steps),
            unit="action_frames",
        )

    def frames(self) -> list[Any]:
        return self.backend.frames

    def capture(self) -> Any | None:
        return self.backend.capture_frame()

    # ------------------------------------------------------------ tools: observing

    @tool
    def get_goal(self) -> dict:
        """The BDDL goal this episode is SCORED on, evaluated right now. Returns
        {"instruction", "conditions": [natural-language BDDL conditions],
        "satisfied": [...], "unsatisfied": [...], "success"} — indices refer to
        the conditions list. Read this before planning and after every placement;
        it is the judge, not your own guess.
        """
        return self.backend.goal_report()

    @tool
    def get_task_objects(self) -> dict:
        """The objects the goal talks about: BDDL name -> {"sim_name", "category",
        "position", "aabb": [lo, hi]} (metres, world frame, Z up). BDDL names
        (e.g. "can__of__soda.n.01_2") and sim names both work in every tool here.
        A None value means that scope entry is not a physical object.
        """
        return self.backend.task_objects()

    @tool
    def list_scene_objects(self, keyword: str = "") -> list[dict]:
        """Scene objects (name, category, position), filtered by a case-insensitive
        substring of either. The scene holds hundreds of objects; filter, don't dump.
        Capped at 80 entries.
        """
        out = []
        for obj in self.backend.env.scene.objects:
            category = getattr(obj, "category", "")
            if keyword and keyword.lower() not in obj.name.lower() and keyword.lower() not in category.lower():
                continue
            pos = obj.get_position_orientation()[0].cpu().numpy()
            out.append(
                {
                    "name": obj.name,
                    "category": category,
                    "position": [round(float(v), 3) for v in pos],
                }
            )
            if len(out) >= 80:
                break
        return out

    @tool
    def get_object_pose(self, name: str) -> dict:
        """Position and orientation (xyzw quaternion) of an object, plus its aabb."""
        obj = self.backend.resolve(name)
        pos, quat = (t.cpu().numpy() for t in obj.get_position_orientation())
        lo, hi = (t.cpu().numpy() for t in obj.aabb)
        return {
            "position": [round(float(v), 4) for v in pos],
            "quaternion_xyzw": [round(float(v), 4) for v in quat],
            "aabb": [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
        }

    @tool
    def check_state(self, kind: str, name: str, other: str | None = None) -> bool | None:
        """Probe one object state the way the judge sees it. Binary kinds need
        ``other``: inside, ontop, under, nextto, touching, covered, filled. Unary:
        open, toggled_on, folded, cooked, frozen, heated. Returns None when the
        object does not carry that state. E.g. check_state("inside",
        "can__of__soda.n.01_1", "ashcan.n.01_1").
        """
        if kind not in STATES:
            raise ValueError(f"unknown state {kind!r}; known: {sorted(STATES)}")
        return self.backend.check_state(kind, name, other)

    @tool
    def get_step_budget(self) -> dict:
        """Simulator steps used / available this episode. Every primitive costs
        steps (grasp ~100, placements ~150, settle as asked); the episode is over
        at the cap.
        """
        return {"used": int(self.backend.steps_used), "max": int(self.backend.max_steps)}

    # ------------------------------------------------------------ tools: acting

    @tool
    def grasp(self, name: str) -> dict:
        """Grasp an object with the nearer hand, walking to it first if needed.
        One object per hand pair: release() or place before the next grasp.
        Returns {"ok", "steps"} or {"ok": False, "errors": [why]} — a False is a
        real refusal (hand full, unreachable), not noise; change the plan.
        """
        return self.backend.run_primitive("grasp", name)

    @tool
    def place_on_top(self, target: str) -> dict:
        """Place the currently grasped object on top of ``target``."""
        return self.backend.run_primitive("place_on_top", target)

    @tool
    def place_inside(self, target: str) -> dict:
        """Place the currently grasped object inside ``target``. KNOWN LIMIT: the
        pose sampler often cannot find an "inside" pose for containers with a
        narrow mouth (errors with SAMPLING_ERROR) even when the placement is
        physically easy — release() near the container and use drop_into() then.
        """
        return self.backend.run_primitive("place_inside", target)

    @tool
    def drop_into(self, name: str, container: str) -> dict:
        """Put ``name`` into an open-top ``container`` by holding it centred above
        the mouth and letting it fall (40 settle steps). Works where
        place_inside's sampler fails. Refuses while ``name`` is grasped —
        release() first. Returns {"ok", "steps", "inside"}; "inside" is the
        judge's own predicate after settling.
        """
        return self.backend.drop_into(name, container)

    @tool
    def open_object(self, name: str) -> dict:
        """Open an articulated object (door, drawer, lid). Fails on objects with
        no openable joints — check_state("open", name) returning None is the tell.
        """
        return self.backend.run_primitive("open", name)

    @tool
    def close_object(self, name: str) -> dict:
        """Close an articulated object."""
        return self.backend.run_primitive("close", name)

    @tool
    def toggle_on(self, name: str) -> dict:
        """Toggle an appliance on (radio, stove, faucet...)."""
        return self.backend.run_primitive("toggle_on", name)

    @tool
    def toggle_off(self, name: str) -> dict:
        """Toggle an appliance off."""
        return self.backend.run_primitive("toggle_off", name)

    @tool
    def cut(self, target: str) -> dict:
        """Slice or dice ``target`` with the currently grasped cutting tool (knife,
        axe...). Refuses unless a slicer is in hand and the target is sliceable
        or diceable. Slicing replaces the target with its halves (new objects
        named half__<x>...); dicing turns it into a particle pile. The judge's
        "real" conditions flip when the new objects exist — read get_goal().
        """
        return self.backend.run_primitive("cut", target)

    @tool
    def wipe(self, target: str) -> dict:
        """Wipe ``target`` with the currently grasped cloth/sponge/broom to clear
        what covers it. Refuses if nothing suitable is in hand.
        """
        return self.backend.run_primitive("wipe", target)

    @tool
    def soak_under(self, source: str) -> dict:
        """Hold the grasped object under a running fluid ``source`` (a toggled-on
        faucet) so it becomes soaked.
        """
        return self.backend.run_primitive("soak_under", source)

    @tool
    def soak_inside(self, container: str) -> dict:
        """Dip the grasped object into the fluid inside ``container``."""
        return self.backend.run_primitive("soak_inside", container)

    @tool
    def release(self) -> dict:
        """Let go of whatever is grasped, where the hand currently is."""
        return self.backend.run_primitive("release")

    @tool
    def settle(self, steps: int = 30) -> dict:
        """Step the world with no commands so physics comes to rest (and the
        judge sees the settled state). Costs ``steps`` from the budget.
        """
        steps = max(1, min(int(steps), 200))
        return self.backend.settle_report(steps)

    # primitives table is public API surface; keep names in sync with the backend
    assert set(PRIMITIVES) >= {"grasp", "place_on_top", "place_inside", "open", "close", "release"}
