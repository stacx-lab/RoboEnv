"""LIBERO-PRO (single Franka Panda, MuJoCo) — bridges the cap-x integration.

Runs inside ``.venv-libero``.  The privileged API plans through the PyRoKi
server (port 8116) — start cap-x's api servers first (8114 SAM3 / 8115
graspnet / 8116 PyRoKi).

Pick tasks with ``suite_name`` + ``task_id`` (suites: ``libero_spatial``,
``libero_object``, ``libero_goal``, ``libero_10``, ``libero_90``).  The
benchmark's language goal is appended to the situation after reset.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from roboenv_runtime.adapters.base import CapxBridge
from roboenv_runtime.adapters.libero_primitives import _quat_wxyz_to_mat

# pyroki's IK runs on jaxls, which orders residual groups by object address; make the
# order deterministic before any solve happens (roboenv/compat/jaxls_order.py has the
# measurement).  Harmless where jaxls is absent.
from roboenv_runtime.compat.jaxls_order import make_ik_deterministic
from roboenv_runtime.core.adapter import register_adapter
from roboenv_runtime.core.types import Capability
from roboenv_runtime.embodiments import load_embodiment

if os.environ.get("ROBOENV_DETERMINISTIC_IK", "1") != "0":  # =0: the old, layout-dependent runs
    make_ik_deterministic()

__all__ = ["LiberoAdapter"]

BASE_CONFIG = "env_configs/libero/franka_libero_spatial_0_privileged.yaml"

_CONFIG_KEYS = {
    "benchmark_root": ".",
    "bddl_files": "bddl_files",
    "init_states": "init_files",
    "assets": "assets",
}


def ensure_libero_config(config_dir: Path | None = None) -> Path:
    """Point LIBERO at the vendored ``LIBERO-PRO`` checkout before it is imported.

    LIBERO resolves its bddl/init-state/asset directories through
    ``$LIBERO_CONFIG_PATH/config.yaml`` (default ``~/.libero``) and, when that
    file is missing, *prompts on stdin* — fatal inside a worker.  A config
    written on another machine silently carries dead absolute paths.  roboenv
    therefore keeps its own config under ``<repo>/.cache/libero`` (or the
    directory ``LIBERO_CONFIG_PATH`` already names) and rewrites it whenever a
    listed path no longer exists.  Returns the config directory.
    """
    import yaml

    from roboenv_runtime.paths import capx_root, repo_root

    if config_dir is None:
        env_dir = os.environ.get("LIBERO_CONFIG_PATH")
        config_dir = Path(env_dir) if env_dir else repo_root() / ".cache" / "libero"
    config_dir = config_dir.expanduser().resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    root = capx_root() / "capx" / "third_party" / "LIBERO-PRO" / "libero" / "libero"
    if not root.is_dir():
        return config_dir  # nothing to bootstrap from; LIBERO will use its own defaults
    wanted = {key: str((root / rel).resolve()) for key, rel in _CONFIG_KEYS.items()}
    wanted["datasets"] = str(config_dir / "datasets")
    config_file = config_dir / "config.yaml"
    current: dict[str, str] = {}
    if config_file.exists():
        try:
            current = dict(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
        except (OSError, ValueError, yaml.YAMLError):
            current = {}
    stale = any(not Path(str(current.get(k, ""))).exists() for k in _CONFIG_KEYS)
    if stale or "datasets" not in current:
        config_dir.mkdir(parents=True, exist_ok=True)
        Path(wanted["datasets"]).mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump({**current, **wanted}), encoding="utf-8")
    return config_dir


@register_adapter
class LiberoAdapter(CapxBridge):
    """LIBERO task ``suite_name``/``task_id`` with the ``FrankaLiberoPrivilegedApi``."""

    name: ClassVar[str] = "libero"
    embodiment: ClassVar[str] = "panda"
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {Capability.PRIVILEGED, Capability.VIDEO, Capability.CHECKPOINT}
    )

    def __init__(
        self,
        suite_name: str = "libero_spatial",
        task_id: int = 0,
        init_order: str = "capx",
        **_: Any,
    ):
        self.suite_name, self.task_id = suite_name, int(task_id)
        if init_order not in ("capx", "benchmark"):
            raise ValueError("init_order must be 'capx' or 'benchmark'")
        self.init_order = init_order
        ensure_libero_config()
        super().__init__(
            BASE_CONFIG,
            overrides={
                "cfg.low_level.suite_name": suite_name,
                "cfg.low_level.task_id": int(task_id),
            },
        )

    def situation(self) -> str:
        """The panda profile's situation plus the benchmark's language goal."""
        base = load_embodiment(self.embodiment).render_situation().rstrip()
        goal = self.goal()
        return base + (f"\nBenchmark goal: {goal}" if goal else "") + "\n"

    def facts(self) -> str:
        """From the embodiment profile."""
        return load_embodiment(self.embodiment).render_facts()

    def _sim(self) -> Any:
        handle = self._lle.handle
        return getattr(getattr(handle, "env", handle), "sim", None)

    def reset(self, seed: int) -> None:
        """Reset the episode; also forget the IK warm start of the previous one.

        pyroki's ``api.cfg`` seed used to outlive ``reset``: the first ``goto`` of the next
        in-process episode then started from the previous episode's last posture, which
        is why two identical reset-and-pick runs in one process gave different descent
        residuals (0.030 vs 0.060 m, round-3 review).  A fresh process has no seed, so
        this only makes in-process episodes behave like fresh ones.
        """
        super().reset(seed)
        if self.init_order == "benchmark":
            self._apply_benchmark_init_state(seed)
        prims = getattr(self, "_prims", None)
        if prims is not None:
            with contextlib.suppress(Exception):
                prims.api.cfg = None

    def _apply_benchmark_init_state(self, seed: int) -> None:
        """Re-apply the benchmark's own init state AFTER cap-x's reset (opt-in,
        ``init_order="benchmark"``).

        cap-x resets in the wrong order: ``handle.reset`` -> ``set_init_state(init[i])``
        -> ``env.reset()``, and that last ``env.reset()`` re-samples every placement and
        discards the init state — so every "seed 0" layout the certificates were earned
        on is cap-x's own sample, not the benchmark's ``init_states[(seed - 1) % N]``, and
        on ``libero_10_with_milk/2`` the sample ejects the basket 3.7 m and the milk 11 m
        up.  The correct order (verified 2026-08-27) is ``env.reset()`` then
        ``set_init_state``.  This is NOT the default: every stored certificate was earned
        on the cap-x layout and switching the default decertifies them all — the
        integrator flips it per task (``adapter_kwargs: {init_order: benchmark}``) or
        globally with a regression.  The benchmark layout is applied here, then the same
        10-step settle cap-x runs, and the step counters/IK seed are set to what a plain
        cap-x reset leaves behind (cap-x zeroes them BEFORE its settle, so a fresh episode
        starts at ``_sim_step_count == 10``; zeroing them after the settle used to hand
        this path 10 free steps against the 4000 horizon).  A task with no ``init_states``
        raises rather than silently falling back to the cap-x layout — the whole point of
        the switch is knowing which layout a certificate was earned on.
        """
        lle = self._lle
        handle = lle.handle
        states = getattr(handle, "init_states", None)
        if states is None or len(states) == 0:
            raise RuntimeError(
                "init_order='benchmark' but this task has no init_states: the benchmark "
                "layout cannot be applied (use init_order='capx')"
            )
        idx = (int(seed) - 1) % len(states)
        handle.env.set_init_state(states[idx])
        with contextlib.suppress(Exception):
            handle.env.sim.forward()
        lle._current_joints = handle.env.sim.data.qpos[:7].copy()
        lle._gripper_fraction = 1.0
        lle._step_count = 0
        lle._sim_step_count = 0
        for _ in range(10):
            lle._step_once()
        with contextlib.suppress(Exception):
            obs = lle.get_observation()
            lle.home_joint_position = np.array(obs["robot0_joint_pos"], dtype=np.float64)
        with contextlib.suppress(Exception):
            lle.gripper_link_wxyz_xyz = np.concatenate(
                [
                    handle.env.sim.data.xquat[lle.gripper_link_idx],
                    handle.env.sim.data.xpos[lle.gripper_link_idx],
                ]
            )

    def _primitives(self) -> Any:
        """The one :class:`~roboenv_runtime.adapters.libero_primitives.LiberoPrimitives` of this
        adapter (checkpoints and the IK loop must not be split across two instances).
        """
        from roboenv_runtime.adapters.libero_primitives import LiberoPrimitives

        cached = getattr(self, "_prims", None)
        if cached is None:
            cached = LiberoPrimitives(self)
            self._prims = cached
        return cached

    def api_functions(self) -> dict[str, Callable[..., Any]]:
        """The cap-x API, with the three raw motion calls made honest.

        ``goto_pose`` upstream is a single velocity-limited IK solve that reports nothing:
        it lands 5-16 cm short of the request whenever the arm has to travel, and the
        program only finds out when the grasp closes on air.  Here it MEASURES the result
        and returns it, without changing the motion — the one-shot undershoot is load
        bearing for programs written against it (routing this name through the closed loop
        moved ``libero.goal_2``'s depth-estimated placement and decertified it), so the
        closed-loop version lives beside it as ``goto_pose_closed``.
        ``open_gripper``/``close_gripper``/``goto_pose`` also refuse to step past LIBERO's
        hard 4000-step horizon (the 4001st step raises inside the simulator and the whole
        program dies with its achievements discarded).
        """
        fns = dict(super().api_functions())
        prim = self._primitives()

        def _achieved(pos: Any, quat: Any, exhausted: bool) -> dict[str, Any]:
            rot = _quat_wxyz_to_mat(quat)
            _, hand = prim.hand_pose()
            dev = _quat_wxyz_to_mat(hand).T @ rot
            angle = float(np.degrees(np.arccos(np.clip((np.trace(dev) - 1.0) / 2.0, -1.0, 1.0))))
            error = float(np.linalg.norm(np.asarray(pos, dtype=float) - prim.tcp()))
            return {
                "position_error_m": round(error, 4),
                "orientation_dev_deg": round(angle, 2),
                "reached": bool(error < 0.01),
                "tcp": [round(float(v), 4) for v in prim.tcp()],
                "budget_exhausted": exhausted,
            }

        def goto_pose(
            position: Any, quaternion_wxyz: Any, z_approach: float = 0.0
        ) -> dict[str, Any]:
            """Move the fingertip midpoint to ``position`` (robot-base frame) with hand
            orientation ``quaternion_wxyz`` (wxyz): ONE velocity-limited IK solve per
            waypoint (two with ``z_approach``), exactly the upstream motion of up to 120
            controller steps each, but the result is measured and returned.

            It routinely lands 5-16 cm short when the arm has to travel, and it does not
            raise when it does — so read ``reached`` (position error < 1 cm; the
            orientation is reported in ``orientation_dev_deg``, not gated) before you
            close the gripper on what you think you are over.  Use ``goto_pose_closed``
            for a move that keeps re-solving until it arrives.  ``z_approach`` > 0 first
            stops that far back along the tool axis (an insertion approach).  Returns
            ``{"position_error_m", "orientation_dev_deg", "reached", "tcp",
            "budget_exhausted"}``.  The controller run is clamped to the steps left
            before LIBERO's hard horizon and ``budget_exhausted`` means this call was
            refused or truncated by it (the same meaning in every tool here).
            """
            pos = np.asarray(position, dtype=float).reshape(3)
            quat = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
            api = prim.api
            rot = _quat_wxyz_to_mat(quat)
            offset_pos = pos + rot @ np.asarray(api._TCP_OFFSET, dtype=float)
            waypoints = [offset_pos]
            if float(z_approach) != 0.0:
                waypoints.insert(0, offset_pos + rot @ np.array([0.0, 0.0, -float(z_approach)]))
            truncated = False
            for hand_target in waypoints:
                left = prim._steps_left()
                if left <= 1:
                    truncated = True
                    break
                # the upstream solve, verbatim (one velocity-limited IK step) ...
                if api.cfg is None:
                    api.cfg = api._pks.solve_ik(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=hand_target,
                        target_wxyz=quat,
                    )
                else:
                    api.cfg = api._pks.solve_ik_vel_cost(
                        robot=api._robot,
                        target_link_name=api._target_link_name,
                        target_position=hand_target,
                        target_wxyz=quat,
                        prev_cfg=api.cfg,
                    )
                joints = np.asarray(api.cfg[:-1], dtype=np.float64).reshape(7)
                # ... with the controller run clamped to the horizon (upstream would step
                # 120 times regardless and the 4001st step kills the episode)
                prim.lle.move_to_joints_blocking(joints, max_steps=min(120, left - 1))
                truncated = truncated or left - 1 < 120
            return _achieved(pos, quat, truncated)

        def goto_pose_closed(
            position: Any, quaternion_wxyz: Any, z_approach: float = 0.0
        ) -> dict[str, Any]:
            """``goto_pose`` with the loop closed: measure the fingertips, add the residual
            to the commanded target, solve again, until it arrives or stops closing in.

            This is what the primitives use, and what you want for anything you are about
            to grasp or place — one IK solve is smoothness-limited and undershoots.  It
            costs more steps (50-250 against 50-90) and it will PUSH whatever is in the
            way, so keep the one-shot ``goto_pose`` for motions meant to be resisted.
            With ``z_approach`` it runs two closed loops (up to 2 x 8 x 80 steps).  Same
            report as ``goto_pose``; ``budget_exhausted`` means the loop was refused or
            cut by the horizon.
            """
            pos = np.asarray(position, dtype=float).reshape(3)
            quat = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
            if prim._steps_left() <= 1:
                return _achieved(pos, quat, True)
            rot = _quat_wxyz_to_mat(quat)
            if float(z_approach) != 0.0:
                prim.goto(pos + rot @ np.array([0.0, 0.0, -float(z_approach)]), quat)
            prim.goto(pos, quat)
            # goto stops early when the horizon is within reach: that IS a truncation
            return _achieved(pos, quat, prim._steps_left() <= 1)

        def open_gripper(gap_m: float | None = None) -> dict[str, Any]:
            """Open the gripper fully (40 simulator steps), unless that would cross
            LIBERO's hard horizon.  Returns ``{"opening_m", "budget_exhausted"}``;
            ``budget_exhausted`` True means the call was REFUSED (nothing moved) — a
            ``goto_pose`` that reported False can still be followed by a refused gripper
            call when fewer than 40-60 steps are left, so check ``get_step_budget``.

            With ``gap_m`` the jaws stop at that measured finger gap instead of at the
            full 0.077 m and STAY there through the following motions (a later
            ``open_gripper()`` or ``close_gripper()`` releases the hold).  Partial opening
            is an ENVELOPE tool: an open jaw is ``gap_m + 0.053`` m wide across its closing
            axis, so a tool that has to enter a slot — the bottom drawer's handle bar of
            libero.90_6, 3.3 cm under the middle drawer's face — needs a small ``gap_m``
            to fit at all, and the pinch that follows only has to close the rest of the way.
            """
            done = prim.open_fingers(gap_m)
            return {
                "opening_m": round(prim.get_gripper_opening(), 4),
                "budget_exhausted": not done,
            }

        def close_gripper() -> dict[str, Any]:
            """Close the gripper fully (60 simulator steps), unless that would cross
            LIBERO's hard horizon.  ``opening_m`` above ~0.004 means the fingers met
            material; ~0.0 means they closed on nothing.
            """
            done = prim.close_fingers()
            return {
                "opening_m": round(prim.get_gripper_opening(), 4),
                "budget_exhausted": not done,
            }

        fns.update(
            {
                "goto_pose": goto_pose,
                "goto_pose_closed": goto_pose_closed,
                "open_gripper": open_gripper,
                "close_gripper": close_gripper,
            }
        )
        return fns

    def extra_tools(self) -> dict[str, Callable[..., Any]]:
        """Articulation/sub-body observations plus the closed-loop primitives
        (:mod:`roboenv_runtime.adapters.libero_primitives`) the planner API lacks.
        """
        primitives = self._primitives()

        def get_joint_states() -> dict:
            """Joint positions of ARTICULATED scene parts: drawers, doors,
            buttons/knobs (e.g. a cabinet's drawer slides, a stove's burner
            button). Returns {joint_name: value}. A zero usually means
            closed/off; pulling a drawer or turning a knob changes it.
            Robot joints and free-floating objects are excluded.
            """
            sim = self._sim()
            out: dict[str, float] = {}
            if sim is None:
                return out
            model = sim.model
            for i in range(model.njnt):
                joint = model.joint_id2name(i)
                if not joint or "robot" in joint or "gripper" in joint:
                    continue
                q = np.atleast_1d(sim.data.get_joint_qpos(joint))
                if q.size == 1:  # hinge/slide = articulation
                    out[joint] = float(q[0])
            return out

        def get_body_positions(keyword: str = "") -> dict:
            """Positions (robot-base frame, same as get_object_pose) of scene SUB-BODIES
            that have no entry in list_objects(): drawer panels, handles, burners,
            buttons. Optional case-insensitive substring filter (e.g. "drawer",
            "button"). Use these as motion targets for articulated parts.
            """
            sim = self._sim()
            out: dict[str, list[float]] = {}
            if sim is None:
                return out
            model = sim.model
            for i in range(model.nbody):
                body = model.body_id2name(i)
                if not body or "robot" in body or "gripper" in body or body == "world":
                    continue
                if keyword and keyword.lower() not in body.lower():
                    continue
                world = sim.data.get_body_xpos(body)
                out[body] = [float(v) for v in primitives.base_from_world(world)]
            return out

        def get_goal_predicates() -> dict[str, Any]:
            """The BDDL goal this episode is SCORED on, parsed — the ground truth the
            language instruction paraphrases.  Returns ``{"predicates": [[pred, arg, ...],
            ...], "language", "problem"}``: e.g. ``[["on", "moka_pot_1",
            "flat_stove_1_cook_region"]]``, ``[["in", "akita_black_bowl_1",
            "wooden_cabinet_1_top_region"], ["open", "wooden_cabinet_1_top_region"]]``,
            ``[["turnon", "flat_stove_1"]]``.  Args name object INSTANCES (``_1``, ``_2``)
            and region SITES (``get_region_boxes``) or fixture/joint owners; every goal
            predicate must hold at once.  ``on``/``in`` compare the object's BODY ORIGIN
            with the site box (see ``get_region_boxes``), ``on`` between two objects needs
            contact and origins within 3 cm; ``open``/``close``/``turnon`` are joint
            thresholds (``get_joint_info``).  Read this before planning instead of guessing
            which instance or compartment the English names.  ``found`` is False (with
            ``reason``) when the parsed problem could not be reached — an empty
            ``predicates`` then means "not read", not "no goal".
            """
            env = getattr(self._lle, "handle", None)
            env = getattr(env, "env", env)
            for _ in range(6):
                if env is None or hasattr(env, "parsed_problem"):
                    break
                env = getattr(env, "env", None)
            problem = getattr(env, "parsed_problem", None)
            if problem is None:
                return {
                    "found": False,
                    "predicates": [],
                    "language": self.goal(),
                    "problem": None,
                    "reason": "no parsed_problem on the env wrapper chain",
                }
            goal = problem.get("goal_state") or []
            return {
                "found": True,
                "predicates": [[str(x) for x in g] for g in goal],
                # LIBERO's robosuite_parse_problem writes `language_instruction`;
                # `language` was always missing, so this silently returned the task YAML
                "language": problem.get("language_instruction")
                or problem.get("language")
                or self.goal(),
                "problem": problem.get("problem_name"),
            }

        tools: dict[str, Callable[..., Any]] = {
            "get_joint_states": get_joint_states,
            "get_body_positions": get_body_positions,
            "get_goal_predicates": get_goal_predicates,
        }
        tools.update(primitives.tools())
        return tools
