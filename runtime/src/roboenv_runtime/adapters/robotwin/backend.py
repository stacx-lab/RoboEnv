"""RoboTwin dual-arm (Piper) low-level environment wrapper.

Wraps RoboTwin 2.0 bimanual tasks (SAPIEN 3, Vulkan ray-tracing renderer)
through the native SAPIEN API. Requires the ``.venv-robotwin`` environment;
in other venvs the import fails and the registration guard skips this family.

Two control routes are exposed to the API layer:

- Route A (planner-level): RoboTwin's own motion layer — ``plan_move_to_pose``,
  ``plan_grasp_object`` etc. drive mplib/curobo planned trajectories.
- Route B (joint-level): ``move_to_joints_blocking`` sets PD drive targets
  directly, mirroring the Franka/LIBERO envs, for use with external IK (pyroki).

RoboTwin resolves many asset paths relative to the CWD ("./assets/..."), so
this module chdirs the worker process into the vendored RoboTwin root at env
construction. Use absolute ``output_dir`` paths in capx YAML configs.
"""

# Moved from third_party/cap-x/capx/envs/simulators/robotwin.py on 2026-08-26 (refactor P2); cap-x keeps a shim.

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import transforms3d as t3d
import yaml

from roboenv_runtime.paths import robotwin_root

ROBOTWIN_ROOT = str(robotwin_root())  # $ROBOTWIN_ROOT or the selected RoboTwin checkout
if not os.path.isdir(ROBOTWIN_ROOT):
    raise ModuleNotFoundError(f"RoboTwin not found at {ROBOTWIN_ROOT}")
if ROBOTWIN_ROOT not in sys.path:
    sys.path.insert(0, ROBOTWIN_ROOT)

import sapien  # noqa: E402, F401  (probe: fails fast outside .venv-robotwin)

# RoboTwin's envs.utils reads "./assets/..." at IMPORT time and everywhere else
# relative to the CWD, so importing it requires chdir'ing into the RoboTwin
# root. Defer both until an env is actually constructed: the launch process can
# then still resolve its own relative paths (config, output dir) while workers
# get chdir'ed when they build the env.
Action = None  # populated by _ensure_robotwin_imported
ArmTag = None
Actor = None
_ROBOTWIN_IMPORTED = False


def _ensure_robotwin_imported() -> None:
    global _ROBOTWIN_IMPORTED, Action, ArmTag, Actor
    if _ROBOTWIN_IMPORTED:
        os.chdir(ROBOTWIN_ROOT)  # keep CWD stable for repeated env builds
        return
    os.chdir(ROBOTWIN_ROOT)
    from envs.utils.action import Action as _Action  # noqa: PLC0415
    from envs.utils.action import ArmTag as _ArmTag  # noqa: PLC0415
    from envs.utils.actor_utils import Actor as _Actor  # noqa: PLC0415

    Action, ArmTag, Actor = _Action, _ArmTag, _Actor
    _ROBOTWIN_IMPORTED = True


_CONFIGS_PATH = os.path.join(ROBOTWIN_ROOT, "env_cfg", "task_config")

Side = Literal["left", "right"]

#: RoboTwin's ``delay()`` counts simulator steps at 250 Hz; ~0.15 s of settling.
SETTLE_STEPS = 40
#: Length of the tool along its local +x, from the commanded end-effector
#: ("endpose") frame to the point between the fingertips.  Measured on both
#: Piper arms: ``get_tcp_pose() - get_gripper_pose_*() == 0.12 * R @ [1,0,0]``.
TOOL_LENGTH_M = 0.12


class PlanningFailed(RuntimeError):
    """RoboTwin's planner produced no motion (its sticky ``plan_success`` dropped)."""


def _as_id_list(contact_point_id: Any) -> list[int] | None:
    """Normalise a ``contact_point_id`` argument to a list of ids, or None when
    the caller named no point at all (search every annotated point).

    Ids must be integers (a bool or a 3.7 used to be truncated silently); duplicates
    are dropped so a repeated id does not cost the budget twice.
    """
    if contact_point_id is None:
        return None
    raw = (
        list(contact_point_id)
        if isinstance(contact_point_id, (list, tuple, set))
        else [contact_point_id]
    )
    ids: list[int] = []
    for c in raw:
        if isinstance(c, bool) or (not isinstance(c, (int, np.integer)) and float(c) != int(c)):
            raise ValueError(f"contact point ids must be integers, got {c!r}")
        if int(c) not in ids:
            ids.append(int(c))
    return ids


def list_robotwin_tasks() -> list[str]:
    """Enumerate RoboTwin task names by scanning the vendored envs directory."""
    envs_dir = os.path.join(ROBOTWIN_ROOT, "envs")
    return sorted(
        f[:-3]
        for f in os.listdir(envs_dir)
        if f.endswith(".py") and not f.startswith("_") and f != "robot"
    )


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _assemble_task_args(
    task_name: str,
    task_config: str,
    embodiment: list[Any],
    planner: str | None,
    collect_wrist_camera: bool,
    save_freq: int,
) -> dict[str, Any]:
    """Replicates RoboTwin's collect_data.py config assembly, with absolute paths."""
    args = _load_yaml(os.path.join(_CONFIGS_PATH, f"{task_config}.yml"))
    embodiment_registry = _load_yaml(os.path.join(_CONFIGS_PATH, "_embodiment_config.yml"))

    def embodiment_file(name: str) -> str:
        file_path = embodiment_registry[name]["file_path"]
        return os.path.normpath(os.path.join(ROBOTWIN_ROOT, file_path))

    def embodiment_config(robot_file: str) -> dict[str, Any]:
        cfg = _load_yaml(os.path.join(robot_file, "config.yml"))
        if planner is not None:
            cfg["planner"] = planner
        return cfg

    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = float(embodiment[2])
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must be [name] or [left, right, distance]")

    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = (
        str(embodiment[0]) if len(embodiment) == 1 else f"{embodiment[0]}+{embodiment[1]}"
    )

    args["task_name"] = task_name
    args["save_path"] = None
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_video_save_dir"] = None
    args["render_freq"] = 0
    args["save_freq"] = save_freq
    args["need_plan"] = True

    args["camera"]["collect_wrist_camera"] = collect_wrist_camera
    args["data_type"]["rgb"] = True
    args["data_type"]["depth"] = True
    args["data_type"]["qpos"] = True
    args["data_type"]["endpose"] = True
    return args


class RoboTwinEnv:
    """A source-backed RoboTwin task, independent of any agent harness."""

    def __init__(
        self,
        task_name: str = "handover_block",
        task_config: str = "demo_clean",
        embodiment: list[Any] | None = None,
        planner: str | None = "mplib_RRT",
        privileged: bool = True,
        max_steps: int = 60000,
        seed: int | None = None,
        enable_render: bool = False,
        viser_debug: bool = False,
        collect_wrist_camera: bool = True,
        save_freq: int = 25,
        task_module_path: str | None = None,
        auto_reset: bool = True,
        allow_seed_substitution: bool = True,
    ) -> None:
        self.privileged = privileged
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render
        self.viser_debug = viser_debug  # not supported yet for RoboTwin
        self.task_name = task_name
        self.allow_seed_substitution = allow_seed_substitution

        _ensure_robotwin_imported()  # chdirs this process into the RoboTwin root

        self._base_args = _assemble_task_args(
            task_name=task_name,
            task_config=task_config,
            embodiment=embodiment if embodiment is not None else ["piper", "piper", 0.6],
            planner=planner,
            collect_wrist_camera=collect_wrist_camera,
            save_freq=save_freq,
        )

        if task_module_path is not None:
            # Generated tasks (taskgen) live outside RoboTwin/envs; load by path.
            # Must come after _ensure_robotwin_imported(): the module does
            # "from envs._base_task import ..." which needs the chdir/sys.path.
            module_name = f"taskgen_generated.{task_name}"
            spec = importlib.util.spec_from_file_location(module_name, task_module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load task module from {task_module_path}")
            task_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = task_module
            spec.loader.exec_module(task_module)
        else:
            task_module = importlib.import_module(f"envs.{task_name}")
        self._task_cls = getattr(task_module, task_name)
        self.task = None

        self._env_active = False
        self._step_count = 0
        self._sim_step_count = 0
        self._subsample_rate = 8

        self._planner_tracked: dict[str, dict[str, Any]] = {}
        self._record_frames = False
        self._record_wrist_camera = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []

        if auto_reset:
            self.reset(seed=seed)

    # ------------------------------ lifecycle ------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is None:
            seed = self.seed if self.seed is not None else 0
        self._close_task()

        # RoboTwin scene generation rejects some seeds outright (UnStableError:
        # randomized objects never settle). Upstream's own data pipeline skips
        # such seeds and moves to the next episode number, so mirror that:
        # bump the seed until a stable scene builds (bounded), and record the
        # seed actually used.
        # RoboTwin resolves assets relative to CWD; harness code may have
        # chdir'ed elsewhere (e.g. a per-trial agent workspace), so pin the
        # cwd for the whole scene build and restore it afterwards
        _prev_cwd = os.getcwd()
        os.chdir(ROBOTWIN_ROOT)
        try:
            return self._reset_inner(seed)
        finally:
            os.chdir(_prev_cwd)

    def _reset_inner(self, seed: int):
        from envs.utils.create_actor import UnStableError

        last_exc: Exception | None = None
        attempts = []
        self._initialization_steps_total = 0
        started = time.monotonic()
        for bump in range(20 if self.allow_seed_substitution else 1):
            try_seed = seed + bump
            done = False
            # transient asset/scene construction failures happen under heavy
            # concurrent startup (observed: create_actor returning None) —
            # retry the same seed once before treating it as fatal
            for attempt in range(2):
                self.task = self._new_task()
                args = deepcopy(self._base_args)
                try:
                    self.task.setup_demo(now_ep_num=try_seed, seed=try_seed, is_test=True, **args)
                    attempts.append({"seed": try_seed, "attempt": attempt, "success": True})
                    done = True
                    break
                except UnStableError as exc:
                    attempts.append({"seed": try_seed, "attempt": attempt, "error": type(exc).__name__})
                    last_exc = exc
                    self._close_task()
                    print(f"[robotwin] seed {try_seed} unstable, trying {try_seed + 1}", flush=True)
                    break
                except Exception as exc:
                    attempts.append({"seed": try_seed, "attempt": attempt, "error": type(exc).__name__})
                    last_exc = exc
                    self._close_task()
                    if attempt == 0:
                        print(
                            f"[robotwin] scene construction failed ({type(exc).__name__}: {exc}) — retrying once",
                            flush=True,
                        )
                        continue
                    raise
            if done:
                break
        else:
            raise last_exc
        # Upstream's own policy-eval protocol (scripts/eval_policy_*.py) runs
        # setup_demo -> play_once() -> setup_demo again: the expert dry-run
        # populates evaluation attributes that check_success reads but only
        # play_once assigns (e.g. self.arm_tag), and the second setup rebuilds
        # a fresh scene for the policy. Mirror it exactly — without this,
        # some tasks' check_success raises forever and can never pass.
        expert_error = None
        before_expert = self._initialization_steps_total
        try:
            self.task.play_once()
        except Exception as exc:
            expert_error = type(exc).__name__ + ": " + str(exc)
        expert_plan_success = bool(getattr(self.task, "plan_success", False))
        expert_steps = self._initialization_steps_total - before_expert
        args = deepcopy(self._base_args)
        args["eval_mode"] = getattr(self, "evaluation_mode", False)
        self.task.setup_demo(now_ep_num=try_seed, seed=try_seed, is_test=True, **args)
        self.actual_seed = try_seed
        self.initialization = {
            "requested_seed": seed, "effective_seed": try_seed, "attempts": attempts,
            "seed_substitution_allowed": self.allow_seed_substitution,
            "expert_prepass": True, "expert_error": expert_error,
            "expert_plan_success": expert_plan_success,
            "physics_steps": self._initialization_steps_total,
            "expert_physics_steps": expert_steps,
            "wall_time_s": time.monotonic() - started,
        }
        self._env_active = True

        # Video: motion loops call _take_picture() every save_freq sim steps;
        # replace it with our lightweight head-camera capture (save_data stays
        # False so no pkl dumping can trigger). The same hook doubles as a
        # periodic success poll for tasks with momentary success events
        # (opt-in via a LATCH_SUCCESS class attribute on the task).
        self.task._take_picture = self._video_tick
        self._success_latched = False
        self._poll_success = bool(getattr(self.task, "LATCH_SUCCESS", False))

        self._step_count = 0
        self._sim_step_count = 0
        # a checkpoint from a previous episode would unpack a foreign layout into this
        # scene without any error (measured 2026-08-27: an object teleported 0.5 m)
        self._checkpoints = {}
        self._ckpt_counter = 0
        # opt-in planning-world tracking: {side: {mplib name: sapien entity}}
        self._planner_tracked: dict[str, dict[str, Any]] = {}

        obs = self.get_observation()
        info = {"task_prompt": self.task_name.replace("_", " ")}
        return obs, info

    def _new_task(self):
        """Count this world's native initialization without patching global classes."""
        task = self._task_cls()
        setup_scene = task.setup_scene

        def setup_counted_scene(**kwargs):
            setup_scene(**kwargs)
            step = task.scene.step

            def counted_step():
                step()
                self._initialization_steps_total += 1

            task.scene.step = counted_step

        task.setup_scene = setup_counted_scene
        return task

    def _close_task(self) -> None:
        if self.task is not None:
            try:
                self.task.close_env(clear_cache=False)
            except Exception:
                pass
        self._env_active = False

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level fallback step; control normally flows through the API layer."""
        self._step_count += 1
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.task_completed()
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    # ------------------------------ route A: planner-level ------------------------------

    def _arm_plan_flag(self) -> None:
        """Arm RoboTwin's sticky ``plan_success`` before a motion.

        ``Base_Task.move`` (and ``grasp_actor``/``place_actor``/
        ``get_grasp_pose``...) return immediately while ``plan_success`` is
        False, so ONE failed plan silently turns every later motion of the
        episode into a no-op.  ``_finish_plan`` re-arms the flag on the way
        out, but only when it is reached: an exception raised between the
        motion and it (a failed grasp verification, for one) used to leak the
        dirty flag into the next call, which then reported "the arm did not
        move" for a motion that was never attempted.  Measured 2026-08-27 on
        ``handover_mic``: one unplannable grasp poisoned the whole episode and
        every later call failed with a misleading message.  Arming on entry
        makes each call answer for itself.

        It is also where the planning world is refreshed, because every planner route in
        this backend calls it on entry — ``_run_plan``, ``_plan_path_status``, the grasp
        path (``_grasp_move``) and ``plan_drag_articulated``.  Putting the sync here
        instead of in ``_run_plan`` is what makes ``avoid_in_planner``'s "with its pose
        kept live" true on the grasp and drag routes too.
        """
        if self.task is not None:
            self.task.plan_success = True
        if self._planner_tracked:
            self._sync_planner_world()

    def _finish_plan(self, what: str) -> None:
        """Report sticky plan failures as exceptions and re-arm the flag."""
        ok = bool(self.task.plan_success)
        self.task.plan_success = True
        if not ok:
            carried = sorted(
                {name for objects in self._planner_tracked.values() for name in objects}
            )
            payload = (
                ""
                if not carried
                else (
                    f" NOTE: {carried} is declared in the planning world "
                    "(carry_in_planner / avoid_in_planner), so it is collision-checked "
                    "too and a motion that would drive it into something is refused "
                    "here rather than in the physics. Undeclare it if the contact is "
                    "what you want."
                )
            )
            raise PlanningFailed(
                f"Motion planning failed for {what}. The arm did not move. "
                "Try a nearby position or a different approach orientation." + payload
            )

    def _run_plan(self, what: str, thunk: Callable[[], Any]) -> None:
        """Arm the sticky flag, run ONE planner-executed motion, report its failure.

        Every planner-route wrapper goes through here (round 3): before, only the grasp
        path armed the flag on entry, so a leak from any other exception between a motion
        and its ``_finish_plan`` made the six other wrappers report "the arm did not move"
        for motions never attempted.
        """
        self._arm_plan_flag()  # arms the flag AND syncs tracked objects into the world
        thunk()
        self._finish_plan(what)

    @staticmethod
    def _to_pose7(position: Any, quat_wxyz: Any) -> list[float]:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        return [*pos.tolist(), *quat.tolist()]

    def plan_move_to_pose(self, side: Side, position: Any, quat_wxyz: Any) -> None:
        pose7 = self._to_pose7(position, quat_wxyz)
        self._run_plan(
            f"{side} arm move to {pose7[:3]}",
            lambda: self.task.move(self.task.move_to_pose(ArmTag(side), pose7)),
        )

    def plan_move_both(
        self,
        left_position: Any,
        left_quat_wxyz: Any,
        right_position: Any,
        right_quat_wxyz: Any,
    ) -> None:
        left = self.task.move_to_pose(ArmTag("left"), self._to_pose7(left_position, left_quat_wxyz))
        right = self.task.move_to_pose(
            ArmTag("right"), self._to_pose7(right_position, right_quat_wxyz)
        )
        # RoboTwin's together_move_to_pose keeps executing the arm that planned after the
        # other one failed, so "did not move" is only true for at least one arm here
        self._run_plan(
            "both arms simultaneous move (at least one arm's plan failed; the other may have "
            "moved)",
            lambda: self.task.move(left, right),
        )

    def plan_move_by_displacement(
        self,
        side: Side,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat_wxyz: Any | None = None,
        move_axis: Literal["world", "arm"] = "world",
    ) -> None:
        if move_axis not in ("world", "arm"):
            raise ValueError(f"move_axis must be 'world' or 'arm', got {move_axis!r}")
        if move_axis == "arm" and (float(x) != 0.0 or float(y) != 0.0):
            raise ValueError(
                "move_axis='arm' uses only z: RoboTwin moves the hand by -z along its own "
                "approach (finger) axis and ignores x and y; use move_axis='world' or "
                "glide_to_pose_* for lateral steps"
            )
        quat = None
        if quat_wxyz is not None:
            try:
                q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
            except ValueError as exc:
                raise ValueError("quaternion_wxyz must be 4 numbers [w, x, y, z]") from exc
            norm = float(np.linalg.norm(q))
            if not np.isfinite(norm) or norm < 1e-6:
                raise ValueError("quaternion_wxyz must be a non-zero, finite quaternion")
            quat = (q / norm).tolist()
        what = (
            f"{side} arm displacement z={z} along -approach axis"
            if move_axis == "arm"
            else f"{side} arm displacement ({x}, {y}, {z})"
        )
        self._run_plan(
            what,
            lambda: self.task.move(
                self.task.move_by_displacement(
                    ArmTag(side), x=x, y=y, z=z, quat=quat, move_axis=move_axis
                )
            ),
        )

    def set_gripper(self, side: Side, fraction: float) -> None:
        """Planner-executed gripper motion. fraction: 0.0 closed .. 1.0 open."""
        fraction = float(np.clip(fraction, 0.0, 1.0))
        # NOTE: Action(action="gripper") never assigns target_gripper_pos;
        # "open"/"close" both honor the explicit target, so pick by direction.
        kind = "open" if fraction >= 0.5 else "close"
        self._run_plan(
            f"{side} gripper -> {fraction}",
            lambda: self.task.move(
                (ArmTag(side), [Action(side, kind, target_gripper_pos=fraction)])
            ),
        )

    def plan_back_to_origin(self, side: Side) -> None:
        self._run_plan(
            f"{side} arm back to origin",
            lambda: self.task.move(self.task.back_to_origin(ArmTag(side))),
        )

    def plan_grasp_object(
        self,
        object_name: str,
        side: Side,
        pre_grasp_dis: float = 0.1,
        grasp_dis: float = 0.0,
        contact_point_id: int | None = None,
        verify: bool = True,
        search: bool = False,
    ) -> None:
        actor = self.get_actor(object_name)
        kwargs: dict[str, Any] = dict(pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis)
        if contact_point_id is not None:
            # an id outside the annotations used to be dropped silently by RoboTwin's
            # choose_grasp_pose (get_contact_point(99) -> None -> skipped)
            n_points = self._n_contact_points(actor)
            for cid in _as_id_list(contact_point_id) or []:
                if n_points and not 0 <= cid < n_points:
                    raise RuntimeError(
                        f"{object_name} has contact points 0..{n_points - 1}; got {cid}"
                    )
            kwargs["contact_point_id"] = contact_point_id
        try:
            self._grasp_move(actor, side, kwargs)
        except PlanningFailed as exc:
            # The chosen annotated pose planned no motion.  RoboTwin picks that
            # pose by ORIENTATION SCORE alone and never checks the descend, so
            # "unplannable" says nothing about the other annotated points.  Only
            # search when the caller asked: an unheralded ladder of 16 more
            # grasps would change the cost (and the outcome) of every stored
            # program that relies on this call failing fast.  Only a PLANNING
            # failure starts the search: a verification failure ("closed on
            # air") used to as well, and the retries then ran with the fingers
            # already closed (round-3 review).
            self._arm_plan_flag()
            if not search:
                raise
            used_cid, report = self._retry_annotated_grasp(
                actor, side, kwargs, ids=_as_id_list(contact_point_id)
            )
            if used_cid is None:
                raise RuntimeError(
                    f"grasp_object search over contact points {report['tried']} of "
                    f"{object_name} for the {side} arm found no plannable grasp: "
                    f"{report['reasons']}; the arm executed {report['motions']} approach "
                    "motion(s) on the way - read get_tcp_pose / the object pose before "
                    "continuing"
                ) from exc
            if verify:
                self._verify_grasp_contact(actor, side, used_cid)
            return
        except AssertionError as exc:
            # Some assets ship grasp annotations that yield no usable pose
            # for one arm/direction (upstream asserts "target_pose cannot be
            # None"). Their scripted experts never hit this; a free agent
            # does. Degrade generically instead of dying: first try each
            # annotated point explicitly, then synthesize a top-down pinch
            # from the actor's pose. No task- or asset-specific logic.
            self._arm_plan_flag()
            if "target_pose cannot be None" not in str(exc):
                raise
            if contact_point_id is not None:
                # the caller named the point: an unplannable one is an answer, not a
                # licence to pinch the object top-down (a fixture's top, for instance)
                raise RuntimeError(
                    f"contact point {contact_point_id} of {object_name} has no plannable "
                    f"grasp pose for the {side} arm; try another contact_point_id or arm"
                ) from exc
            used_cid, _report = self._retry_annotated_grasp(actor, side, kwargs)
            if used_cid is not None:
                if verify:
                    self._verify_grasp_contact(actor, side, used_cid)
                return
            self._synth_top_down_grasp(actor, side, pre_grasp_dis)
            return
        if verify:
            self._verify_grasp_contact(actor, side, contact_point_id)

    def _retry_annotated_grasp(
        self, actor: Any, side: Side, base_kwargs: dict, ids: list[int] | None = None
    ) -> tuple[int | None, dict[str, Any]]:
        """Try annotated contact points in turn.  Returns ``(cid, report)``: the id of
        the one that planned (so the caller can verify against ITS link), or None when
        none of them has a plannable grasp for this arm, and ``report`` =
        ``{"tried", "reasons", "motions"}`` (which ids were tried, why each failed, how
        many approach motions were EXECUTED on the way - a failed descend still runs its
        approach, so the arm does not end where it started).

        ``ids`` restricts the search to the caller's own points; None searches
        every point of the asset.  Goes through :meth:`_grasp_move`, which
        RAISES when the plan failed: the older loop called ``task.move``
        directly, and since that call only drops RoboTwin's ``plan_success``
        flag instead of raising, it accepted its first candidate whether or not
        the arm had moved and never reached the second one.  Ids outside the
        asset's annotated range raise (they used to be dropped silently).
        """
        n_points = self._n_contact_points(actor)
        if ids is None:
            ids = list(range(n_points))
        for cid in ids:
            if not 0 <= int(cid) < n_points:
                raise RuntimeError(
                    f"{actor.get_name()} has contact points 0..{n_points - 1}; got {cid}"
                )
        report: dict[str, Any] = {"tried": [], "reasons": {}, "motions": 0}
        for cid in ids:
            kw = dict(base_kwargs)
            kw["contact_point_id"] = cid
            report["tried"].append(int(cid))
            try:
                self._grasp_move(actor, side, kw)
                return int(cid), report
            except (PlanningFailed, AssertionError) as exc:
                self._arm_plan_flag()
                report["reasons"][int(cid)] = str(exc)[:90]
                if not isinstance(exc, AssertionError):
                    report["motions"] += 1  # the approach ran before the descend failed
                continue
        return None, report

    @staticmethod
    def _n_contact_points(actor: Any) -> int:
        """How many annotated contact points the asset carries (0 when none)."""
        try:
            key = actor.POINTS["contact"]
            return len(actor.config.get(key, []) or [])
        except Exception:  # assets without annotations
            try:
                return sum(1 for _ in actor.iter_contact_points())
            except Exception:
                return 0

    def _grasp_move(self, actor: Any, side: Side, kwargs: dict) -> None:
        """Run ONE annotated-grasp motion and report a planning failure as one.

        The order matters: RoboTwin's ``move()`` is a no-op once ``plan_success``
        is False, so an unplannable grasp leaves the fingers wherever they were.
        Checking the contact first would call that "closed on air" — a true but
        useless diagnosis that also hides the sticky flag from the caller.
        ``_finish_plan`` is therefore called BEFORE any verification: it raises
        the planning failure and re-arms the flag so the next call is honest.
        """
        self._arm_plan_flag()
        self.task.move(self.task.grasp_actor(actor, arm_tag=ArmTag(side), **kwargs))
        self._finish_plan(f"grasp {actor.get_name()} with {side} arm")

    # ---------------- contact identity (who is really touching what) ----------------
    #
    # RoboTwin's own ``get_gripper_actor_contact_position`` matches contacts by
    # entity NAME.  Measured on the dual Piper (2026-08-27): BOTH arms' finger
    # links are named ``link7``/``link8`` and an articulation's links are named
    # ``link_0``..``link_3``, so that query cannot tell the left hand from the
    # right one, and it reports "the cabinet is grasped" when the fingers rest
    # on the cabinet's frame.  Everything below matches by ``Entity``
    # ``per_scene_id`` instead, which is unique per entity in the scene.

    #: a contact point with zero impulse counts as touching only within this separation:
    #: PhysX pairs appear inside the shapes' contact offsets (0.01 m each on the Piper),
    #: an open gripper over a box reads +12 mm / 0 impulse, a squeeze -0.9..+0.02 mm with
    #: impulse 0.12-0.14 (measured 2026-08-27, put_object_cabinet seed 0)
    CONTACT_SEPARATION_M = 1e-3

    def _gripper_link_ids(self, side: Side) -> dict[int, str]:
        """``{per_scene_id: link_name}`` of the finger links of THIS arm only.

        Read off the arm's own articulation (``robot.left_entity`` /
        ``robot.right_entity``), so the ids belong to that arm even though the
        two arms' links share names.
        """
        if side not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {side!r}")
        robot = self.task.robot
        entity = robot.left_entity if side == "left" else robot.right_entity
        joints = robot.left_gripper if side == "left" else robot.right_gripper
        fixed = (
            robot.left_fix_gripper_name if side == "left" else robot.right_fix_gripper_name
        ) or []
        wanted = {j[0].child_link.get_name() for j in joints} | set(fixed)
        out: dict[int, str] = {}
        for link in entity.get_links():
            if link.get_name() in wanted:
                out[int(link.entity.per_scene_id)] = link.get_name()
        return out

    def _actor_link_ids(self, actor: Any, link_names: list[str] | None = None) -> dict[int, str]:
        """``{per_scene_id: link_name}`` for the target.  A rigid actor is one
        entity; an articulation is one entity per link, and ``link_names``
        narrows it to those links."""
        inner = getattr(actor, "actor", actor)
        get_links = getattr(inner, "get_links", None)
        if callable(get_links):
            names = [link.get_name() for link in get_links()]
            if link_names is not None:
                missing = [n for n in link_names if n not in names]
                if missing:
                    raise RuntimeError(
                        f"{actor.get_name()} has no link named {missing!r}; links are {names}"
                    )
            return {
                int(link.entity.per_scene_id): link.get_name()
                for link in get_links()
                if link_names is None or link.get_name() in link_names
            }
        if link_names is not None:
            raise RuntimeError(
                f"{actor.get_name()} is a single rigid body (no links): drop part= / "
                "contact_point_id= for it"
            )
        return {int(inner.per_scene_id): inner.get_name()}

    def contact_point_links(self, actor: Any, contact_point_id: Any) -> list[str] | None:
        """Names of the links an ARTICULATION's annotated contact point(s)
        ride on.  ``None`` for a rigid actor (whose points ride on the single
        body) or for an actor without contact annotations.  ``contact_point_id``
        may be one id or a list of them, matching ``grasp_object``."""
        cfg = getattr(actor, "config", None)
        key = getattr(actor, "POINTS", {}).get("contact")
        if not isinstance(cfg, dict) or key not in cfg or contact_point_id is None:
            return None
        ids = (
            list(contact_point_id)
            if isinstance(contact_point_id, (list, tuple, set))
            else [contact_point_id]
        )
        points = cfg[key]
        n = len(points)
        label = getattr(actor, "get_name", lambda: "?")()
        names: list[str] = []
        for cid in ids:
            if isinstance(cid, bool) or not isinstance(cid, (int, np.integer)):
                raise RuntimeError(
                    f"{label} contact_point_id must be an int in 0..{n - 1}, got {cid!r}"
                )
            if not 0 <= int(cid) < n:  # no negative wrap-around, no silent skip
                raise RuntimeError(f"{label} has contact points 0..{n - 1}; got {cid!r}")
            entry = points[int(cid)]
            base = entry.get("base") if isinstance(entry, dict) else None
            if base is not None and base not in names:
                names.append(base)
        return names or None

    def gripper_contacts(
        self,
        target: Any,
        side: Side | None = None,
        link_name: str | list[str] | None = None,
        contact_point_id: Any = None,
    ) -> dict:
        """Ground-truth report of which gripper links touch which links of an
        object, matched by entity identity.

        ``target`` is an object name or an ``Actor`` (callers that already hold
        one pass it directly: resolving a name back from ``actor.get_name()``
        can land on a sibling instance in tasks with two of the same asset).
        ``side=None`` looks at both arms.  ``contact_point_id`` narrows the
        target to the link(s) those annotated points ride on (articulations
        only); ``link_name`` names them directly.

        Only REAL contact counts: PhysX reports a contact pair as soon as two
        shapes are within their contact offsets (0.01 m each on the Piper, so
        pairs appear up to ~2 cm apart), and a fully OPEN gripper hovering over
        a box reads 8 pair points with separation +12 mm and zero impulse.
        Each point is kept only when its impulse is non-zero or its separation
        is at most ``CONTACT_SEPARATION_M`` (a squeeze measured 2026-08-27:
        link8 separation -0.9 mm / impulse 0.14, link7 +0.02 mm / 0.12).
        ``min_separation_m`` / ``max_impulse`` expose the raw pair readings so
        a marginal grip is visible.
        """
        if side is not None and side not in ("left", "right"):
            raise ValueError(f"arm must be 'left', 'right' or None, got {side!r}")
        actor = self.get_actor(target) if isinstance(target, str) else target
        links: list[str] | None
        if link_name is not None:
            links = [link_name] if isinstance(link_name, str) else list(link_name)
        else:
            links = self.contact_point_links(actor, contact_point_id)
        targets = self._actor_link_ids(actor, links)
        sides: list[Side] = ["left", "right"] if side is None else [side]
        fingers = {sd: self._gripper_link_ids(sd) for sd in sides}
        report: dict[str, Any] = {
            "touching": False,
            "arms": [],
            "object_links": [],
            "finger_links": [],
            "positions": [],
            "required_link": links,
            "min_separation_m": None,
            "max_impulse": 0.0,
        }
        for contact in self.task.scene.get_contacts():
            ids = [int(b.entity.per_scene_id) for b in contact.bodies]
            for a, b in ((0, 1), (1, 0)):
                if ids[a] not in targets:
                    continue
                for sd in sides:
                    if ids[b] not in fingers[sd]:
                        continue
                    real = []
                    for point in contact.points:
                        sep = float(getattr(point, "separation", 0.0))
                        imp = float(
                            np.linalg.norm(np.asarray(getattr(point, "impulse", 0.0), dtype=float))
                        )
                        if report["min_separation_m"] is None or sep < report["min_separation_m"]:
                            report["min_separation_m"] = round(sep, 5)
                        report["max_impulse"] = max(report["max_impulse"], round(imp, 5))
                        if imp > 0.0 or sep <= self.CONTACT_SEPARATION_M:
                            real.append(point)
                    if not real:
                        continue  # a pair inside the contact offset, nothing touching
                    report["touching"] = True
                    if sd not in report["arms"]:
                        report["arms"].append(sd)
                    if targets[ids[a]] not in report["object_links"]:
                        report["object_links"].append(targets[ids[a]])
                    if fingers[sd][ids[b]] not in report["finger_links"]:
                        report["finger_links"].append(fingers[sd][ids[b]])
                    report["positions"].extend([float(v) for v in point.position] for point in real)
        return report

    def _verify_grasp_contact(
        self, actor: Any, side: Side, contact_point_id: int | None = None
    ) -> None:
        """A grasp that closed on air reports success but holds nothing — the
        agent then manipulates a phantom.  Verify against the physics engine
        and fail honestly: the contact must be between THIS arm's finger links
        and, when a contact point was named, the very link that point rides on
        (touching a cabinet's frame is not holding its drawer)."""
        name = actor.get_name()
        try:
            report = self.gripper_contacts(actor, side, contact_point_id=contact_point_id)
        except (AttributeError, KeyError, TypeError) as exc:
            # this scene/actor type has no usable contact query: say so ONCE rather than
            # turning every grasp into an unverified success without a trace
            if not getattr(self, "_contact_query_warned", False):
                self._contact_query_warned = True
                print(f"[robotwin] grasp verification unavailable here: {exc!r}", flush=True)
            return
        if report["touching"]:
            return
        part = (
            f" on link(s) {report['required_link']}" if report["required_link"] is not None else ""
        )
        loose = ""
        try:
            anywhere = self.gripper_contacts(actor, None)
            if anywhere["touching"]:
                loose = (
                    f" (a gripper IS touching {anywhere['object_links']} with the "
                    f"{anywhere['arms']} hand — wrong arm or wrong part)"
                )
        except Exception:
            pass
        raise RuntimeError(
            f"grasp of {name}{part} completed its motion but the {side} "
            f"gripper's fingers have NO physical contact with it{loose} — the "
            "annotated grasp pose missed (closed on air). Try a different "
            "contact_point_id, approach closer with servo_delta first, or "
            "grasp a different part."
        )

    def _synth_top_down_grasp(self, actor: Any, side: Side, pre_dis: float) -> None:
        import transforms3d as t3d

        pose = actor.get_pose()
        px, py, pz = (float(v) for v in pose.p)
        base = np.array([-0.5, 0.5, -0.5, -0.5])  # canonical piper top-down wxyz
        self.set_gripper_blocking(side, 1.0)
        last_exc: Exception | None = None
        # yaw-sweep the wrist: annotationless objects have unknown grasp
        # width orientation, and each arm reaches only part of SO(2)
        for yaw_deg in (0, 90, 45, -45, 135):
            quat = t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([0, 0, 1], np.deg2rad(yaw_deg)), base
            ).tolist()
            for hover in (0.18, 0.16):
                try:
                    self.plan_move_to_pose(side, [px, py, pz + hover], quat)
                except Exception as exc:
                    last_exc = exc
                    continue
                try:
                    # EE→TCP vertical offset is ~0.12 m: this puts the
                    # fingertips at the object's center height
                    self.plan_move_to_pose(side, [px, py, pz + 0.12], quat)
                except Exception as exc:
                    last_exc = exc  # may still be close enough — try closing
                self.set_gripper_blocking(side, 0.0)
                try:
                    # this arm's fingers, not "some gripper somewhere"
                    touching = self.gripper_contacts(actor, side)["touching"]
                except Exception:
                    touching = False
                if touching:
                    return
                self.set_gripper_blocking(side, 1.0)
                try:
                    self.plan_move_to_pose(side, [px, py, pz + hover], quat)
                except Exception:
                    pass
        raise RuntimeError(
            "grasp fallback: no annotated grasp works for this arm and the "
            "synthesized top-down grasp found no reachable approach"
        ) from last_exc

    def plan_place_object(
        self,
        object_name: str,
        side: Side,
        target_pose: Any,
        functional_point_id: int | None = None,
        pre_dis: float = 0.1,
        dis: float = 0.02,
        is_open: bool = True,
        constrain: str = "auto",
        pre_dis_axis: str = "grasp",
    ) -> None:
        actor = self.get_actor(object_name)
        target = np.asarray(target_pose, dtype=np.float64).tolist()
        kwargs: dict[str, Any] = dict(
            pre_dis=pre_dis,
            dis=dis,
            is_open=is_open,
            constrain=constrain,
            pre_dis_axis=pre_dis_axis,
        )
        if functional_point_id is not None:
            kwargs["functional_point_id"] = functional_point_id
        self._run_plan(
            f"place {object_name} with {side} arm",
            lambda: self.task.move(
                self.task.place_actor(actor, arm_tag=ArmTag(side), target_pose=target, **kwargs)
            ),
        )

    # ---------------- checkpoint / rollback (sim-state undo) ----------------

    def _physx(self):
        scene = self.task.scene
        if hasattr(scene, "get_physx_system"):
            return scene.get_physx_system()
        return scene.physx_system

    def _drive_targets(self) -> list[list[Any]]:
        """Per-articulation, per-active-joint PD (position, velocity) drive targets
        (None when the joint has no readable target)."""
        snapshot: list[list[Any]] = []
        for art in self.task.scene.get_all_articulations():
            joints: list[Any] = []
            for joint in art.get_active_joints():
                try:
                    pos = np.asarray(joint.drive_target, dtype=np.float64).reshape(-1)
                    try:
                        vel = np.asarray(joint.drive_velocity_target, dtype=np.float64)
                        vel = vel.reshape(-1)
                    except Exception:
                        vel = None
                    joints.append((pos, vel))
                except Exception:
                    joints.append(None)
            snapshot.append(joints)
        return snapshot

    def _gripper_vals(self) -> tuple[Any, Any]:
        robot = getattr(self.task, "robot", None)
        return (getattr(robot, "left_gripper_val", None), getattr(robot, "right_gripper_val", None))

    def checkpoint(self) -> str:
        """Snapshot the full physics state (all object and robot poses,
        velocities, joint states) AND every PD drive target. Returns a
        checkpoint id for restore()."""
        if not hasattr(self, "_checkpoints"):
            self._checkpoints = {}
            self._ckpt_counter = 0
        self._ckpt_counter += 1
        cid = f"ckpt{self._ckpt_counter}"
        self._checkpoints[cid] = (self._physx().pack(), self._drive_targets(), self._gripper_vals())
        return cid

    def restore(self, checkpoint_id: str) -> None:
        # NOTE: SAPIEN's contact list is refreshed by scene.step(), not by a state
        # write — read contacts only after the next step.
        """Rewind the physics state to a checkpoint. The step budget is NOT
        refunded — time spent rehearsing stays spent."""
        if not hasattr(self, "_checkpoints") or checkpoint_id not in self._checkpoints:
            raise KeyError(f"unknown checkpoint {checkpoint_id!r}; call checkpoint() first")
        entry = self._checkpoints[checkpoint_id]
        if isinstance(entry, tuple) and len(entry) == 3:
            state, targets, grip_vals = entry
        elif isinstance(entry, tuple):
            state, targets = entry
            grip_vals = (None, None)
        else:
            state, targets, grip_vals = entry, None, (None, None)
        self._physx().unpack(state)
        # SAPIEN 3.0.0b1's pack()/unpack() DOES carry the PD drive and velocity targets
        # (measured 2026-08-27: a target set before pack() came back after unpack()).
        # The old bug was a loop that re-pinned every target to the restored qpos "to
        # stop drift" — a grip is exactly a drive target commanded past the current qpos,
        # so that erased the squeeze and DROPPED whatever the gripper held.  Replaying the
        # snapshot below is belt-and-braces (position AND velocity targets, never zeroed);
        # qpos pinning only happens for joints whose target could not be read.
        arts = self.task.scene.get_all_articulations()
        if targets is not None and len(targets) != len(arts):
            raise RuntimeError(
                f"checkpoint {checkpoint_id!r} was taken with {len(targets)} articulations, "
                f"the scene now has {len(arts)} — it belongs to another episode"
            )
        for index, art in enumerate(arts):
            saved = targets[index] if targets is not None and index < len(targets) else None
            qpos = art.get_qpos()
            i = 0
            for j, joint in enumerate(art.get_active_joints()):
                dof = joint.get_dof() if hasattr(joint, "get_dof") else 1
                want = vel = None
                if saved is not None and j < len(saved) and saved[j] is not None:
                    want, vel = saved[j] if isinstance(saved[j], tuple) else (saved[j], None)
                if want is None:
                    want = qpos[i : i + dof]
                try:
                    joint.set_drive_target([float(v) for v in np.asarray(want).reshape(-1)])
                    if vel is not None:
                        joint.set_drive_velocity_target(
                            [float(v) for v in np.asarray(vel).reshape(-1)]
                        )
                except Exception:
                    pass
                i += dof
        # RoboTwin's Python-side gripper bookkeeping (what get_observation and
        # is_*_gripper_open read) is not part of the physx state: rewind it too
        robot = getattr(self.task, "robot", None)
        if robot is not None:
            for attr, val in zip(("left_gripper_val", "right_gripper_val"), grip_vals):
                if val is not None:
                    setattr(robot, attr, val)

    # ---------------- higher-level primitives (general, task-agnostic) ----------------

    @staticmethod
    def _slide_along_tool(pose7: Any, distance: float) -> list[float]:
        """RoboTwin's own pre-grasp -> grasp step: advance the pose along its
        tool axis (local +x) by ``distance`` metres, keeping the orientation."""
        p = np.array(pose7, dtype=np.float64)
        mat = t3d.quaternions.quat2mat(p[-4:])
        p[:3] += np.asarray([distance, 0.0, 0.0]) @ np.linalg.inv(mat)
        return [float(v) for v in p]

    def _plan_path_status(self, side: Side, pose7: Any, last_qpos: Any = None) -> tuple[str, Any]:
        if self._planner_tracked:
            self._sync_planner_world()
        fn = self.task.robot.left_plan_path if side == "left" else self.task.robot.right_plan_path
        try:
            res = fn(list(pose7)) if last_qpos is None else fn(list(pose7), last_qpos=last_qpos)
        except Exception as exc:  # noqa: BLE001 - the planner reports failure many ways
            self._arm_plan_flag()
            return f"{type(exc).__name__}: {exc}", None
        finally:
            self._arm_plan_flag()
        return str(res.get("status", "?")), res

    def grasp_candidates(
        self,
        object_name: str,
        side: Side,
        contact_point_ids: Any = None,
        pre_grasp_dis: Any = (0.10, 0.07, 0.05, 0.03),
        grasp_dis: Any = (0.0, 0.02, 0.04),
        max_plans: int = 140,
    ) -> dict:
        """Planning-only probe: which annotated grasp of this object can this
        arm actually REACH, and from how far back?

        RoboTwin chooses the contact point by ORIENTATION SCORE alone and never
        checks that the descend onto it has an IK solution, so ``grasp_object``
        either raises or (worse) leaves the arm where it stood.  This asks the
        arm's own planner about every (contact point, pre-grasp standoff, final
        standoff) combination and reports what it found — no motion, no
        simulation state is changed.

        Args:
            object_name: see list_objects().
            side: 'left' or 'right'.
            contact_point_ids: ids to consider (int or list); None = all the
                annotated points of the asset.
            pre_grasp_dis: standoff(s) to try for the APPROACH pose, in metres,
                largest first.  Same meaning as grasp_object's argument.
            grasp_dis: how far SHORT of the annotated contact point to stop, in
                metres.  0.0 closes on the point itself; a non-zero value keeps
                the fingertips that far back — which is what makes an object
                lying on the table graspable at all when the point itself sits
                at table height and the descend is refused.
            max_plans: hard bound on the CHARGED planner calls (call COUNT, not
                wall clock, so a replay costs the same on a busy host):
                ``plans_used`` never exceeds it and ``budget_exhausted`` is True
                only when a call was actually skipped for lack of budget.  One
                charged call is one ``get_grasp_pose`` (itself a sweep over
                RoboTwin's ROTATE_NUM yaw variants), one approach plan or one
                descend plan.

        Returns:
            {'candidates': [...], 'best': {...}|None, 'plans_used': int,
             'budget_exhausted': bool, 'reasons': {...}} where every candidate
            is {'contact_point_id', 'pre_grasp_dis', 'grasp_dis', 'pre_pose',
            'grasp_pose'} and both poses planned.  'best' is the first
            candidate in preference order (smallest grasp_dis — the firmest
            grip — then the largest approach standoff).  Feed it straight to
            grasp_object(contact_point_ids=..., pre_grasp_dis=...,
            grasp_dis=...).
        """
        actor = self.get_actor(object_name)
        n_points = self._n_contact_points(actor)
        if n_points == 0:
            raise RuntimeError(f"{object_name} has no annotated contact points to probe")
        ids = _as_id_list(contact_point_ids)
        if ids is None:
            ids = list(range(n_points))
        if not ids:
            raise ValueError("contact_point_ids is empty: nothing to probe")
        for cid in ids:
            if not 0 <= cid < n_points:
                raise RuntimeError(f"{object_name} has contact points 0..{n_points - 1}; got {cid}")
        pres = [float(v) for v in np.atleast_1d(pre_grasp_dis)]
        gaps = [float(v) for v in np.atleast_1d(grasp_dis)]
        if side not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {side!r}")
        if not pres or not gaps:
            raise ValueError("pre_grasp_dis and grasp_dis must each name at least one value")
        if all(g >= min(pres) for g in gaps):
            raise ValueError(
                "every grasp_dis >= the smallest pre_grasp_dis: the descend would go backwards, "
                "nothing to probe"
            )
        if int(max_plans) < 1:
            raise ValueError("max_plans must be >= 1")

        entity = self.task.robot.left_entity if side == "left" else self.task.robot.right_entity
        used = 0
        skipped = False  # a planner call was NOT made because the budget was gone
        candidates: list[dict] = []
        reasons: dict[str, str] = {}

        def spend() -> bool:
            """Charge one planner call; False (and remember it) when none is left."""
            nonlocal used, skipped
            if used >= int(max_plans):
                skipped = True
                return False
            used += 1
            return True

        # breadth-first over standoffs: a truncated budget still sees every point at the
        # first standoff (depth-first used to spend it all on the first few ids)
        for pre in pres:
            for cid in ids:
                key = f"cid{cid}@pre{pre:g}"
                if not spend():
                    reasons[key] = "skipped: budget"
                    continue
                self._arm_plan_flag()
                pre_pose = self.task.get_grasp_pose(
                    actor, ArmTag(side), contact_point_id=cid, pre_dis=pre
                )
                self._arm_plan_flag()
                if pre_pose is None:
                    reasons[key] = "no approach pose of this point is reachable"
                    continue
                if not spend():
                    reasons[key] = "skipped: budget (before the approach plan)"
                    continue
                status, res = self._plan_path_status(side, pre_pose)
                if status != "Success":
                    reasons[key] = f"approach pose unplannable ({status})"
                    continue
                seed = np.asarray(res["position"][-1], dtype=np.float32)
                full = np.asarray(entity.get_qpos(), dtype=np.float32).copy()
                full[: seed.shape[0]] = seed
                for gap in gaps:
                    gkey = f"{key}+gap{gap:g}"
                    if gap >= pre:
                        reasons[gkey] = "skipped: grasp_dis >= pre_grasp_dis"
                        continue
                    if not spend():
                        reasons[gkey] = "skipped: budget"
                        continue
                    grasp_pose = self._slide_along_tool(pre_pose, pre - gap)
                    status2, _ = self._plan_path_status(side, grasp_pose, last_qpos=full)
                    if status2 != "Success":
                        reasons[gkey] = f"descend unplannable ({status2})"
                        continue
                    candidates.append(
                        {
                            "contact_point_id": cid,
                            "pre_grasp_dis": pre,
                            "grasp_dis": gap,
                            "pre_pose": [float(v) for v in pre_pose],
                            "grasp_pose": grasp_pose,
                        }
                    )
        candidates.sort(key=lambda c: (c["grasp_dis"], -c["pre_grasp_dis"]))
        return {
            "candidates": candidates,
            "best": candidates[0] if candidates else None,
            "plans_used": used,
            "budget_exhausted": skipped,
            "reasons": reasons,
        }

    def plan_servo_delta(
        self,
        side: Side,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        max_norm: float = 0.06,
    ) -> list:
        """Fast small Cartesian step via screw interpolation — no RRT, no
        collision avoidance, deterministic. The closed-loop building block:
        read feedback, step a little, read again."""
        delta = np.array([x, y, z], dtype=np.float64)
        n = float(np.linalg.norm(delta))
        if n > max_norm:
            raise ValueError(
                f"servo step too large ({n:.3f} m > {max_norm} m); use "
                "move_by_displacement for large motions"
            )
        ee = self.get_arm_pose(side)
        return self.plan_glide_to_pose(
            side, [float(ee[0] + x), float(ee[1] + y), float(ee[2] + z)], ee[3:7]
        )

    def plan_glide_to_pose(self, side: Side, position: Any, quat_wxyz: Any) -> list:
        """Screw (constant-twist) motion of the hand to a full pose in ONE monotone
        motion (internal building block; ``servo_delta_*`` is its translation-only
        face).  Measured 2026-08-27: a pure translation is straight to < 1 mm; a
        translate-and-rotate bows the tool and anything it carries off the chord by
        up to ~1 cm per 30-60 deg over 8 cm (the end link is 12 cm off the screw
        axis) — split a large reorientation from the translation when clearance is
        tight.  Deterministic and quick (~0.2 s), and unlike the RRT path behind
        goto_pose it never dips or swings, which is what keeps a CARRIED object out
        of the table or a shelf (the planner does not model the held object).

        It does NOT see obstacles at all: mplib 0.2's plan_screw ignores its own
        collision check and the planning world holds only table/wall/ground.  A target
        through solid geometry therefore ends one of two ways and NEITHER is arrival:
        it returns Success and the physics stops the hand short (a glide 10 cm below
        the tabletop returned Success 0.24 m short), or the screw is rejected for the
        joint limits ALONG the way — an error that reads like a collision but is not
        one (``tests/sim/test_robotwin_glide.py`` covers both).  It raises only on a
        joint limit, a singular posture (the reset/home pose is one — goto_pose first),
        or a 180-degree reorientation.  Arrival is not checked here: callers close the
        loop on what they carry."""
        robot = self.task.robot
        qn = float(np.linalg.norm(np.asarray(quat_wxyz, dtype=float).reshape(-1)[:4]))
        if not 0.95 <= qn <= 1.05:
            raise ValueError(f"quaternion_wxyz must be a unit wxyz quaternion (norm {qn:.3f})")
        target = self._to_pose7(position, quat_wxyz)
        # a zero-length screw is not plannable (mplib stops on |twist| < 1e-4 BEFORE it
        # checks arrival): a request within 0.2 mm / 0.2 mrad of the current pose is
        # already satisfied, so return the pose instead of raising
        here = np.asarray(self.get_arm_pose(side), dtype=float)
        dp = float(np.linalg.norm(np.asarray(target[:3], dtype=float) - here[:3]))
        qa, qb = np.asarray(target[3:7], dtype=float), here[3:7]
        dang = 2.0 * float(np.arccos(np.clip(abs(float(qa @ qb)) / max(qn, 1e-9), -1.0, 1.0)))
        if dp < 2e-4 and dang < 2e-4:
            return [float(v) for v in here]
        trans = robot._trans_from_gripper_to_endlink(target, arm_tag=side)
        planner = robot.left_mplib_planner if side == "left" else robot.right_mplib_planner
        entity = robot.left_entity if side == "left" else robot.right_entity
        result = planner.plan_screw(entity.get_qpos(), trans, arms_tag=side)
        status = str((result or {}).get("status"))
        if status == "screw plan failed":
            # mplib's default 0.1 rad qpos_step leaves a last remainder under its 1e-4
            # twist floor one step short of arrival (measured: a 3 cm +x glide 'failed'
            # with no joint near a limit and re-planned fine at a finer step)
            try:
                fine = planner.planner.plan_screw(
                    goal_pose=trans,
                    current_qpos=entity.get_qpos(),
                    time_step=1 / 250,
                    qpos_step=0.02,
                )
                if fine.get("status") == "Success" and len(fine.get("position", [])) <= 2500:
                    result, status = fine, "Success"
            except Exception:
                pass
        if not result or status != "Success":
            cause = {
                "screw plan failed.": "a ~180-degree reorientation (the screw axis is undefined)",
                "screw plan failed": "a joint limit along the way",
                "Fail": "a path longer than 2500 steps",
            }.get(status, "the planner")
            raise RuntimeError(
                f"screw motion not plannable ({status}): {cause}; a singular posture (the "
                "reset pose: goto_pose first) reads the same way — split the rotation, use "
                "a smaller delta, or a planned move"
            )
        seq: dict[str, Any] = {
            "left_arm": None,
            "left_gripper": None,
            "right_arm": None,
            "right_gripper": None,
        }
        seq["left_arm" if side == "left" else "right_arm"] = {
            "position": result["position"],
            "velocity": result["velocity"],
        }
        self.task.take_dense_action(seq)
        return [float(v) for v in self.get_arm_pose(side)]

    def plan_drag_articulated(
        self,
        object_name: str,
        side: Side,
        contact_point_id: int = 1,
        max_iters: int = 15,
    ) -> int:
        """Drag an articulated part (lid, door, handle) along its arc by
        repeatedly pulling the gripper onto the part's MOVING annotated
        contact point — the same pattern RoboTwin's own experts use. The
        gripper should already hold / touch the part. Returns the number of
        pull iterations that planned successfully (0 = no progress)."""
        actor = self.get_actor(object_name)
        done = 0
        for _ in range(max_iters):
            try:
                self.task.move(
                    self.task.grasp_actor(
                        actor,
                        arm_tag=ArmTag(side),
                        pre_grasp_dis=0.0,
                        grasp_dis=0.0,
                        contact_point_id=contact_point_id,
                    )
                )
            except Exception:
                break
            if not bool(self.task.plan_success):
                self.task.plan_success = True
                break
            done += 1
        self.task.plan_success = True
        return done

    def plan_move_to_pose_flex(
        self, side: Side, position: Any, quat_wxyz: Any | None = None
    ) -> list:
        """Like plan_move_to_pose, but sweeps wrist yaw until IK succeeds.
        Returns the quaternion actually used."""
        import transforms3d as t3d

        base = np.asarray(
            quat_wxyz if quat_wxyz is not None else [-0.5, 0.5, -0.5, -0.5], dtype=np.float64
        )
        last: Exception | None = None
        for yaw in (0, 30, -30, 60, -60, 90, -90, 135, 180):
            quat = t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([0, 0, 1], np.deg2rad(yaw)), base
            ).tolist()
            try:
                self.plan_move_to_pose(side, position, quat)
                return quat
            except Exception as exc:
                last = exc
        raise RuntimeError(
            f"no wrist orientation reaches {list(np.asarray(position).reshape(3))} "
            f"for the {side} arm — the position itself is likely outside the "
            "reachable envelope; move closer or use the other arm"
        ) from last

    def _articulated_entity(self, name: str):
        """Find the articulated scene entity (drawer/door/lid owner) the task
        stores under this attribute or actor name."""
        for attr, value in self.task.__dict__.items():
            if attr == "robot":
                continue
            if callable(getattr(value, "get_qpos", None)) and callable(
                getattr(value, "get_qlimits", None)
            ):
                actor_name = None
                try:
                    actor_name = value.get_name()
                except Exception:
                    pass
                if attr == name or actor_name == name:
                    return value
        raise KeyError(f"no articulated object named {name!r} in this scene")

    @staticmethod
    def _collision_bodies(ent: Any) -> list:
        """Every physx body of an entity whose collision shapes can be read:
        each link of an articulation, or the single rigid component of a plain
        actor (which has no ``get_links`` at all — the reason an earlier
        version of this helper silently returned ``None`` for tables and
        boxes)."""
        links = getattr(ent, "get_links", None)
        if not callable(links):
            inner = getattr(ent, "actor", None)
            links = getattr(inner, "get_links", None)
        if callable(links):
            return list(links())
        target = ent if hasattr(ent, "get_components") else getattr(ent, "actor", None)
        if target is None or not hasattr(target, "get_components"):
            return []
        return [c for c in target.get_components() if hasattr(c, "get_collision_shapes")]

    @staticmethod
    def _shape_local_bounds(shape: Any) -> np.ndarray | None:
        """Corner points bounding one SAPIEN collision shape in its own frame: mesh
        vertices, a box's corners, or the bounding box of a sphere / capsule /
        cylinder (radius on y/z, half length + radius along x — SAPIEN's capsule and
        cylinder axes).  ``None`` for a plane or an unreadable shape; before this the
        round shapes were skipped silently and a sphere under a query point 'had no
        surface'."""
        corners = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
        )
        try:
            return np.asarray(shape.get_vertices()) * np.asarray(shape.get_scale())
        except Exception:
            pass
        half = getattr(shape, "half_size", None)
        if half is not None:
            return corners * np.asarray(half, dtype=np.float64)
        radius = getattr(shape, "radius", None)
        if radius is None:
            return None
        r = float(radius)
        half_len = getattr(shape, "half_length", None)
        kind = type(shape).__name__.lower()
        if half_len is None:
            return corners * r
        along = float(half_len) + (r if "capsule" in kind else 0.0)
        return corners * np.array([along, r, r], dtype=np.float64)

    def _entity_world_aabb(self, ent: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """World-frame axis-aligned bounds of an entity's collision geometry
        (articulation or rigid actor), or ``None`` when no shape can be read."""
        lo: list[np.ndarray] = []
        hi: list[np.ndarray] = []
        for link in self._collision_bodies(ent):
            try:
                pose = link.get_entity_pose().to_transformation_matrix()
            except Exception:
                try:
                    pose = link.entity.get_pose().to_transformation_matrix()
                except Exception:
                    continue
            for shape in link.get_collision_shapes():
                local = shape.get_local_pose().to_transformation_matrix()
                pts = self._shape_local_bounds(shape)
                if pts is None:
                    continue  # a plane (infinite) or an unknown shape class
                pts = (local[:3, :3] @ pts.T).T + local[:3, 3]
                pts = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
                lo.append(pts.min(axis=0))
                hi.append(pts.max(axis=0))
        if not lo:
            return None
        return np.min(np.asarray(lo), axis=0), np.max(np.asarray(hi), axis=0)

    def _articulation_qpos(self) -> dict[str, np.ndarray]:
        """Joint positions of every articulated scene object (robot excluded)."""
        out: dict[str, np.ndarray] = {}
        for attr, value in self.task.__dict__.items():
            if attr == "robot":
                continue
            if callable(getattr(value, "get_qpos", None)) and callable(
                getattr(value, "get_qlimits", None)
            ):
                try:
                    out[attr] = np.asarray(value.get_qpos(), dtype=np.float64).reshape(-1)
                except Exception:
                    continue
        return out

    def plan_open_articulated(
        self,
        object_name: str,
        side: Side | None = None,
        target_fraction: float = 0.6,
        joint_index: int | None = None,
        max_strokes: int = 20,
        stroke: float = 0.045,
        max_planner_calls: int = 300,
        time_budget_s: float | None = None,
    ) -> dict:
        """Open an articulated part (drawer / door / lid) to ``target_fraction``
        of its joint limit, as a closed progress loop:

        hook the handle on its annotated contact points -> DISCOVER the pull
        direction (one real short stroke along the common front-pull, else a
        rehearse-and-rewind sweep) -> pull stroke by stroke, re-reading qpos
        every time, in decreasing step sizes, RE-DISCOVERING the direction on
        each stall -> when strokes stop paying, SLIDE THE GRIP ALONG THE
        HANDLE and pull again, keeping whichever grip point got furthest.

        Two things decide how far a part opens:

        *Where on the handle you hold.*  The stall is kinematic, not
        mechanical: the arm folds up long before the joint runs out, and the
        grip point moves the whole arm.  Measured on the RoboTwin cabinet with
        the right arm: gripping in front of its own shoulder stalls at 34 % of
        the drawer's travel, a cross-body grip one hand-width across reaches
        69 %.

        *Whether you follow the part.*  A drawer slides along one fixed world
        direction, but a HINGED part's tangent rotates as it swings, so a pull
        that was on-axis at the start fights the hinge a few centimetres later
        and reads as a stall.  Every stall therefore re-sweeps the direction
        around the current one (rehearse-and-rewind, +-25/50/75 deg) before
        giving up, which turns one fixed pull into a piecewise arc.

        Strokes are screw-interpolated (~0.2 s, deterministic); an RRT
        displacement is only the fallback, since an RRT *failure* costs the
        whole planning budget.

        The call gives up cleanly on a DETERMINISTIC bound —
        ``max_planner_calls`` motion/rehearsal calls — rather than on the wall
        clock, so a replay on a loaded machine takes the same path as the
        original run.  ``time_budget_s`` is an optional extra safety valve
        (``None`` = off); it makes the call non-deterministic, so leave it off
        for anything you intend to certify.

        Returns ``{opened, qpos, target, fraction, strokes, arm, holding,
        grip_offset_m, planner_calls, reason}``; ``reason`` is one of
        ``already open``, ``opened``, ``arm reach limit``, ``planner budget``,
        ``time budget``.
        """
        import math
        import time as _time

        spent = {"n": 0}
        t_deadline = None if time_budget_s is None else _time.monotonic() + float(time_budget_s)

        def spend(n: int = 1) -> None:
            spent["n"] += n

        def exhausted() -> str | None:
            """Deterministic bound first; the wall clock only if asked for."""
            if spent["n"] >= max_planner_calls:
                return "planner budget"
            if t_deadline is not None and _time.monotonic() >= t_deadline:
                return "time budget"
            return None

        ent = self._articulated_entity(object_name)
        # bound BEFORE result(), which measures the grip against it
        actor = self.get_actor(object_name)
        limits = [[float(l), float(h)] for l, h in np.asarray(ent.get_qlimits()).reshape(-1, 2)]
        active_joints = list(ent.get_active_joints()) if hasattr(ent, "get_active_joints") else []

        def _child_link_name(j) -> str | None:
            link = getattr(j, "child_link", None)
            if link is None and callable(getattr(j, "get_child_link", None)):
                link = j.get_child_link()
            return link.get_name() if link is not None else None

        if joint_index is None:
            finite = [
                (i, float(hi - lo))
                for i, (lo, hi) in enumerate(limits)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-6
            ]
            if not finite:
                raise RuntimeError(f"{object_name} has no finite-range joint to open")
            # the joint the HANDLE moves: the one whose child link carries the annotated
            # contact points (a cabinet has three identical drawer joints; `max` over
            # equal ranges only picked joint 0 by luck), else the largest range
            try:
                n_ann = sum(1 for _ in actor.iter_contact_points())
                on_links = self.contact_point_links(actor, list(range(n_ann))) or []
            except Exception:
                on_links = []
            handled = [
                (i, r)
                for i, r in finite
                if i < len(active_joints) and _child_link_name(active_joints[i]) in on_links
            ]
            joint_index = max(handled or finite, key=lambda t: t[1])[0]
        lo, hi = (float(x) for x in limits[joint_index])
        target = lo + target_fraction * (hi - lo)
        part_link = (
            _child_link_name(active_joints[joint_index])
            if joint_index < len(active_joints)
            else None
        )

        def _q_guarded(v: float) -> float:
            if v > hi + 0.5 * (hi - lo):
                raise RuntimeError(
                    f"open_articulated: joint reading {v:.3f} exceeds limit "
                    f"{hi:.3f} — entity/joint mismatch on repeat call; read "
                    "get_joint_states() yourself instead of re-calling"
                )
            return v

        def q() -> float:
            return _q_guarded(float(np.asarray(ent.get_qpos()).reshape(-1)[joint_index]))

        def result(sd, strokes, offset, reason):
            final = q()
            # Whether the jaws are still ON the part is a physics question, and
            # the answer is often no: a pull that ends at the arm's reach limit
            # frequently ends with the handle slipped out of the fingers.
            # Measured 2026-08-27: the microwave door stops at 31 % with the
            # left gripper touching NO link of the microwave at all.  Reporting
            # "still holding" there would be a lie the caller plans around.
            # SAPIEN's contact list is only refreshed by a scene step: after a
            # restore() the previous state's contacts are still what get_contacts()
            # returns, so the world is stepped once before measuring (a checkpoint
            # restored to a held drawer read "touching False" until then, measured).
            touching = False
            links: list = []
            if sd is not None:
                try:
                    self.task.scene.step()
                    on_part = self.gripper_contacts(actor, sd, link_name=part_link)
                    anywhere = self.gripper_contacts(actor, sd)
                    touching = bool(on_part["touching"] if part_link else anywhere["touching"])
                    links = anywhere["object_links"]
                except Exception:
                    touching = False
            if touching:
                note = (
                    f"the {sd} gripper is still ON the part ({', '.join(links)}) — a part "
                    "held open against gravity snaps shut if released (a drawer on "
                    "horizontal rails stays), so do the task with the OTHER arm and open "
                    "this gripper last"
                )
            else:
                note = (
                    f"the {sd} gripper is NO LONGER touching the part (the handle slipped "
                    "out at the reach limit): nothing is holding it open. A drawer on "
                    "horizontal rails will stay where it is, but a door or lid loaded by "
                    "gravity may already be closing — re-read the joint before you rely "
                    "on this opening, and re-hook if it moved."
                )
            return {
                "opened": final >= target,
                "qpos": final,
                "target": target,
                "fraction": (final - lo) / (hi - lo) if hi > lo else 0.0,
                "strokes": strokes,
                "arm": sd,
                "holding": sd if touching else None,
                "gripper_on_part": touching,
                "gripper_on_links": links,
                "grip_offset_m": round(float(offset), 4),
                "planner_calls": spent["n"],
                "reason": reason,
                "note": note,
            }

        if q() >= target:
            rep = result(side, 0, 0.0, "already open")
            rep["arm"] = rep["holding"] = None
            rep["gripper_on_part"] = False
            rep["note"] = "already at or past the target: no arm engaged, nothing is held"
            return rep

        try:
            n_points = max(1, sum(1 for _ in actor.iter_contact_points()))
        except Exception:
            n_points = 4

        def engage_annotated(sd) -> bool:
            """Hook the handle on its annotated contact points WITHOUT pinch
            verification — for articulated parts, joint progress (not
            fingertip contact) is the honest engagement test.  Contact ids are
            enumerated explicitly so the call can never fall through to the
            synthesised top-down grasp: that one approaches from above and
            shoves the part shut."""
            for cid_try in range(n_points):
                if exhausted():
                    return False
                try:
                    spend(3)  # approach + close + settle
                    self.plan_grasp_object(
                        object_name,
                        sd,
                        pre_grasp_dis=0.05,
                        grasp_dis=0.0,
                        verify=False,
                        contact_point_id=cid_try,
                    )
                    return True
                except Exception:
                    continue
            return False

        def park(sd):
            spend(2)
            self.set_gripper(sd, 1.0)
            try:
                self.plan_move_by_displacement(sd, z=0.05)
            except Exception:
                pass
            try:
                self.plan_back_to_origin(sd)
            except Exception:
                pass

        DIRS = (
            (0, -1),
            (-1, 0),
            (1, 0),
            (0, 1),
            (-0.7, -0.7),
            (0.7, -0.7),
            (-0.7, 0.7),
            (0.7, 0.7),
        )

        def pull(sd, dx, dy, dist, allow_plan: bool = False) -> float:
            """One stroke along (dx, dy).  Screw interpolation only, unless
            ``allow_plan``: a *failed* RRT costs the whole planning budget
            (20-30 s here), and every stall inside the stroke loop is a
            failure by construction — falling back to RRT there is what made
            this call take ten minutes and still stop short."""
            q0 = q()
            spend()
            try:
                self.plan_servo_delta(sd, x=dist * dx, y=dist * dy)
            except Exception:
                if not allow_plan:
                    return 0.0
                try:
                    spend(4)  # an RRT attempt, successful or not, is the expensive one
                    self.plan_move_by_displacement(sd, x=dist * dx, y=dist * dy)
                except Exception:
                    return 0.0
            return q() - q0

        def discover(sd, around: tuple | None = None):
            """Rehearse a short pull in each candidate direction and rewind,
            keeping the first that moves the joint.  ``around`` narrows the
            sweep to headings either side of the direction that just stalled —
            that is the hinge tangent chasing its own arc."""
            if around is None:
                cands = DIRS
            else:
                base = math.atan2(around[1], around[0])
                cands = tuple(
                    (math.cos(base + math.radians(a)), math.sin(base + math.radians(a)))
                    for a in (25, -25, 50, -50, 75, -75)
                )
            for dx, dy in cands:
                if exhausted():
                    return None
                cid = self.checkpoint()
                gain = pull(sd, dx, dy, 0.03)
                self.restore(cid)
                self._checkpoints.pop(cid, None)  # a rehearsal snapshot is not kept
                if gain > 0.006:
                    return (dx, dy)
            return None

        def pull_to_stall(sd, direction, budget, redirects: int | None = None):
            """Long strokes until they stop paying, then finer ones — the last
            centimetre is always won with small steps.  A stall at any step
            size first asks whether the part has simply turned under the pull
            (hinges) before the step size is dropped — and it pays on prismatic
            parts as well (a jammed drawer frees under a slightly turned pull).

            Returns ``(strokes, direction)``: the heading it ended up pulling
            along, which the grip-slide phase then reuses."""
            strokes = 0
            d = (float(direction[0]), float(direction[1]))
            if redirects is None:
                # the sweep pays on prismatic parts too (measured: the cabinet drawer went
                # 0.67 -> 0.72/0.77 with it; skipping it for slides was tried 2026-08-27
                # and cost the place_held_object test 2 cm of placement accuracy)
                redirects = 3
            redirects_left = redirects
            for size in (stroke, 0.02, 0.01, 0.006):
                while strokes < budget and q() < target and not exhausted():
                    if pull(sd, d[0], d[1], size) >= 0.002:
                        strokes += 1
                        continue
                    if redirects_left <= 0:
                        break
                    redirects_left -= 1
                    turned = discover(sd, around=d)
                    if turned is None:
                        break
                    d = turned
            return strokes, d

        sides = [side] if side is not None else ["right", "left"]
        sd = direction = None
        hooked_ckpt: str | None = None
        hooked_pose: np.ndarray | None = None
        for cand in sides:
            if exhausted():
                break
            if engage_annotated(cand):
                hooked_ckpt = self.checkpoint()
                hooked_pose = self.get_arm_pose(cand)
                # the front-pull (-y) is the overwhelmingly common case: test
                # it FOR REAL first — a real stroke that pays is both
                # discovery and progress in one
                if pull(cand, 0, -1, 0.04, allow_plan=True) > 0.003:
                    sd, direction = cand, (0, -1)
                    break
                # the real stroke moved the hand 4 cm along the bar without moving the
                # joint: rewind to the clean engagement before rehearsing other headings
                self.restore(hooked_ckpt)
                direction = discover(cand)
                if direction is not None:
                    sd = cand
                    break
            park(cand)
        if sd is None or hooked_ckpt is None or hooked_pose is None:
            why = exhausted()
            raise RuntimeError(
                f"open_articulated: could not hook a handle on {object_name} "
                f"that moves its joint (tried arms {sides}, all contact "
                f"points, all pull directions"
                + (f"; stopped early on {why} after {spent['n']} calls)" if why else ")")
            )

        # NOTE: the heading pull_to_stall ends on is deliberately DISCARDED.
        # It is where the part's tangent had rotated to (hinges), but the grip
        # slide below is laid out around the pose the hand was in at
        # ENGAGEMENT, and cross(pull, up) has to be the handle's own axis at
        # that moment.  Feeding the rotated heading back in tilts the slide
        # off the bar and every offset then misses: measured on the cabinet
        # drawer, that cost 5 points of travel (0.67 -> 0.62) and turned a
        # certified put_object_cabinet into a 0.0.  Each re-grip re-discovers
        # the turn for itself anyway.
        pulled, _turned = pull_to_stall(sd, direction, max_strokes)
        strokes = 1 + pulled
        if q() >= target or exhausted():
            return result(
                sd, strokes, 0.0, "opened" if q() >= target else (exhausted() or "opened")
            )

        # ---- slide the grip along the handle and pull again ----
        pull3 = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        pull3 /= max(float(np.linalg.norm(pull3)), 1e-9)
        lat = np.cross(pull3, [0.0, 0.0, 1.0])
        lat /= max(float(np.linalg.norm(lat)), 1e-9)
        span = 0.08
        aabb = self._entity_world_aabb(ent)
        if aabb is not None:
            span = max(0.02, float(abs(np.dot(aabb[1] - aabb[0], lat))) / 2.0)
        engaged_pos = np.asarray(hooked_pose[:3], dtype=np.float64)
        engaged_quat = [float(v) for v in hooked_pose[3:7]]
        # a cross-body grip — the side of the handle AWAY from this arm's own
        # base — is the one that usually wins, so try that sign first
        toward_base = np.asarray(self.get_arm_base_pose(sd)[:3], dtype=np.float64) - engaged_pos
        sign = -1.0 if float(np.dot(toward_base * [1.0, 1.0, 0.0], lat)) > 0 else 1.0
        offsets = [sign * f * span for f in (0.9, 0.7, 0.45)]
        offsets += [-o for o in offsets]

        best_q, best_ckpt, best_offset, best_strokes = q(), self.checkpoint(), 0.0, strokes
        for offset in offsets:
            if exhausted():
                break
            self.restore(hooked_ckpt)
            try:
                spend(3)
                self.set_gripper_blocking(sd, 1.0)
                self.plan_move_to_pose(sd, engaged_pos + offset * lat - 0.05 * pull3, engaged_quat)
                self.plan_move_to_pose(sd, engaged_pos + offset * lat, engaged_quat)
                self.set_gripper_blocking(sd, 0.0)
            except Exception:
                continue
            got, _ = pull_to_stall(sd, direction, max_strokes)
            if q() > best_q + 1e-4:
                best_q, best_ckpt, best_offset, best_strokes = q(), self.checkpoint(), offset, got
            if q() >= target:
                break
        if abs(q() - best_q) > 1e-4:
            self.restore(best_ckpt)
        for cid in list(self._checkpoints):  # only the kept best state survives the call
            if cid not in (best_ckpt, hooked_ckpt):
                self._checkpoints.pop(cid, None)
        if q() >= target:
            reason = "opened"
        elif exhausted():
            reason = exhausted()
        elif best_strokes >= max_strokes:
            reason = "stroke cap"
        else:
            reason = "arm reach limit"
        # deliberately NO park: a part held open against gravity self-closes
        # when released.  The caller acts with the OTHER arm and releases this
        # one afterwards.
        return result(sd, best_strokes, best_offset, reason)

    def plan_check_reachable(
        self, side: Side, position: Any, quat_wxyz: Any | None = None, flex: bool = True
    ) -> dict:
        """Pure planning probe: is this gripper pose plannable for this arm?
        Runs the motion planner WITHOUT executing anything — the robot does
        not move and the world does not change. With flex=True it sweeps the
        same wrist yaws as goto_pose_flex and reports the first plannable
        orientation, so programs can PICK a working approach instead of
        burning a failed motion to discover it."""
        import transforms3d as t3d

        base = np.asarray(
            quat_wxyz if quat_wxyz is not None else [-0.5, 0.5, -0.5, -0.5], dtype=np.float64
        )
        yaws = (0, 30, -30, 60, -60, 90, -90, 135, 180) if flex else (0,)
        # rehearse-and-rewind: attempt the REAL motion (exact conventions by
        # construction), then restore the pre-probe snapshot either way —
        # bit-exact, so the caller observes zero world change
        cid = self.checkpoint()
        try:
            for yaw in yaws:
                quat = t3d.quaternions.qmult(
                    t3d.quaternions.axangle2quat([0, 0, 1], np.deg2rad(yaw)), base
                )
                try:
                    self.plan_move_to_pose(side, position, [float(q) for q in quat])
                    return {
                        "reachable": True,
                        "yaw": int(yaw),
                        "quat_wxyz": [float(q) for q in quat],
                    }
                except Exception:
                    continue
            return {"reachable": False}
        finally:
            self.restore(cid)

    # --------------------- route A: the mplib planning world ---------------------
    #
    # RoboTwin builds each arm's mplib planning world from the SAPIEN scene inside
    # ``robot.set_planner()``, which runs BEFORE the task spawns its props, and it
    # never calls ``update_from_simulation()`` afterwards.  Measured 2026-08-27 on
    # every RoboTwin task tried (handover_mic, put_object_cabinet): the world holds
    # exactly ``table``, ``wall``, ``ground`` and the two arm articulations at their
    # RESET poses — no bottle, no cabinet, no bowl, and nothing in the jaws.  Every
    # planned transit is therefore computed as if the hand were empty in an empty
    # room, which is why a carried object gets dragged along the table edge or
    # stripped out of the fingers, and why the planner never routes around a prop.
    #
    # The two calls below let a program repair that world for the motions where it
    # matters.  They are OPT-IN: with nothing registered, ``_sync_planner_world`` is
    # never called and the planner behaves exactly as RoboTwin ships it, so existing
    # programs replay unchanged.

    def _ik_route_available(self, side: Side) -> bool:
        """Whether :meth:`_ik_reachable` can use mplib's millisecond IK route for ``side``.

        False means every probe costs a FULL path plan (seconds, and up to 20-30 s on an
        RRT failure), which is what ``plan_reach_probe`` reports as ``method: "plan"`` and
        charges to ``plans_used``.
        """
        planner = self._mplib_planner(side)
        return planner is not None and hasattr(planner, "_transform_goal_to_wrt_base")

    def _mplib_planner(self, side: Side):
        """The per-arm ``SapienPlanner``, or None for a curobo-only embodiment."""
        planner = getattr(self.task.robot, f"{side}_mplib_planner", None)
        inner = getattr(planner, "planner", None)
        return inner if getattr(inner, "planning_world", None) is not None else None

    @staticmethod
    def _as_mplib_pose(pose: Any):
        """A SAPIEN pose as an mplib one (mplib's Pose takes p/q, not sapien.Pose)."""
        from mplib.pymp import Pose as MplibPose

        return MplibPose(p=[float(v) for v in pose.p], q=[float(v) for v in pose.q])

    @staticmethod
    def _mplib_object_name(ent: Any) -> str:
        """mplib's unique name for a SAPIEN entity: ``<name>_<per_scene_id>``."""
        from mplib.sapien_utils.conversion import convert_object_name

        return convert_object_name(ent)

    def _scene_entity(self, object_name: str) -> Any:
        """The raw SAPIEN entity behind a task object name (RoboTwin wraps most
        props in an ``Actor``/``ArticulationActor`` that carries it in ``.actor``)."""
        actor = self.get_actor(object_name)
        return getattr(actor, "actor", actor)

    def _tool_link_pose(self, side: Side, link_name: str):
        """LIVE pose of the arm link the planner plans for (its move group).

        Read from SAPIEN, never from the planning world: the world's own forward
        kinematics is only as fresh as the last plan that ran in it, and an
        attachment computed against a stale link pose puts the payload metres away
        (measured 2026-08-27 — the box landed 1.06 m from the hand and collided
        with the other arm).
        """
        entity = self.task.robot.left_entity if side == "left" else self.task.robot.right_entity
        for link in entity.get_links():
            if link.get_name() == link_name:
                pose = getattr(link, "entity_pose", None)
                return pose if pose is not None else link.get_pose()
        raise RuntimeError(f"the {side} arm has no link named {link_name!r}")

    def _sync_planner_world(self, side: Side | None = None) -> None:
        """Refresh the registered planning worlds from the live scene.

        Does what ``SapienPlanningWorld.update_from_simulation`` does — arm and
        articulation base poses / qpos, object poses, attached-body relative
        poses — except that it SKIPS entities the world does not know instead of
        raising, because the world knows only the handful of things RoboTwin put
        in it at reset plus whatever ``carry_in_planner`` / ``avoid_in_planner``
        added.
        """
        scene = self.task.scene
        sides = (side,) if side is not None else tuple(self._planner_tracked)
        for sd in sides:
            planner = self._mplib_planner(sd)
            if planner is None:
                continue
            world = planner.planning_world
            for art in scene.get_all_articulations():
                model = world.get_articulation(self._mplib_object_name(art))
                if model is not None:
                    model.set_base_pose(art.root_pose)
                    model.set_qpos(art.qpos, full=True)
            for name, ent in list(self._planner_tracked.get(sd, {}).items()):
                attached = world.get_attached_object(name)
                if attached is not None:
                    link_pose = self._tool_link_pose(sd, planner.move_group)
                    attached.pose = self._as_mplib_pose(link_pose.inv() * ent.pose)
                    attached.update_pose()
                    continue
                obj = world.get_object(name)
                if obj is not None:
                    from mplib.collision_detection.fcl import FCLObject

                    world.add_object(FCLObject(name, ent.pose, obj.shapes, obj.shape_poses))

    def _planner_world_object(self, planner: Any, ent: Any, name: str) -> str:
        """Put ``ent``'s real collision shapes into ``planner``'s world (idempotent).

        Returns the geometry actually used: ``"shapes"`` for the asset's own
        convex/mesh/box shapes, ``"box"`` when mplib refuses them (it asserts on a
        convex mesh whose scale is not 1, which is most scaled RoboTwin assets) and
        a box of the object's measured extents, in the OBJECT's own frame so that it
        turns with it, is used instead, and ``"already there"`` when the world already
        holds it (``avoid_in_planner`` declared it earlier) and nothing was added.
        """
        import sapien.physx as physx
        from mplib.collision_detection.fcl import Box, CollisionObject, FCLObject
        from mplib.pymp import Pose as MplibPose
        from mplib.sapien_utils.conversion import SapienPlanningWorld

        world = planner.planning_world
        if world.get_object(name) is not None or world.get_attached_object(name) is not None:
            return "already there"
        comp = ent.find_component_by_type(physx.PhysxRigidBaseComponent)
        if comp is not None:
            try:
                fcl_obj = SapienPlanningWorld.convert_physx_component(comp)
            except (AssertionError, TypeError):
                # mplib asserts on a convex mesh whose scale is not 1 and raises TypeError
                # on a shape class it does not know: both mean "fall back to a box".  A
                # broader catch would swallow an mplib API change as a silent downgrade.
                fcl_obj = None
            if fcl_obj is not None:
                world.add_object(fcl_obj)
                return "shapes"
        bounds = self._entity_local_bounds(ent)
        if bounds is None:
            raise RuntimeError(
                f"{ent.get_name()} has no readable collision geometry: the planner cannot "
                "carry or avoid it"
            )
        lo, hi = bounds
        size = np.maximum(hi - lo, 1e-3)
        centre = 0.5 * (lo + hi)
        world.add_object(
            FCLObject(
                name,
                ent.pose,
                [CollisionObject(Box(side=size.tolist()))],
                [MplibPose(p=centre.tolist())],  # the box sits in the OBJECT's frame
            )
        )
        return "box"

    def _entity_local_bounds(self, ent: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """Bounds of an entity's collision geometry in the ENTITY's own frame, so a
        box built from them rotates with the object instead of with the world."""
        world_bounds = self._entity_world_aabb(ent)
        if world_bounds is None:
            return None
        lo: list[np.ndarray] = []
        hi: list[np.ndarray] = []
        inv = np.asarray(ent.pose.inv().to_transformation_matrix(), dtype=np.float64)
        for link in self._collision_bodies(ent):
            try:
                pose = link.get_entity_pose().to_transformation_matrix()
            except Exception:
                try:
                    pose = link.entity.get_pose().to_transformation_matrix()
                except Exception:
                    continue
            for shape in link.get_collision_shapes():
                local = shape.get_local_pose().to_transformation_matrix()
                pts = self._shape_local_bounds(shape)
                if pts is None:
                    continue
                pts = (local[:3, :3] @ pts.T).T + local[:3, 3]
                pts = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
                pts = (inv[:3, :3] @ pts.T).T + inv[:3, 3]
                lo.append(pts.min(axis=0))
                hi.append(pts.max(axis=0))
        if not lo:
            return None
        return np.min(np.asarray(lo), axis=0), np.max(np.asarray(hi), axis=0)

    def plan_carry_in_planner(self, object_name: str, side: Side, carrying: bool = True) -> dict:
        """Tell the motion planner that this hand is CARRYING this object.

        RoboTwin plans every transit as if the jaws were empty (see the note at the
        top of this section), so a held object is swept through the table, the other
        arm and its own target — often stripping it out of the fingers.  This adds
        the object's collision geometry to that arm's planning world and ATTACHES it
        to the gripper link, so from here on every planned path carries it and
        routes around obstacles with it.  The other arm's world gets the same object
        as a live-tracked obstacle, so it stops planning through what you hold.

        Call it right after a successful grasp and pass ``carrying=False`` after the
        release (or before a deliberate contact — a hand-over, seating a lid — where
        the payload MUST touch something and an attached payload would make every
        approach unplannable).

        **On the mplib this checkout ships (0.2.1) the call REFUSES**, with the
        measurement in the message: that build places an attached body from the
        arm's PINOCCHIO link pose, which is in the arm's base frame, while the arm's
        own collision links and every obstacle are in the world frame.  A payload
        attached with a correctly measured relative pose still lands ~1 m away (it
        collided with the wall and made every later plan fail with "IK Failed"), and
        no static relative pose can compensate, because the error depends on the
        configuration.  The check is a live one, so the call starts working by
        itself if the placement is ever fixed.  Until then: ``avoid_in_planner`` for
        the obstacles, lift the payload clear before travelling, and
        ``check_grasp`` to see whether it is still in the jaws.

        Args:
            object_name: see list_objects().  Must be a rigid prop.
            side: the arm that holds it, 'left' or 'right'.
            carrying: False detaches and stops tracking it again.

        Returns:
            ``{'carrying', 'object', 'arm', 'link', 'geometry', 'start_colliding',
            'colliding_with', 'world_objects'}``.  ``geometry`` is 'shapes' or the
            'box' fallback; ``start_colliding`` is the honest warning that the
            payload is ALREADY intersecting something (usually the table it was just
            lifted from): while that is true every plan for this arm fails, so lift
            clear first, or carry only for the transit.
        """
        if side not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {side!r}")
        planner = self._mplib_planner(side)
        if planner is None:
            raise RuntimeError(
                "this embodiment plans with curobo, which has no scene planning world "
                "here: carry_in_planner only works for the mplib planners"
            )
        ent = self._scene_entity(object_name)
        if hasattr(ent, "get_qpos"):
            raise RuntimeError(
                f"{object_name} is articulated; only a rigid object can be carried in the "
                "planning world"
            )
        name = self._mplib_object_name(ent)
        other: Side = "right" if side == "left" else "left"
        if not carrying:
            for sd in (side, other):
                pl = self._mplib_planner(sd)
                if pl is None:
                    continue
                world = pl.planning_world
                if world.get_attached_object(name) is not None:
                    world.detach_object(name, also_remove=True)
                elif world.get_object(name) is not None:
                    world.remove_object(name)
                self._planner_tracked.get(sd, {}).pop(name, None)
                if not self._planner_tracked.get(sd):
                    self._planner_tracked.pop(sd, None)
            return {
                "carrying": False,
                "object": object_name,
                "arm": side,
                "link": None,
                "geometry": None,
                "start_colliding": False,
                "colliding_with": [],
                "world_objects": list(planner.planning_world.get_object_names()),
            }
        pre_declared = planner.planning_world.get_object(name) is not None
        self._planner_tracked.setdefault(side, {})[name] = ent
        self._sync_planner_world(side)  # live arm qpos before the relative pose is taken
        geometry = self._planner_world_object(planner, ent, name)
        art_name = planner.planning_world.get_planned_articulations()[0].name
        link_id = int(planner.move_group_link_id)
        link_pose = self._tool_link_pose(side, planner.move_group)
        planner.planning_world.attach_object(
            name, art_name, link_id, self._as_mplib_pose(link_pose.inv() * ent.pose)
        )
        # Does mplib put the payload where the hand really is?  Measured 2026-08-27 on
        # mplib 0.2.1: it does NOT.  ``set_base_pose`` moves the arm's FCL collision
        # links into the world frame (verified: the box in the jaws collides with
        # link7/link8 at its true world pose), but an AttachedBody is placed from the
        # PINOCCHIO link pose, which is in the arm's BASE frame — so a correctly
        # measured relative pose still lands the payload ~1 m away, where it collided
        # with the wall and made every plan for that arm fail with "IK Failed".  A
        # static relative pose cannot compensate (the error is A^-1·B·A, a function of
        # the configuration), so refuse loudly instead of silently planning nonsense.
        attached = planner.planning_world.get_attached_object(name)
        drift = float(
            np.linalg.norm(
                np.asarray(attached.get_attached_link_global_pose().p, dtype=np.float64)
                - np.asarray(link_pose.p, dtype=np.float64)
            )
        )
        if drift > 1e-3:
            # keep an obstacle the caller had already declared: `also_remove=True` used to
            # drop it out of this arm's world entirely, so the arm silently went back to
            # planning straight through it while the other arm still avoided it
            planner.planning_world.detach_object(name, also_remove=not pre_declared)
            if not pre_declared:
                self._planner_tracked.get(side, {}).pop(name, None)
            if not self._planner_tracked.get(side):
                self._planner_tracked.pop(side, None)
            raise RuntimeError(
                f"this mplib build cannot carry a payload: it places objects attached to "
                f"the {side} arm in the arm's BASE frame, {drift:.3f} m from the hand "
                f"({attached.get_attached_link_global_pose().p} vs the real link at "
                f"{list(link_pose.p)}), so every plan would be checked against a phantom. "
                "Use avoid_in_planner() for the obstacles, lift the payload clear before "
                "you travel, and verify what you hold with check_grasp()."
            )
        pl_other = self._mplib_planner(other)
        if pl_other is not None:
            self._planner_tracked.setdefault(other, {})[name] = ent
            self._planner_world_object(pl_other, ent, name)
            self._sync_planner_world(other)
        # only collisions the PAYLOAD is part of: the hand may be resting on
        # something for reasons that have nothing to do with what it carries
        collisions = [
            c
            for c in planner.planning_world.check_collision()
            if name in (c.object_name1, c.object_name2)
        ]
        return {
            "carrying": True,
            "object": object_name,
            "arm": side,
            "link": planner.move_group,
            "geometry": geometry,
            "start_colliding": bool(collisions),
            "colliding_with": sorted(
                {c.object_name2 if c.object_name1 == name else c.object_name1 for c in collisions}
            ),
            "world_objects": list(planner.planning_world.get_object_names()),
        }

    def plan_avoid_in_planner(self, object_name: str, avoid: bool = True) -> dict:
        """Make the motion planner SEE a task object it was never told about.

        RoboTwin's planning world is built before the props exist and is never
        refreshed, so the planner knows only the table, the wall, the ground and the
        arms: a transit plans straight through a bottle standing in the way.  This
        adds the object (both arms' worlds) with its live pose kept up to date, so
        plans route around it.  ``avoid=False`` removes it again.

        Cost: a few milliseconds per planner call for the pose refresh; the planning
        itself gets slower the more objects the world holds, and an object standing
        where you must reach can make a pose unplannable that used to plan — which is
        the point, but it means adding everything in the scene is rarely wise.
        """
        ent = self._scene_entity(object_name)
        if hasattr(ent, "get_qpos"):
            raise RuntimeError(
                f"{object_name} is articulated; the planning world here can only take rigid "
                "objects (articulations must be added at reset)"
            )
        name = self._mplib_object_name(ent)
        out: dict[str, Any] = {"object": object_name, "avoided": bool(avoid), "arms": {}}
        for sd in ("left", "right"):
            planner = self._mplib_planner(sd)
            if planner is None:
                continue
            world = planner.planning_world
            if avoid:
                self._planner_tracked.setdefault(sd, {})[name] = ent
                out["arms"][sd] = self._planner_world_object(planner, ent, name)
                self._sync_planner_world(sd)
            else:
                if world.get_attached_object(name) is not None:
                    world.detach_object(name, also_remove=True)
                elif world.get_object(name) is not None:
                    world.remove_object(name)
                self._planner_tracked.get(sd, {}).pop(name, None)
                if not self._planner_tracked.get(sd):
                    self._planner_tracked.pop(sd, None)
                out["arms"][sd] = "removed"
        out["world_objects"] = sorted(
            set().union(
                *(
                    set(self._mplib_planner(sd).planning_world.get_object_names())
                    for sd in ("left", "right")
                    if self._mplib_planner(sd) is not None
                )
            )
        )
        return out

    def plan_planner_world(self, side: Side = "left") -> dict:
        """What the motion planner actually knows about the scene (observation only).

        Use it to see the gap between the world and the planner: ``objects`` are the only
        PROPS a plan avoids — everything in ``articulations`` (both arms) is
        collision-checked as well, mplib just keeps it in a different list — and
        ``attached`` is what it believes the hand carries.  Returns ``{"arm", "planner",
        "objects", "attached", "articulations", "tracked_live", "scene_objects",
        "colliding_now"}``.
        """
        planner = self._mplib_planner(side)
        if planner is None:
            return {
                "arm": side,
                "planner": "curobo",
                "objects": [],
                "attached": [],
                "articulations": [],
                "tracked_live": [],
                "scene_objects": self.list_objects(),
                "colliding_now": [],
            }
        world = planner.planning_world
        # the world's arm configuration is only as fresh as the last plan that ran
        # in it, so put the live one back before reporting what collides
        self._sync_planner_world(side)
        names = list(world.get_object_names())
        return {
            "arm": side,
            "planner": "mplib",
            "objects": names,
            "attached": [n for n in names if world.is_object_attached(n)],
            "articulations": list(world.get_articulation_names()),
            "tracked_live": sorted(self._planner_tracked.get(side, {})),
            "scene_objects": self.list_objects(),
            "colliding_now": [
                [c.object_name1, c.link_name1, c.object_name2, c.link_name2]
                for c in world.check_collision()
            ],
        }

    def _tool_point_offset(self, side: Side) -> np.ndarray:
        """Offset from the END-EFFECTOR frame to the point between the fingertips,
        expressed in the tool frame, READ FROM THE ROBOT at call time rather than written
        down — so it follows RoboTwin's frame convention wherever that convention changes.
        It is NOT a measurement of the fingers: RoboTwin builds both poses from one
        ``ee_pose`` and one rotation, differing only by ``_trans_from_gripper_to_endlink``,
        so on this stack it is [0.12, 0, 0] m for every embodiment and every
        ``gripper_bias`` (Dual Piper, both arms, 2026-08-27).  A held object hangs further
        out than that (``piper.yaml``: about 7 cm beyond, for a box).
        """
        ee = np.asarray(self.get_arm_pose(side), dtype=np.float64)
        getter = (
            self.task.robot.get_left_tcp_pose
            if side == "left"
            else self.task.robot.get_right_tcp_pose
        )
        tcp = np.asarray(getter(), dtype=np.float64)[:3]
        rot = t3d.quaternions.quat2mat(ee[3:7])
        return rot.T @ (tcp - ee[:3])

    @staticmethod
    def _approach_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
        """Unit approach (tool) direction from a compass heading and a dive angle:
        elevation 90 points straight down, 0 is horizontal."""
        e = np.deg2rad(float(elevation_deg))
        a = np.deg2rad(float(azimuth_deg))
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])

    @staticmethod
    def _quat_with_approach(quat_wxyz: Any, approach: Any, roll_deg: float = 0.0) -> np.ndarray:
        """The orientation nearest ``quat_wxyz`` whose tool axis (+x) points along
        ``approach``, then rolled about that axis by ``roll_deg``.

        Nearest, not absolute: it keeps whatever wrist roll the hand already has,
        which is both what the arm can reach most cheaply and — while carrying — the
        smallest disturbance to the load.
        """
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        q /= np.linalg.norm(q)
        target = np.asarray(approach, dtype=np.float64).reshape(3)
        target = target / np.linalg.norm(target)
        current = t3d.quaternions.quat2mat(q) @ np.array([1.0, 0.0, 0.0])
        axis = np.cross(current, target)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.dot(current, target))
        if sin_a < 1e-9:
            if cos_a > 0:
                align = np.array([1.0, 0.0, 0.0, 0.0])  # already pointing there
            else:  # exactly opposite: any perpendicular axis turns it around
                perp = np.cross(current, [0.0, 0.0, 1.0])
                if float(np.linalg.norm(perp)) < 1e-9:
                    perp = np.cross(current, [0.0, 1.0, 0.0])
                align = t3d.quaternions.axangle2quat(perp / np.linalg.norm(perp), np.pi)
        else:
            align = t3d.quaternions.axangle2quat(axis / sin_a, np.arctan2(sin_a, cos_a))
        out = t3d.quaternions.qmult(align, q)
        if roll_deg:
            out = t3d.quaternions.qmult(
                out, t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], np.deg2rad(float(roll_deg)))
            )
        return np.asarray(out, dtype=np.float64)

    def _ik_reachable(self, side: Side, pose7: Any) -> str:
        """IK-only reachability of a GRIPPER pose: 'Success' or the planner's reason.

        ~1-30 ms (mplib's own CLIK over 20 random seeds, collision-checked against
        the planning world), against seconds for a full path plan — which is what
        makes an orientation sweep affordable at all.  It answers "is there a joint
        solution", NOT "is there a path": confirm the winner with a real plan.
        """
        planner = self._mplib_planner(side)
        if not self._ik_route_available(side):
            return self._plan_path_status(side, pose7)[0]
        if self._planner_tracked:
            # a declared payload is collision-checked here too, so it must be where
            # the hand actually holds it (IK leaves the world's arm at its last
            # solution; the sync puts the live qpos back before reading the link)
            self._sync_planner_world(side)
        from mplib import Pose as MplibPose

        entity = self.task.robot.left_entity if side == "left" else self.task.robot.right_entity
        end = self.task.robot._trans_from_gripper_to_endlink(list(pose7), arm_tag=side)
        goal = planner._transform_goal_to_wrt_base(
            MplibPose(p=np.asarray(end.p, dtype=np.float64), q=np.asarray(end.q, dtype=np.float64))
        )
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        qpos = np.clip(qpos, planner.joint_limits[:, 0], planner.joint_limits[:, 1])
        status, _ = planner.IK(goal, planner.pad_move_group_qpos(qpos), return_closest=True)
        return str(status)

    def plan_reach_probe(
        self,
        side: Side,
        position: Any,
        at: Literal["fingertips", "ee"] = "fingertips",
        elevations: Any = (90, 75, 60, 45, 30, 15),
        azimuths: Any = None,
        rolls: Any = (0,),
        max_probes: int = 120,
        confirm: int = 1,
    ) -> dict:
        """Which tool ORIENTATIONS put this hand on this point? — sweeping PITCH.

        The dual-arm envelope measured with a top-down tool is narrow and
        misleading: the end-effector frame sits a hand's length back along the tool
        axis, so tipping the tool moves the WRIST several centimetres for the same
        fingertip point.  Measured on the RoboTwin table (2026-08-27): fingertips at
        (-0.05, -0.05, 0.90) and (-0.10, -0.10, 0.90) have NO top-down solution for
        either arm, and both are solvable with the tool tipped 15-30 degrees off
        vertical.  ``check_reachable_*`` and ``goto_pose_flex_*`` sweep wrist yaw
        only, which is the wrong degree of freedom for this, and each of their
        probes costs a rehearsed motion; this sweep is IK-only and free of motion.

        Nothing moves and nothing is touched — no rehearsal, no checkpoint.

        Args:
            side: 'left' or 'right'.
            position: (3,) world point to put the FINGERTIPS on (``at='fingertips'``,
                the default; the offset to the end-effector frame is read from the
                robot at call time — it follows RoboTwin's tcp convention, it is not a
                measurement of THIS hand's fingers) or the end-effector frame itself
                (``at='ee'``).  The wrist ROLL not fixed by (elevation, azimuth, roll) is
                inherited from the hand's current orientation, so the same request can
                answer differently from a different arm pose.
            elevations: dive angle(s) of the approach axis in degrees — 90 straight
                down, 0 horizontal, negative points upward.  Tried in the order
                given, so the first success is the one you asked for most.
            azimuths: compass heading(s) of the approach in the world xy plane
                (degrees, atan2(y, x)).  None (default) = the heading from THIS
                arm's own base toward the point, i.e. reaching straight out.
            rolls: extra rotation(s) about the tool axis in degrees — the degree of
                freedom the flex/yaw helpers sweep.
            max_probes: hard bound on IK probes (call COUNT, not wall clock).
            confirm: how many of the successful orientations to re-check with a real
                path plan (a plan costs seconds where an IK probe costs milliseconds,
                but IK success only means the pose is in the envelope — it does not
                promise a collision-free path).  0 to skip.

        Returns:
            ``{'reachable', 'best', 'tried', 'probes_used', 'plans_used',
            'budget_exhausted', 'tool_offset_m', 'method'}``.  ``best`` is
            ``{'elevation', 'azimuth', 'roll', 'quat_wxyz', 'ee_position',
            'confirmed'}`` for the first orientation that passed (and, when
            ``confirm`` > 0, planned) — feed ``ee_position``/``quat_wxyz`` straight
            to goto_pose_* / glide_to_pose_*.
        """
        if side not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {side!r}")
        if at not in ("fingertips", "ee"):
            raise ValueError(f"at must be 'fingertips' or 'ee', got {at!r}")
        point = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            raise ValueError("position must be three finite numbers")
        if int(max_probes) < 1:
            raise ValueError("max_probes must be >= 1")
        elevs = [float(v) for v in np.atleast_1d(elevations)]
        rolls_deg = [float(v) for v in np.atleast_1d(rolls)]
        if not elevs or not rolls_deg:
            raise ValueError("elevations and rolls must each name at least one value")
        ee_now = np.asarray(self.get_arm_pose(side), dtype=np.float64)
        offset = self._tool_point_offset(side) if at == "fingertips" else np.zeros(3)
        if azimuths is None:
            base = np.asarray(self.get_arm_base_pose(side), dtype=np.float64)[:3]
            out = point - base
            azims = [float(np.rad2deg(np.arctan2(out[1], out[0])))]
        else:
            azims = [float(v) for v in np.atleast_1d(azimuths)]
        if not azims:
            raise ValueError("azimuths is empty: nothing to probe")

        # decided once: the same predicate `_ik_reachable` uses, so `method` cannot claim
        # a millisecond IK sweep for a run that fell back to full path plans
        ik_route = self._ik_route_available(side)
        tried: list[dict] = []
        best: dict | None = None
        probes = 0
        skipped = False
        for elev in elevs:
            for azim in azims:
                for roll in rolls_deg:
                    if probes >= int(max_probes):
                        skipped = True
                        break
                    probes += 1
                    quat = self._quat_with_approach(
                        ee_now[3:7], self._approach_direction(elev, azim), roll
                    )
                    rot = t3d.quaternions.quat2mat(quat)
                    ee_pos = point - rot @ offset
                    status = self._ik_reachable(side, [*ee_pos.tolist(), *quat.tolist()])
                    entry = {
                        "elevation": elev,
                        "azimuth": round(azim, 1),
                        "roll": roll,
                        "ik": status,
                        "ee_position": [round(float(v), 4) for v in ee_pos],
                        "quat_wxyz": [round(float(v), 6) for v in quat],
                    }
                    tried.append(entry)
                    if status == "Success" and best is None:
                        best = entry
        # IK left its last solution in the planning world; put the live arm back
        planner = self._mplib_planner(side)
        if planner is not None:
            entity = self.task.robot.left_entity if side == "left" else self.task.robot.right_entity
            qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
            planner.planning_world.set_qpos_all(qpos[planner.move_group_joint_indices])

        plans = 0
        if best is not None and int(confirm) > 0:
            for entry in [e for e in tried if e["ik"] == "Success"][: int(confirm)]:
                plans += 1
                status, _ = self._plan_path_status(
                    side, [*entry["ee_position"], *entry["quat_wxyz"]]
                )
                entry["plan"] = status
                if status == "Success":
                    best = entry
                    break
        return {
            "reachable": best is not None,
            "best": (best | {"confirmed": best.get("plan") == "Success"}) if best else None,
            "tried": tried,
            "probes_used": probes,
            # without the IK route every probe WAS a path plan: charge them
            "plans_used": plans + (0 if ik_route else probes),
            "budget_exhausted": skipped,
            "tool_offset_m": [round(float(v), 4) for v in offset],
            "method": "ik" if ik_route else "plan",
        }

    def plan_place_gently(
        self,
        object_name: str,
        side: Side,
        target_name: str | None = None,
        max_steps: int = 14,
        step: float = 0.015,
        rehearse: bool = True,
    ) -> dict:
        """Lower the held object in small vertical steps until it CONTACTS
        the target (or the table), then open the gripper and retreat
        straight up. Contact-verified release for small / raised surfaces."""
        actor = self.get_actor(object_name)
        support = self.get_actor(target_name).get_name() if target_name else "table"
        contact = False
        for _ in range(max_steps):
            if self.task.check_actors_contact(actor.get_name(), support):
                contact = True
                break
            try:
                self.plan_move_by_displacement(side, z=-step)
            except Exception:
                break
        if not contact:
            contact = bool(self.task.check_actors_contact(actor.get_name(), support))
        pos_before = np.array(actor.get_pose().p, dtype=np.float64)
        cid = self.checkpoint() if rehearse else None
        self.set_gripper(side, 1.0)
        try:
            self.plan_move_by_displacement(side, z=0.08)
        except Exception:
            pass
        # settle verification: objects released on small/raised surfaces
        # often slide off AFTER release — step the sim briefly and report
        # whether the object actually stayed put
        try:
            # delay() counts SIM STEPS (250 Hz), not seconds — the float this
            # used to pass raised TypeError inside the except and the settle
            # never happened, so `settled` was read off an unsettled scene.
            self.task.delay(SETTLE_STEPS)
        except Exception:
            pass
        pos_after = np.array(actor.get_pose().p, dtype=np.float64)
        drift = float(np.linalg.norm(pos_after - pos_before))
        settled = bool(self.task.check_actors_contact(actor.get_name(), support)) and drift < 0.02
        out = {
            "released_on": support,
            "contact_before_release": contact,
            "settled": settled,
            "drift_after_release_m": round(drift, 4),
        }
        if rehearse and not settled and cid is not None:
            # failed release was a rehearsal: rewind to the held state so
            # the caller can immediately try a different release pose
            self.restore(cid)
            out["world_restored_to_pre_release"] = True
        return out

    def plan_place_held_object(
        self,
        object_name: str,
        side: Side,
        target_position: Any,
        hover: float = 0.10,
        release: bool = True,
        keep_orientation: bool = True,
        transit_clearance: float = 0.15,
        accept_xy_m: float = 0.025,
        max_attempts: int = 5,
        disturb_tol: float = 0.02,
        descend_steps: int = 12,
        descend_step: float = 0.008,
        tilts: Any = (0, 20, 35, 50),
        yaws: Any = (0, 20, -20, 45, -45, 90, -90, 135, 180),
    ) -> dict:
        """Put the object you are ALREADY HOLDING down so that its own origin
        lands on ``target_position``, searching wrist orientations until one is
        reachable and setting it down on contact.

        Three things this does that a plain ``place_object`` does not:

        * It aims the OBJECT, not the hand, and it re-measures the hand-to-
          object offset at every waypoint.  The gripper does not hold objects
          at their centre — a box pinched near its top edge hangs ~19 cm from
          the end-effector frame — and it slips and swings in the jaws while
          you carry it, so a single offset measured at pick-up is stale by the
          time you arrive.
        * It sweeps wrist TILT, not just yaw.  A top-down tool puts the
          end-effector 12 cm straight above the target, which is often outside
          the arm's envelope; the same target with the tool pitched 25-35 deg
          puts it 5-6 cm nearer the robot and 1 cm lower.  That alone decides
          whether an arm can reach into an open drawer.  Tilts of the
          orientation the object is already held in are tried first, because a
          small rotation from where the wrist is swings the load least.
        * It lifts the load clear before travelling and glides in straight
          lines instead of planning: the motion planner does not model what is
          in the gripper, so an RRT transit drags the object through whatever
          is in the way.  Measured here: a planned reorientation swept the box
          through the open drawer's side wall and stripped it out of the jaws.

        The approach is a vertical descent from ``hover`` metres above, then
        small steps down until the object stops following the hand — it is
        then resting on something, and opening the jaws sets it down instead
        of dropping it (a 2.4 cm drop skidded 8 cm across a drawer).

        Returns ``{placed, quat_wxyz, tilt_deg, yaw_deg, object_position,
        error_m, xy_error_m, released}``.  Nothing is left disturbed and
        ``placed`` is False when no orientation reaches the target.
        """
        actor = self.get_actor(object_name)
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        try:
            in_hand = bool(self.gripper_contacts(actor, side)["touching"])
        except Exception:
            in_hand = True  # no contact query for this actor type: cannot tell
        ee0 = np.asarray(self.get_arm_pose(side)[:3], dtype=np.float64)
        obj0 = np.asarray(actor.get_pose().p, dtype=np.float64)
        if not in_hand or np.linalg.norm(obj0 - ee0) > 0.30:
            return {
                "placed": False,
                "held": False,
                "error": (
                    f"{object_name} is not in the {side} gripper (no finger contact, "
                    f"{np.linalg.norm(obj0 - ee0):.2f} m from the hand) — grasp_object it first"
                ),
            }

        def grip_vector() -> np.ndarray:
            """Object origin expressed in the tool frame, measured NOW."""
            ee = self.get_arm_pose(side)
            held = np.asarray(actor.get_pose().p, dtype=np.float64)
            return t3d.quaternions.quat2mat(np.asarray(ee[3:7])).T @ (held - np.asarray(ee[:3]))

        held_quat = np.asarray(self.get_arm_pose(side)[3:7], dtype=np.float64)
        base = np.array([-0.5, 0.5, -0.5, -0.5])  # canonical piper top-down
        candidates: list[tuple[float | None, float | None, np.ndarray]] = []
        if keep_orientation:
            candidates.append((None, None, held_quat))
        for tilt in tilts:
            if float(tilt) == 0.0:
                continue
            for signed in (float(tilt), -float(tilt)):
                candidates.append(
                    (
                        signed,
                        None,
                        t3d.quaternions.qmult(
                            t3d.quaternions.axangle2quat([1, 0, 0], np.deg2rad(signed)),
                            held_quat,
                        ),
                    )
                )
        for tilt in tilts:
            pitched = t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([1, 0, 0], np.deg2rad(float(tilt))), base
            )
            for yaw in yaws:
                candidates.append(
                    (
                        float(tilt),
                        float(yaw),
                        t3d.quaternions.qmult(
                            t3d.quaternions.axangle2quat([0, 0, 1], np.deg2rad(float(yaw))),
                            pitched,
                        ),
                    )
                )

        # lift the load clear of everything between here and the target
        guard = float(target[2]) + transit_clearance
        for _ in range(12):
            if float(actor.get_pose().p[2]) >= guard:
                break
            for step in (0.05, 0.03, 0.02):
                try:
                    self.plan_servo_delta(side, z=step)
                    break
                except Exception:
                    continue
            else:
                break

        def travel(position, quat) -> None:
            """Glide (straight line, no swing) when possible; plan only when
            the straight path is blocked."""
            try:
                self.plan_glide_to_pose(side, position, quat)
            except Exception:
                self.plan_move_to_pose(side, position, quat)

        def aim(quat, at) -> np.ndarray:
            return np.asarray(at, dtype=np.float64) - t3d.quaternions.quat2mat(quat) @ grip_vector()

        lift = np.array([0.0, 0.0, hover])
        joints_before = self._articulation_qpos()

        def disturbance() -> float:
            """Largest change to any articulated scene joint since we started.
            A placement that shoves the container it is reaching into has not
            succeeded, however neatly the object sits."""
            worst = 0.0
            now = self._articulation_qpos()
            for name, was in joints_before.items():
                is_now = now.get(name)
                if is_now is None or is_now.shape != was.shape:
                    continue
                worst = max(worst, float(np.max(np.abs(is_now - was))))
            return worst

        best: tuple[float, str, dict] | None = None
        tried = 0
        for tilt, yaw, quat in candidates:
            if tried >= max_attempts:
                break
            cid = self.checkpoint()
            try:
                travel(aim(quat, target) + lift, quat)
                # re-aim from the hover pose: the object has settled into its
                # carried orientation there, so this offset is the honest one
                travel(aim(quat, target), quat)
            except Exception:
                self.restore(cid)
                continue
            # close the loop on the OBJECT: the planner reports success from
            # up to 2 cm away, the wrist settles a degree or two off, and the
            # long lever from tool to object turns that into centimetres
            for _ in range(8):
                residual = target - np.asarray(actor.get_pose().p, dtype=np.float64)
                span = float(np.linalg.norm(residual))
                if span < 0.008:
                    break
                if span > 0.055:
                    residual = residual * (0.055 / span)
                try:
                    self.plan_servo_delta(
                        side, x=float(residual[0]), y=float(residual[1]), z=float(residual[2])
                    )
                except Exception:
                    break
            # set down: step until the object stops following the hand
            stalled = 0
            for _ in range(descend_steps):
                before = float(actor.get_pose().p[2])
                try:
                    self.plan_servo_delta(side, z=-descend_step)
                except Exception:
                    break
                # a dangling object lags and swings, so one lazy step is not
                # proof of support — require two in a row
                stalled = (
                    stalled + 1 if before - float(actor.get_pose().p[2]) < 0.4 * descend_step else 0
                )
                if stalled >= 2:
                    break
            released = False
            if release:
                self.set_gripper_blocking(side, 1.0)
                try:
                    self.task.delay(SETTLE_STEPS)
                except Exception:
                    pass
                for step in (0.05, 0.05):
                    try:
                        self.plan_servo_delta(side, z=step)
                    except Exception:
                        break
                try:
                    self.task.delay(4 * SETTLE_STEPS)
                except Exception:
                    pass
                released = True
            tried += 1
            landed = np.asarray(actor.get_pose().p, dtype=np.float64)
            xy_error = float(np.linalg.norm(landed[:2] - target[:2]))
            moved = disturbance()
            outcome = {
                "placed": True,
                "quat_wxyz": [float(v) for v in quat],
                "tilt_deg": tilt,
                "yaw_deg": yaw,
                "object_position": [float(v) for v in landed],
                "error_m": round(float(np.linalg.norm(landed - target)), 4),
                "xy_error_m": round(xy_error, 4),
                "released": released,
                "attempts": tried,
                "articulation_disturbed": round(moved, 4),
                "accepted": xy_error <= accept_xy_m and moved <= disturb_tol,
            }
            # Judge the orientation by WHERE THE OBJECT ENDED, not by whether
            # the arm could plan to the pose.  Which candidate plans first is
            # sensitive to things that have nothing to do with the placement
            # (the hover height, planner luck), and a steeper tilt that plans
            # fine can still tip the object out of the box on release.
            if outcome["accepted"]:
                return outcome
            # rank by the error the caller cares about, with a disturbed
            # container counting against the candidate
            score = xy_error + max(0.0, moved - disturb_tol)
            if best is None or score < best[0]:
                if best is not None:
                    self._checkpoints.pop(best[1], None)
                best = (score, self.checkpoint(), outcome)
            self.restore(cid)
            self._checkpoints.pop(cid, None)
        if best is not None:
            self.restore(best[1])
            self._checkpoints.pop(best[1], None)
            return best[2]
        return {
            "placed": False,
            "error": (
                f"no wrist orientation puts {object_name} at "
                f"{[round(float(v), 3) for v in target]} for the {side} arm — "
                "the spot is outside this arm's envelope with the object held "
                "the way it is; hand the object to the other arm, or bring "
                "the target closer (open the container further)"
            ),
        }

    def support_height_at(
        self, x: float, y: float, ignore: tuple = (), below_z: float | None = None
    ) -> float | None:
        """Height of the highest collision-AABB top under the world point ``(x, y)``
        among the scene's actors (table, walls, furniture, objects; the robot is not
        in the registry), or ``None`` when no actor's footprint covers the point.

        ``ignore`` names actors to leave out — matched by IDENTITY, so an actor
        listed under both its asset name and its role name ("113_coffee-box" /
        "object") is skipped whichever name you give (it used to be skipped under
        one name and counted under the other, so a held object over the table
        reported its own top as the support).  ``below_z`` skips any actor whose top
        is above that height (ask for the surface under the hand, not the shelf
        above it).  AABB footprints over-approximate rotated or opened objects (an
        open drawer widens the cabinet's box); :meth:`support_under` also tells you
        WHICH actor won.
        """
        hit = self.support_under(x, y, ignore, below_z)
        return None if hit is None else hit["height"]

    def support_under(
        self, x: float, y: float, ignore: tuple = (), below_z: float | None = None
    ) -> dict | None:
        """:meth:`support_height_at` with provenance: ``{"height", "actor"}``."""
        reg = self._actor_registry()
        skip = {id(reg[n]) for n in ignore if n in reg}
        seen: set[int] = set()
        best: dict | None = None
        for name, actor in reg.items():
            if id(actor) in skip or id(actor) in seen:
                continue
            seen.add(id(actor))
            ent = getattr(actor, "actor", actor)
            aabb = self._entity_world_aabb(ent)
            if aabb is None:
                continue
            low, high = aabb
            top = float(high[2])
            if below_z is not None and top > float(below_z):
                continue
            if low[0] <= x <= high[0] and low[1] <= y <= high[1]:
                if best is None or top > best["height"]:
                    best = {"height": top, "actor": name}
        return best

    def plan_transfer_object(
        self,
        object_name: str,
        from_side: Side,
        to_side: Side,
        via_xy: Any | None = None,
        via_z: float | None = None,
    ) -> None:
        """Two-stage relay when one arm cannot reach the goal region: park
        the object at a mid-table point both arms reach, release, then
        regrasp with the other arm.

        The drop height is MEASURED (the top of whatever surface lies under
        the handoff point, plus 1 cm) rather than assumed — the tables in this
        benchmark are not all at the same height, and a hard-coded 0.76 either
        drops the object or drives the hand into the table."""
        via = list(via_xy) if via_xy is not None else [0.0, -0.12]
        if via_z is None:
            hand_z = float(self.get_arm_pose(from_side)[2])
            support = self.support_height_at(
                float(via[0]), float(via[1]), ignore=(object_name,), below_z=hand_z
            )
            if support is None:
                print(
                    f"[robotwin] transfer_object: no actor under ({via[0]:.2f}, {via[1]:.2f}); "
                    "releasing at z 0.76 (a guess — pass via_z)",
                    flush=True,
                )
            via_z = 0.76 if support is None else support + 0.01
        target = [float(via[0]), float(via[1]), float(via_z), 0.0, 1.0, 0.0, 0.0]
        self.plan_place_object(
            object_name, from_side, target, pre_dis=0.08, dis=0.005, constrain="free"
        )
        try:
            self.plan_back_to_origin(from_side)
        except Exception:
            pass
        self.plan_grasp_object(object_name, to_side)

    def get_arm_pose(self, side: Side) -> np.ndarray:
        """End-effector pose in world frame: [x, y, z, qw, qx, qy, qz]."""
        return np.asarray(self.task.get_arm_pose(ArmTag(side)), dtype=np.float64)

    def get_arm_base_pose(self, side: Side) -> np.ndarray:
        """Arm base (URDF root) pose in world frame: [x, y, z, qw, qx, qy, qz]."""
        robot = self.task.robot
        entity = robot.left_entity if side == "left" else robot.right_entity
        pose = entity.get_root_pose() if hasattr(entity, "get_root_pose") else entity.get_pose()
        return np.concatenate([np.asarray(pose.p), np.asarray(pose.q)]).astype(np.float64)

    def endpose_to_ee_link_wxyz_xyz(self, side: Side, position: Any, quat_wxyz: Any) -> np.ndarray:
        """Convert a world-frame RoboTwin endpose into the raw ee-link (link6)
        pose in the arm's base frame, as [qw, qx, qy, qz, x, y, z] for pyroki IK."""
        robot = self.task.robot
        pose7 = self._to_pose7(position, quat_wxyz)
        link_pose = robot._trans_from_gripper_to_endlink(pose7, arm_tag=side)
        entity = robot.left_entity if side == "left" else robot.right_entity
        base = entity.get_root_pose() if hasattr(entity, "get_root_pose") else entity.get_pose()
        rel = base.inv() * link_pose
        return np.concatenate([np.asarray(rel.q), np.asarray(rel.p)]).astype(np.float64)

    def play_expert_once(self) -> dict[str, Any] | None:
        """Run RoboTwin's built-in expert policy (debug/validation only)."""
        info = self.task.play_once()
        self._finish_plan("expert rollout")
        return info

    # ------------------------------ route B: joint-level ------------------------------

    def get_joint_positions(self, side: Side) -> np.ndarray:
        """Measured arm joint positions plus normalized gripper opening (7,)."""
        if side == "left":
            return np.asarray(self.task.robot.get_left_arm_real_jointState(), dtype=np.float64)
        return np.asarray(self.task.robot.get_right_arm_real_jointState(), dtype=np.float64)

    def move_to_joints_blocking(
        self,
        joints: Any,
        side: Side = "left",
        *,
        tolerance: float = 0.01,
        max_steps: int = 800,
    ) -> None:
        """Drive one arm to target joint positions with PD targets (no planning)."""
        robot = self.task.robot
        n = len(robot.left_arm_joints if side == "left" else robot.right_arm_joints)
        target = np.asarray(joints, dtype=np.float64).reshape(n)
        zeros = np.zeros(n)
        for _ in range(max_steps):
            current = self.get_joint_positions(side)[:-1]
            if np.linalg.norm(current - target) < tolerance:
                break
            robot.set_arm_joints(target, zeros, side)
            self._sim_tick()

    def move_to_joints_blocking_both(
        self,
        left_joints: Any,
        right_joints: Any,
        *,
        tolerance: float = 0.01,
        max_steps: int = 800,
    ) -> None:
        robot = self.task.robot
        nl = len(robot.left_arm_joints)
        nr = len(robot.right_arm_joints)
        left = np.asarray(left_joints, dtype=np.float64).reshape(nl)
        right = np.asarray(right_joints, dtype=np.float64).reshape(nr)
        for _ in range(max_steps):
            err_l = np.linalg.norm(self.get_joint_positions("left")[:-1] - left)
            err_r = np.linalg.norm(self.get_joint_positions("right")[:-1] - right)
            if err_l < tolerance and err_r < tolerance:
                break
            robot.set_arm_joints(left, np.zeros(nl), "left")
            robot.set_arm_joints(right, np.zeros(nr), "right")
            self._sim_tick()

    def set_gripper_blocking(self, side: Side, fraction: float, *, steps: int = 150) -> None:
        """Direct PD gripper control. fraction: 0.0 closed .. 1.0 open."""
        fraction = float(np.clip(fraction, 0.0, 1.0))
        robot = self.task.robot
        for _ in range(steps):
            robot.set_gripper(fraction, side)
            self._sim_tick()

    def _sim_tick(self) -> None:
        """One physics step for route-B loops, with video capture."""
        self.task.scene.step()
        self._sim_step_count += 1
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self.task._update_render()
            self._video_tick()

    # ------------------------------ privileged object access ------------------------------

    def _actor_registry(self) -> dict[str, Actor]:
        # Tasks with several instances of the same asset (two cans, three
        # bottles) reuse one asset name, so key those by their attribute name
        # (object1/object2/...) and keep the asset name only when unambiguous.
        by_name: dict[str, list[tuple[str, Actor]]] = {}

        def _add(attr: str, value: Actor) -> None:
            name = (value.get_name() or attr).lower().replace(" ", "_")
            by_name.setdefault(name, []).append((attr, value))

        for attr, value in self.task.__dict__.items():
            if isinstance(value, Actor):
                _add(attr, value)
            elif isinstance(value, sapien.Entity):
                # visual-only markers (target pads, drop zones) are plain
                # sapien Entities, not RoboTwin Actor wrappers — several
                # tasks score against their pose, so they must be visible.
                # Their given names are junk ("box", ""), so key them by
                # the attribute name, which carries the semantics (target,
                # pad, ...).
                by_name.setdefault(attr.lower(), []).append((attr, value))
            elif isinstance(value, (list, tuple)):
                # Several tasks keep their object instances in a list
                # (self.bottles = [...], self.breads = [...]).
                for i, item in enumerate(value):
                    if isinstance(item, (Actor, sapien.Entity)):
                        _add(f"{attr}{i}", item)

        actors: dict[str, Actor] = {}
        for name, entries in by_name.items():
            if len(entries) == 1:
                actors[name] = entries[0][1]
                continue
            for attr, value in entries:
                actors[f"{name}_{attr.lower()}"] = value
        # also expose each actor under its task-attribute name (object,
        # target_object, rack, ...) — assets are drawn randomly per episode,
        # so the attribute name is the only observable carrier of the
        # actor's ROLE in the scene (RoboTwin's own language instructions
        # encode the same information; the bridge has no language channel)
        for name, entries in by_name.items():
            for attr, value in entries:
                actors.setdefault(attr.lower(), value)
        return actors

    def list_objects(self) -> list[str]:
        return sorted(self._actor_registry().keys())

    def get_actor(self, object_name: str) -> Actor:
        actors = self._actor_registry()
        key = object_name.lower().replace(" ", "_")
        if key in actors:
            return actors[key]
        matches = [n for n in actors if key in n or n in key]
        if len(matches) == 1:
            return actors[matches[0]]
        raise KeyError(f"Object '{object_name}' not found. Available: {sorted(actors)}")

    def get_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """World-frame pose: position (3,), quaternion wxyz (4,)."""
        pose = self.get_actor(object_name).get_pose()
        return np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)

    def get_object_functional_pose(self, object_name: str, point_id: int = 0) -> np.ndarray:
        """Functional-point pose [x, y, z, qw, qx, qy, qz] in world frame."""
        actor = self.get_actor(object_name)
        return np.asarray(actor.get_functional_point(point_id), dtype=np.float64)

    # ------------------------------ observation ------------------------------

    def get_observation(self) -> dict[str, Any]:
        raw = self.task.get_obs()
        obs: dict[str, Any] = {}

        for cam_name, cam in raw["observation"].items():
            entry: dict[str, Any] = {"images": {}}
            K = np.asarray(cam["intrinsic_cv"], dtype=np.float64)
            extr = np.asarray(cam["extrinsic_cv"], dtype=np.float64)
            world2cam = np.eye(4)
            world2cam[:3, :4] = extr
            cam2world = np.linalg.inv(world2cam)
            entry["pose_mat"] = cam2world
            entry["pose"] = np.concatenate(
                [cam2world[:3, 3], t3d.quaternions.mat2quat(cam2world[:3, :3])]
            )
            entry["intrinsics"] = K
            if "rgb" in cam:
                entry["images"]["rgb"] = np.asarray(cam["rgb"])
            if "depth" in cam:
                depth_m = np.asarray(cam["depth"], dtype=np.float64) / 1000.0
                entry["images"]["depth"] = depth_m[..., None]
            obs[cam_name] = entry

        robot = self.task.robot
        obs["robot_left_joint_pos"] = np.asarray(
            robot.get_left_arm_real_jointState(), dtype=np.float64
        )
        obs["robot_right_joint_pos"] = np.asarray(
            robot.get_right_arm_real_jointState(), dtype=np.float64
        )
        obs["robot_left_cartesian_pos"] = np.concatenate(
            [self.get_arm_pose("left"), [robot.get_left_gripper_val()]]
        )
        obs["robot_right_cartesian_pos"] = np.concatenate(
            [self.get_arm_pose("right"), [robot.get_right_gripper_val()]]
        )
        return obs

    def get_current_time_s(self) -> float:
        return self._sim_step_count / 250.0  # RoboTwin steps physics at 250 Hz

    # ------------------------------ reward / success ------------------------------

    def compute_reward(self) -> float:
        return 1.0 if self.task_completed() else 0.0

    def task_completed(self) -> bool:
        if getattr(self, "_success_latched", False):
            return True
        try:
            return bool(self.task.check_success())
        except Exception:
            return False

    # ------------------------------ video ------------------------------

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False
    ) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self.task._update_render()
            self._video_tick()

    def _head_camera(self):
        cams = self.task.cameras
        idx = cams.static_camera_name.index("head_camera")
        return cams.static_camera_list[idx]

    def _capture_camera_rgb(self, camera) -> np.ndarray:
        camera.take_picture()
        rgba = camera.get_picture("Color")
        return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)

    def _video_tick(self) -> None:
        """Head-camera frame capture; also patched in as task._take_picture."""
        if self._poll_success and not self._success_latched:
            try:
                if bool(self.task.check_success()):
                    self._success_latched = True
            except Exception:
                pass
        if not self._record_frames:
            return
        self._frame_buffer.append(self._capture_camera_rgb(self._head_camera()))
        if self._record_wrist_camera:
            cams = self.task.cameras
            if getattr(cams, "collect_wrist_camera", False):
                self._wrist_frame_buffer.append(self._capture_camera_rgb(cams.left_camera))

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._wrist_frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        self.task._update_render()
        return self._capture_camera_rgb(self._head_camera())

    def render_wrist(self) -> np.ndarray:
        self.task._update_render()
        return self._capture_camera_rgb(self.task.cameras.left_camera)


__all__ = ["RoboTwinEnv", "list_robotwin_tasks", "ROBOTWIN_ROOT"]
