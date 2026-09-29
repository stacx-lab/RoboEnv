"""Native RoboTwin scene creation and reset for the typed device world."""

from __future__ import annotations

import importlib
import os
import sys
import time
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
ArmTag = None
_ROBOTWIN_IMPORTED = False


def _ensure_robotwin_imported() -> None:
    global _ROBOTWIN_IMPORTED, ArmTag
    if _ROBOTWIN_IMPORTED:
        os.chdir(ROBOTWIN_ROOT)  # keep CWD stable for repeated env builds
        return
    os.chdir(ROBOTWIN_ROOT)
    from envs.utils.action import ArmTag as _ArmTag  # noqa: PLC0415

    ArmTag = _ArmTag
    _ROBOTWIN_IMPORTED = True


_CONFIGS_PATH = os.path.join(ROBOTWIN_ROOT, "env_cfg", "task_config")

Side = Literal["left", "right"]


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
    """Native construction shared by the typed device implementation."""

    def __init__(
        self,
        task_name: str = "handover_block",
        task_config: str = "demo_clean",
        embodiment: list[Any] | None = None,
        planner: str | None = "mplib_RRT",
        seed: int | None = None,
        collect_wrist_camera: bool = True,
        save_freq: int = 25,
        allow_seed_substitution: bool = True,
    ) -> None:
        self.seed = seed
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

        task_module = importlib.import_module(f"envs.{task_name}")
        self._task_cls = getattr(task_module, task_name)
        self.task = None

        self._env_active = False
        self._sim_step_count = 0

        self._record_frames = False
        self._record_wrist_camera = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []

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
                    attempts.append(
                        {"seed": try_seed, "attempt": attempt, "error": type(exc).__name__}
                    )
                    last_exc = exc
                    self._close_task()
                    print(f"[robotwin] seed {try_seed} unstable, trying {try_seed + 1}", flush=True)
                    break
                except Exception as exc:
                    attempts.append(
                        {"seed": try_seed, "attempt": attempt, "error": type(exc).__name__}
                    )
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
            "requested_seed": seed,
            "effective_seed": try_seed,
            "attempts": attempts,
            "seed_substitution_allowed": self.allow_seed_substitution,
            "expert_prepass": True,
            "expert_error": expert_error,
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

        self._sim_step_count = 0
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

    def get_arm_pose(self, side: Side) -> np.ndarray:
        """End-effector pose in world frame: [x, y, z, qw, qx, qy, qz]."""
        return np.asarray(self.task.get_arm_pose(ArmTag(side)), dtype=np.float64)

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
