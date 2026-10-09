"""One native ABC world using shared device control and command evidence."""

import copy
import os
import sys
from pathlib import Path

from roboenv_runtime.paths import repo_root

SOURCE = repo_root() / "third_party/abc"
if not SOURCE.is_dir():
    raise ModuleNotFoundError("ABC source is missing; run roboenv prepare abc")
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import numpy as np

from roboenv_runtime.adapters.device_control import DeviceControl
from roboenv_runtime.adapters.native_record import compare, digest, initialize_record
from .adapter import frame_budget
from .rig import CAMERA_SIZE, aperture, describe, rig_id


class ABCEnv:
    embodiment = "yam"
    unbounded = False

    def __init__(
        self,
        task_name="put_plastic_bottles_in_bin",
        *,
        randomize=True,
        randomization=None,
        max_steps=None,
        camera_backend="mujoco",
    ):
        from roboenv.provisioning.gpu import configure_gpu

        configure_gpu("abc")
        import abc_sim
        from abc_sim.task_registry import get_task_physics_defaults

        if not Path(abc_sim.__file__).resolve().is_relative_to(SOURCE.resolve()):
            raise ImportError("ABC imported outside its pinned source")
        self._abc_sim = abc_sim

        self.spec = abc_sim.get_task_spec(task_name)
        self.task_name = self.spec.name
        if camera_backend not in {"mujoco", "blender"}:
            raise ValueError("ABC camera_backend must be mujoco or blender")
        self.camera_backend = camera_backend
        self.blender_settings = None
        if camera_backend == "blender":
            from .rendering import settings

            self.blender_settings = settings()
        if type(randomize) is not bool or (
            randomization is not None and not isinstance(randomization, dict)
        ):
            raise ValueError("Expected randomize boolean and optional randomization object")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if os.environ.get("DEPLOY_INIT_Q"):
            raise ValueError("DEPLOY_INIT_Q is unsupported: use the pinned ABC home configuration")
        self.randomize, self.randomization = randomize, copy.deepcopy(randomization)
        self.max_steps = max_steps
        defaults = get_task_physics_defaults(self.spec.env_task)
        self.control_interval = defaults.get("physics_dt", 0.002) * defaults.get(
            "control_decimation", 17
        )
        self.env = self.renderer = self.control = None
        self.asset_roots = {
            p.name: p.resolve()
            for p in sorted((SOURCE / "abc_sim/models/assets").iterdir())
            if p.is_dir()
        }
        self._camera_size = CAMERA_SIZE
        self._camera_names = ("top", "left", "right")

    def reset(self, *, seed=0):
        from .scoring import supplemental_evaluator

        self.close()
        self.seed = seed
        self.env = self._abc_sim.make_env(
            task=self.task_name,
            render_cameras=False,
            terminate_on_success=False,
            **({"seed": seed} if self.spec.env_task == "inhand_transfer" else {}),
        )
        self.env.reset(
            seed=seed,
            randomize=self.randomize,
            options={"randomization": copy.deepcopy(self.randomization)},
        )
        self.control_interval = float(self.env.model.opt.timestep) * self.env._control_decimation
        self.frame_limit = (
            None
            if self.unbounded
            else (
                self.max_steps
                if self.max_steps is not None
                else frame_budget(self.control_interval)
            )
        )
        self.env.max_episode_steps = self.frame_limit
        self.device_facts = describe(self.env.model)
        self.rig_identity = rig_id(self.device_facts)
        self.supplemental = supplemental_evaluator(self.task_name, self.env.model)
        targets, limits = {}, {}
        state = self.env.get_state()
        for offset, side in ((0, "left"), (7, "right")):
            arm, grip = side + "_arm", side + "_gripper"
            targets[arm] = state[offset : offset + 6].astype(float).tolist()
            targets[grip] = [float(state[offset + 6])]
            limits[arm] = [
                tuple(self.env.model.joint(f"{side}_joint{i}").range) for i in range(1, 7)
            ]
            limits[grip] = [(0, 1)]
        self.control = DeviceControl(
            limits,
            targets,
            self._apply,
            self._advance,
            exhausted=self._exhausted,
            interval_s=self.control_interval,
        )
        self._evaluate()
        if self.camera_backend == "blender":
            from .rendering import BlenderImages

            # Cold startup belongs to launch/reset, not a camera request deadline.
            self.renderer = BlenderImages(self)
        self.initialization = {
            "requested_seed": seed,
            "rig": self.rig_identity,
            "control_interval_s": self.control_interval,
            "native_timestep_s": float(self.env.model.opt.timestep),
            "control_decimation": self.env._control_decimation,
            "initial_time_s": self.timestamp,
            "randomization_sha256": digest(self._randomization_info()),
            "rendering": (self.blender_settings["identity"] if self.blender_settings
                          else {"backend": "mujoco"}),
        }
        initialize_record(
            self,
            {
                "task_name": self.task_name,
                "randomize": self.randomize,
                "randomization": self.randomization,
                "max_steps": self.max_steps,
                "camera_backend": self.camera_backend,
            },
            "abc",
        )

    def _randomization_info(self):
        value = self.env._last_randomization
        return plain(vars(value)) if value is not None else None

    @property
    def instruction(self):
        if self.supplemental is not None:
            return self.supplemental.instruction
        return self.env.prompt

    @property
    def timestamp(self):
        return float(self.env.data.time)

    def _exhausted(self):
        return (
            self.frame_limit is not None
            and self.control is not None
            and self.control.frame >= self.frame_limit
        )

    def _apply(self, goals):
        action = []
        for side in ("left", "right"):
            action.extend(goals[side + "_arm"])
            action.extend(goals[side + "_gripper"])
        self._action = np.asarray(action, dtype=np.float32)
        # Apply references immediately for stop, without advancing physics.
        for i, actuator in enumerate(self.env._ctrl_indices):
            self.env.data.ctrl[actuator] = self._action[i] * (0.0475 if i in (6, 13) else 1.0)

    def _advance(self):
        _, _, _, _, info = self.env.step(self._action, render_obs=False)
        self._evaluate(info)
        # mj_step integrates qpos after computing transforms. Refresh derived
        # observations after native evaluation, without advancing dynamics or
        # changing the solver's warm-start state.
        import mujoco

        mujoco.mj_kinematics(self.env.model, self.env.data)
        mujoco.mj_comPos(self.env.model, self.env.data)
        mujoco.mj_comVel(self.env.model, self.env.data)
        mujoco.mj_camlight(self.env.model, self.env.data)

    def _evaluate(self, info=None):
        if self.supplemental is not None:
            self.evaluation = self.supplemental.evaluate(self.env.data)
        else:
            if info is None:
                result = self.env.evaluate_task()
                if result is None:
                    raise ValueError("ABC task has no native or supplemental evaluator")
                result = result.to_info(squeeze=True)
            else:
                result = info["task_eval"]
            self.evaluation = {**plain(result), "scoring": "abc-native"}

    def task_completed(self):
        return bool(self.evaluation["success"])

    def device_write(self, name, target):
        if name.endswith("_gripper") and (
            not isinstance(target, list)
            or len(target) != 1
            or type(target[0]) not in (int, float)
            or not 0 <= target[0] <= 1
        ):
            raise ValueError("Gripper target must be in [0, 1]")
        return self.control.write(name, target)

    def device_stop(self, name):
        target = (
            self.control.targets[name]
            if name.endswith("_gripper")
            else self.device_read(name)["positions"]
        )
        return self.control.stop(name, target)

    def device_read(self, name):
        side = name.split("_")[0]
        result = {"faults": [], "timestamp": self.timestamp}
        if name not in self.control.targets:
            raise ValueError("Unknown ABC actuator")
        if name.endswith("_gripper"):
            return {**result, "aperture_m": aperture(self.env.model, self.env.data, side)}
        joints = [self.env.model.joint(f"{side}_joint{i}") for i in range(1, 7)]
        return {
            **result,
            "positions": [float(self.env.data.qpos[j.qposadr[0]]) for j in joints],
            "velocities": [float(self.env.data.qvel[j.dofadr[0]]) for j in joints],
            "efforts": [],
        }

    def device_image(self, name):
        if name not in ("camera_top", "camera_left", "camera_right"):
            raise ValueError("Unknown ABC camera")
        if self.camera_backend == "blender":
            if self.renderer is None:
                from .rendering import BlenderImages

                self.renderer = BlenderImages(self)
            return self.renderer.read()[name.removeprefix("camera_")].copy(), None
        if self.renderer is None:
            import mujoco

            self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        self.renderer.update_scene(self.env.data, camera=name.removeprefix("camera_"))
        return self.renderer.render().copy(), None

    def owner_state(self):
        data = self.env.data
        return {
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "act": data.act.tolist(),
            "ctrl": data.ctrl.tolist(),
            "time": self.timestamp,
            "runtime": plain(self.env.task_runtime_debug_state()),
            "evaluation": copy.deepcopy(self.evaluation),
            "instruction": self.instruction,
        }

    def checkpoint(self):
        return {
            "seed": self.seed,
            "frame": self.control.frame,
            "commands": copy.deepcopy(self.control.commands),
            "state": self.owner_state(),
        }

    def restore(self, token):
        # Deterministic command reconstruction also restores hidden native runtime,
        # evaluator, RNG and model state without copying renderer/native handles.
        self.reset(seed=token["seed"])
        for event in token["commands"]:
            while self.control.frame < event["frame"]:
                self.control.tick()
            if event["op"] == "write":
                for name, target in event["commands"].items():
                    self.device_write(name, target)
            elif event["op"] == "stop":
                self.device_stop(event["device"])
            else:
                raise ValueError("Unsupported checkpoint command")
        while self.control.frame < token["frame"]:
            self.control.tick()
        compare(token["state"], self.owner_state(), atol=1e-8)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.env is not None:
            self.env.close()
            self.env = None
        self.control = None


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
