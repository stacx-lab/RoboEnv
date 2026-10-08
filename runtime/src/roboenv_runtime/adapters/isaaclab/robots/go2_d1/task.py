"""Shared Go2 task configuration, robot measurements and terminal-state mechanics."""

import copy
import math

import numpy as np

from ...registry import get_task
from .controllers import matrix_quat, quat_matrix

CONTROL = "go2-d1-control"
REACH = "go2-d1-reach"
PICK_PLACE = "go2-d1-pick-place"


def task_type(name):
    task = get_task(name)
    if task.embodiment != "go2_d1":
        raise ValueError("Task requires a different robot than Go2–D1")
    return task


def catalog_definition(name):
    """Defaults for direct callers; launched workers receive resolved task_spec."""
    import yaml

    from roboenv_runtime.paths import repo_root

    task_type(name)
    path = repo_root() / "catalog/tasks/isaaclab" / name / "task.yaml"
    return yaml.safe_load(path.read_text())["backend_kwargs"]["task_spec"]


def catalog_tcp():
    from .environment import load_profiles

    return gripper_tcp(load_profiles()["gripper"])


def gripper_tcp(profile):
    """Validate the Go2 gripper mount and resolve its TCP before starting Isaac."""
    from roboenv_runtime.adapters.urdf_geometry import matrix

    mount = profile["extrinsics"]["mount"]
    tcp = profile["intrinsics"]["frames"]["tcp"]
    if tcp["parent_frame"] is not None or tcp["rotation_xyzw"] != [0, 0, 0, 1]:
        raise ValueError("Go2 task TCP requires an aligned local TCP")
    if (mount["parent"], mount["parent_frame"], mount["rotation_xyzw"]) != (
        "arm", "flange", [0, 0, 0, 1],
    ):
        raise ValueError("Go2 task TCP requires an aligned gripper mount on the arm flange")
    return (matrix(mount["translation_m"] + mount["rotation_xyzw"])
            @ matrix(tcp["translation_m"] + tcp["rotation_xyzw"]))[:3, 3].tolist()


def validate_numeric_spec(spec, vectors=None, nonpositive_fields=()):
    """Optional numeric-field validation for tasks with metric configurations."""
    vectors = {"tcp_in_flange_m": 3, **(vectors or {})}
    for key, value in spec.items():
        values = value if key in vectors else [value]
        if key in vectors and (not isinstance(value, list) or len(value) != vectors[key]):
            raise ValueError(f"{key} requires {vectors[key]} finite numbers")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"{key} must contain finite numbers")
        if key.endswith("_steps") and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key not in vectors and key not in nonpositive_fields and value <= 0:
            raise ValueError(f"{key} must be positive")


class Go2Task:
    """Task-owned scene/scorer composed with the reusable Go2 environment."""

    embodiment = "go2_d1"
    fields = set()
    contact_sensors = False
    payload_mass_field = None

    @classmethod
    def create_environment(cls, **kwargs):
        from .environment import Go2Env

        return Go2Env(task_name=cls.name, **kwargs)

    @classmethod
    def definition(cls, task_spec=None, tcp_in_flange_m=None):
        spec = copy.deepcopy(catalog_definition(cls.name) if task_spec is None else task_spec)
        if not isinstance(spec, dict) or set(spec) != cls.fields:
            raise ValueError(f"Invalid {cls.name} task_spec fields; expected {sorted(cls.fields)}")
        spec["tcp_in_flange_m"] = copy.deepcopy(
            catalog_tcp() if tcp_in_flange_m is None else tcp_in_flange_m
        )
        validate_numeric_spec({"tcp_in_flange_m": spec["tcp_in_flange_m"]})
        cls.validate(spec)
        return {"name": cls.name, "version": 1, **spec}

    @staticmethod
    def validate(spec):
        pass


def definition(name, task_spec=None, tcp_in_flange_m=None):
    return task_type(name).definition(task_spec, tcp_in_flange_m)


def make_state(name, task_spec=None, tcp_in_flange_m=None):
    return task_type(name).state_type(name, task_spec, tcp_in_flange_m)


class Go2TaskState:
    """Evaluates measured physics only; never writes a robot or object pose."""

    def __init__(self, name, task_spec=None, tcp_in_flange_m=None):
        self.spec = definition(name, task_spec, tcp_in_flange_m)
        self.completed = False
        self.failure = None
        # Retain the shared Go2 journal state shape across the existing task kinds.
        self.placement_started = False
        self.hold = self.contact_loss = self.lift_hold = 0
        self.pickup_base = None
        self.carried_distance = 0.0
        self.metrics = {}

    @property
    def done(self):
        return self.completed or self.failure is not None

    def update(self, sample):
        if self.done:
            return
        if sample["fallen"]:
            self.failure = "robot_fallen"
            return
        self.evaluate(sample)

    def evaluate(self, sample):
        pass

    def snapshot(self):
        return copy.deepcopy(vars(self))


class Go2Scene:
    """Robot measurements shared by Go2 tasks; the scheduler owns all stepping."""

    def __init__(self, native, name):
        self.native, self.name = native, name
        self.spec = copy.deepcopy(native.task_spec)
        self.build()

    def spectator_pose(self):
        """None selects the embodiment's translation-following view."""
        return None

    def build(self):
        pass

    def reset(self):
        self.state = make_state(
            self.name,
            {k: v for k, v in self.spec.items() if k not in {"name", "version", "tcp_in_flange_m"}},
            self.spec["tcp_in_flange_m"],
        )

    def physics_update(self):
        pass

    def sample(self, fallen):
        n = self.native
        spec = self.spec
        root = n.root_pose()
        transform = np.eye(4)
        transform[:3, :3] = quat_matrix(root[3:])
        transform[:3, 3] = root[:3]
        flange = transform @ n.model.forward(n.joints())["d1_Link6"]
        tcp = flange[:3, 3] + flange[:3, :3] @ np.asarray(spec["tcp_in_flange_m"])
        sample = {
            "fallen": fallen,
            "base_position": root[:3],
            "base_speed": float(np.linalg.norm(n.array(n.robot.data.root_lin_vel_b))),
            "tcp_position": tcp,
            "tcp_quaternion": matrix_quat(flange[:3, :3]),
        }
        return sample

    def update(self, fallen):
        self.state.update(self.sample(fallen))

    def snapshot(self):
        return {"task": self.state.snapshot()}
