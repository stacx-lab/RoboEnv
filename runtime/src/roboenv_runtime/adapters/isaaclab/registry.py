"""Explicit task factories; discovery never imports the Isaac simulator."""

from importlib import import_module
from typing import Protocol

from .common.environment import IsaacEnvironment


class TaskFactory(Protocol):
    name: str
    embodiment: str

    @classmethod
    def create_environment(cls, **kwargs) -> IsaacEnvironment: ...


# Catalog backend_kwargs.task_name selects one of these implementations.
# Multiple catalog entries may use the same implementation with different task_spec.
TASKS = {
    "roboballet-eight-arm":
        "roboenv_runtime.adapters.isaaclab.tasks.roboballet_eight_arm.task",
    "yam-calibrate-overhead-camera":
        "roboenv_runtime.adapters.isaaclab.tasks.yam_calibrate_overhead_camera.task",
    "yam-calibrate-wrist-cameras":
        "roboenv_runtime.adapters.isaaclab.tasks.yam_calibrate_wrist_cameras.task",
    "yam-calibrate-dual-arm":
        "roboenv_runtime.adapters.isaaclab.tasks.yam_calibrate_dual_arm.task",

    "go2-d1-control": "roboenv_runtime.adapters.isaaclab.tasks.go2_d1_control.task",
    "go2-d1-reach": "roboenv_runtime.adapters.isaaclab.tasks.go2_d1_reach.task",
    "go2-d1-pick-place": "roboenv_runtime.adapters.isaaclab.tasks.go2_d1_pick_place.task",
}


def get_task(name: str) -> type[TaskFactory]:
    if not isinstance(name, str) or name not in TASKS:
        raise ValueError(f"Unknown Isaac Lab task {name!r}; registered tasks: {sorted(TASKS)}")
    task = import_module(TASKS[name]).Task
    if task.name != name or not isinstance(task.embodiment, str) or not task.embodiment:
        raise ValueError(f"Invalid Isaac Lab task registration: {name}")
    return task
