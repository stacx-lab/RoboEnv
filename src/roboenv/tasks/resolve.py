"""Resolve user task references once for environment launch and device preparation."""

from dataclasses import dataclass, replace
from pathlib import Path

from roboenv.devices.bindings.spec import BindingPlan
from roboenv.project import project_root
from roboenv.tasks.model import Task, task_path


@dataclass(frozen=True)
class ResolvedTask:
    task: Task
    path: Path
    catalog: Path
    robot: Path
    bindings: BindingPlan


def resolve_task(reference, *, project=None, catalog=None, arm_gravity_compensation=None):
    catalog = Path(catalog).resolve() if catalog else project_root(project) / "catalog"
    path = task_path(reference, catalog / "tasks")
    task = Task.load(path)
    if arm_gravity_compensation is not None:
        if type(arm_gravity_compensation) is not bool:
            raise ValueError("arm_gravity_compensation must be a boolean")
        if task.backend != "simple":
            raise ValueError("Arm gravity compensation selection is supported only for SIMPLE")
        task = replace(
            task,
            backend_kwargs={
                **task.backend_kwargs,
                "arm_gravity_compensation": arm_gravity_compensation,
            },
        )
    if task.backend == "simple":
        task = replace(
            task,
            backend_kwargs={
                "arm_gravity_compensation": True,
                **task.backend_kwargs,
            },
        )
    robot = catalog / "robots" / task.robot
    if robot.is_symlink() or not robot.resolve().is_relative_to(catalog):
        raise ValueError("robot definition must remain inside the source catalog")
    if not (robot / "workspace.yaml").is_file():
        raise ValueError(f"Robot {task.robot!r} is missing from the selected catalog")
    bindings = BindingPlan.load(robot, backend=task.backend, devices=task.device_bindings)
    if task.backend == "isaaclab" and task.backend_kwargs.get("task_name", "").startswith(
        "go2-d1-"
    ):
        gripper = next(device for device in bindings.devices if device.name == "gripper")
        profile = gripper.resolved_profile(robot)
        mount = profile["extrinsics"]["mount"]
        tcp = profile["intrinsics"]["frames"]["tcp"]
        if tcp["parent_frame"] is not None or tcp["rotation_xyzw"] != [0, 0, 0, 1]:
            raise ValueError("Go2 task TCP requires an aligned local TCP")
        if (mount["parent"], mount["parent_frame"], mount["rotation_xyzw"]) != (
            "arm",
            "flange",
            [0, 0, 0, 1],
        ):
            raise ValueError("Go2 task TCP requires an aligned gripper mount on the arm flange")
        task = replace(
            task,
            backend_kwargs={
                **task.backend_kwargs,
                "tcp_in_flange_m": [
                    a + b for a, b in zip(mount["translation_m"], tcp["translation_m"], strict=True)
                ],
                "device_profiles": {
                    device.name: {
                        key: value
                        for key, value in device.resolved_profile(robot).items()
                        if key in {"type", "intrinsics", "extrinsics"}
                    }
                    for device in bindings.devices
                },
            },
        )
    return ResolvedTask(task, path.resolve(), catalog, robot, bindings)
