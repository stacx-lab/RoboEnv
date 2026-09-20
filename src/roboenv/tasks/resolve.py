"""Resolve user task references once for local runs and Harbor exports."""

from dataclasses import dataclass, replace
from pathlib import Path

from roboenv.devices.bindings.spec import BindingPlan
from roboenv.project import project_root
from roboenv.tasks.model import Task


@dataclass(frozen=True)
class ResolvedTask:
    task: Task
    path: Path
    catalog: Path
    robot: Path
    bindings: BindingPlan

    def runtime_config(self):
        return {
            "id": self.task.backend_task,
            "adapter": self.task.backend,
            "adapter_kwargs": self.task.backend_kwargs,
            "text": self.task.instruction,
            "seeds": [self.task.seed],
            "budget": {"mode": self.task.budget_mode},
        }


def resolve_task(reference, *, project=None, catalog=None, realtime=None,
                 arm_gravity_compensation=None):
    catalog = Path(catalog).resolve() if catalog else project_root(project) / "catalog"
    path = Path(reference)
    if not path.is_file():
        matches = [
            candidate
            for candidate in sorted((catalog / "tasks").rglob("task.yaml"))
            if Task.load(candidate).id == str(reference)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one task matching {reference!r}, found {len(matches)}; "
                "use 'roboenv tasks list' or pass a task.yaml path and --catalog"
            )
        path = matches[0]
    task = Task.load(path)
    if arm_gravity_compensation is not None:
        if type(arm_gravity_compensation) is not bool:
            raise ValueError("arm_gravity_compensation must be a boolean")
        if task.backend != "simple":
            raise ValueError("Arm gravity compensation selection is supported only for SIMPLE")
        task = replace(task, backend_kwargs={
            **task.backend_kwargs, "arm_gravity_compensation": arm_gravity_compensation,
        })
    if task.backend == "simple":
        task = replace(task, backend_kwargs={
            "arm_gravity_compensation": True, **task.backend_kwargs,
        })
    if realtime is not None:
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        selectable = task.backend in {
            "robocasa",
            "libero",
            "robotwin",
            "robodojo",
            "behavior",
            "isaaclab",
            "simple",
        }
        if not selectable:
            raise ValueError(f"Clock selection is not supported for {task.backend}")
        task = replace(task, backend_kwargs={**task.backend_kwargs, "realtime": realtime})
    robot = catalog / "robots" / task.robot
    if robot.is_symlink() or not robot.resolve().is_relative_to(catalog):
        raise ValueError("robot definition must remain inside the source catalog")
    if not (robot / "workspace.yaml").is_file():
        raise ValueError(f"Robot {task.robot!r} is missing from the selected catalog")
    bindings = BindingPlan.load(robot, backend=task.backend, devices=task.device_bindings)
    if task.backend == "libero" and not task.backend_kwargs.get("realtime", True):
        devices = []
        for spec in bindings.devices:
            if spec.entity == "arm":
                intrinsics = spec.resolved_profile(robot)["intrinsics"]
                # LIBERO fixes the native controller frequency at 20 Hz.
                intrinsics["control"]["execution"] = {"mode": "stepped", "interval_s": 0.05}
                spec = replace(spec, overrides={**spec.overrides, "intrinsics": intrinsics})
            devices.append(spec)
        bindings = replace(bindings, devices=tuple(devices))
    if task.backend in {"behavior", "isaaclab"} and task.backend_kwargs.get("realtime", False):
        # Execution mode is an episode setting, while geometry stays in the catalog.
        devices = []
        for spec in bindings.devices:
            intrinsics = spec.resolved_profile(robot)["intrinsics"]
            if isinstance(intrinsics.get("control"), dict) and "execution" in intrinsics["control"]:
                intrinsics["control"]["execution"] = {
                    "mode": "realtime",
                    "command_timeout_s": (
                        task.backend_kwargs.get("command_timeout_s", 0.5)
                        if task.backend == "isaaclab"
                        else None
                    ),
                }
                spec = replace(spec, overrides={**spec.overrides, "intrinsics": intrinsics})
            devices.append(spec)
        bindings = replace(bindings, devices=tuple(devices))
    if task.backend == "simple":
        devices = []
        for spec in bindings.devices:
            profile = spec.resolved_profile(robot)
            intrinsics = profile["intrinsics"]
            if "control" in intrinsics:
                intrinsics["control"]["execution"] = (
                    {"mode": "realtime", "command_timeout_s":
                     task.backend_kwargs.get("command_timeout_s", 0.5)
                     if profile["type"] == "velocity" else None}
                    if task.backend_kwargs.get("realtime", True)
                    else {"mode": "stepped", "interval_s": 0.02}
                )
                spec = replace(spec, overrides={**spec.overrides, "intrinsics": intrinsics})
            devices.append(spec)
        bindings = replace(bindings, devices=tuple(devices))
    return ResolvedTask(task, path.resolve(), catalog, robot, bindings)
