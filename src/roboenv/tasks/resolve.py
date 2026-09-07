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


def resolve_task(reference, *, project=None, catalog=None, realtime=None):
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
    if realtime is not None:
        if type(realtime) is not bool:
            raise ValueError("realtime must be a boolean")
        if task.backend != "robocasa":
            raise ValueError(f"Clock selection is not supported for {task.backend}")
        task = replace(task, backend_kwargs={**task.backend_kwargs, "realtime": realtime})
    robot = catalog / "robots" / task.robot
    if robot.is_symlink() or not robot.resolve().is_relative_to(catalog):
        raise ValueError("robot definition must remain inside the source catalog")
    if not (robot / "workspace.yaml").is_file():
        raise ValueError(f"Robot {task.robot!r} is missing from the selected catalog")
    bindings = BindingPlan.load(robot, backend=task.backend, devices=task.device_bindings)
    return ResolvedTask(task, path.resolve(), catalog, robot, bindings)
