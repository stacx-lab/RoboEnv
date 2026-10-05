"""Environment task definitions deliberately exclude agent configuration."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from roboenv.validation import json_value, yaml_load


@dataclass(frozen=True)
class Task:
    id: str
    robot: str
    instruction: str
    backend_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    budget_mode: str = "official"
    execution_mode: str = "realtime"
    stop_on_success: bool = False

    @property
    def backend(self):
        return self.id.split("/", 1)[0]

    def as_dict(self):
        """Canonical portable task definition, with configurable defaults exposed."""
        return {
            "id": self.id,
            "robot": self.robot,
            "instruction": self.instruction,
            "seed": self.seed,
            "budget_mode": self.budget_mode,
            "execution_mode": self.execution_mode,
            "stop_on_success": self.stop_on_success,
            "backend_kwargs": deepcopy(self.backend_kwargs),
        }

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            value = yaml_load(stream)
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value):
        """Validate the single environment task format."""
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError(
                "task must contain only environment task fields; "
                "agent settings belong in run configuration"
            )
        try:
            task = cls(**value)
        except TypeError as exc:
            raise ValueError("task is missing required environment fields") from exc
        if not isinstance(task.id, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*", task.id
        ):
            raise ValueError("task id must use simulator/lowercase-kebab-case")
        if not isinstance(task.backend_kwargs, dict) or any(
            not isinstance(key, str) for key in task.backend_kwargs
        ):
            raise ValueError("backend_kwargs must be an environment configuration mapping")
        json_value(task.backend_kwargs)
        if "max_steps" in task.backend_kwargs:
            maximum = task.backend_kwargs["max_steps"]
            if maximum is not None and (type(maximum) is not int or maximum <= 0):
                raise ValueError("max_steps must be a positive integer or null")
        if type(task.stop_on_success) is not bool:
            raise ValueError("stop_on_success must be a boolean")
        if "realtime" in task.backend_kwargs:
            raise ValueError(
                "backend_kwargs.realtime is not supported; use execution_mode: stepped or realtime"
            )
        if "arm_gravity_compensation" in task.backend_kwargs:
            if type(task.backend_kwargs["arm_gravity_compensation"]) is not bool:
                raise ValueError("arm_gravity_compensation must be a boolean")
            if task.backend != "simple":
                raise ValueError("Arm gravity compensation selection is supported only for SIMPLE")
        if type(task.seed) is not int or task.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if task.execution_mode not in ("stepped", "realtime"):
            raise ValueError("execution_mode must be stepped or realtime")
        if task.budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be official or explicit practice")
        if not isinstance(task.robot, str) or any(
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", part)
            for part in task.robot.split("/")
        ):
            raise ValueError("invalid task robot")
        if not isinstance(task.instruction, str) or not task.instruction.strip():
            raise ValueError("instruction must be nonempty text")
        return task


def task_path(reference, root):
    """Load a task file or resolve a public ID below a task directory."""
    path = Path(reference)
    if path.is_file():
        return path
    matches = [
        path for path in sorted(Path(root).rglob("*.yaml")) if Task.load(path).id == str(reference)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one task matching {reference!r}, found {len(matches)}; "
            "use roboenv tasks list or pass a task.yaml path"
        )
    return matches[0]
