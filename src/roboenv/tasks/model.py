"""Environment task definitions deliberately exclude agent configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from roboenv.validation import json_value, yaml_load


@dataclass(frozen=True)
class Task:
    id: str
    backend: str
    backend_task: str
    robot: str
    instruction: str
    backend_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    budget_mode: str = "official"
    action_abstraction: str = "cartesian-primitives"
    device_bindings: dict | None = None

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            value = yaml_load(stream)
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value):
        """Validate the canonical task, including legacy flat task files."""
        if isinstance(value, dict) and "simulation" in value:
            value = dict(value)
            simulation = value.pop("simulation")
            allowed = {"backend", "robot", "backend_kwargs", "device_bindings"}
            if not isinstance(simulation, dict) or set(simulation) - allowed:
                raise ValueError(
                    "simulation must contain backend, robot and optional backend config"
                )
            if allowed & set(value):
                raise ValueError("do not mix simulation configuration with flat task fields")
            value.update(simulation)
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
        if not isinstance(task.backend_kwargs, dict) or any(
            not isinstance(key, str) for key in task.backend_kwargs
        ):
            raise ValueError("backend_kwargs must be an environment configuration mapping")
        json_value(task.backend_kwargs)
        if task.device_bindings is not None:
            from roboenv.devices.bindings.spec import BindingSpec

            if not isinstance(task.device_bindings, dict) or not task.device_bindings:
                raise ValueError("device_bindings must be a nonempty device mapping")
            for name, binding in task.device_bindings.items():
                BindingSpec.parse(name, binding)
        if type(task.seed) is not int or task.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if task.action_abstraction not in {"cartesian-primitives", "symbolic-primitives"}:
            raise ValueError("unknown control abstraction")
        if task.budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be official or explicit practice")
        for name in ("id", "backend", "backend_task", "robot"):
            item = getattr(task, name)
            if (
                not isinstance(item, str)
                or not item
                or not item[0].isalnum()
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for c in item
                )
            ):
                raise ValueError(f"invalid task {name}")
        if not isinstance(task.instruction, str) or not task.instruction.strip():
            raise ValueError("instruction must be nonempty text")
        return task
