"""Environment task definitions deliberately exclude agent configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from roboenv.validation import json_value, yaml_load

DEVICE_BACKENDS = frozenset(
    {
        "libero",
        "robocasa",
        "isaaclab",
        "robotwin",
        "robodojo",
        "behavior",
        "simple",
    }
)


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
    execution_mode: str = "realtime"

    @classmethod
    def load(cls, path):
        with Path(path).open() as stream:
            value = yaml_load(stream)
        task = cls.from_dict(value)
        if "execution_mode" not in value:
            raise ValueError(
                "task catalog entries must declare execution_mode: stepped or realtime"
            )
        return task

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
        if "max_steps" in task.backend_kwargs:
            maximum = task.backend_kwargs["max_steps"]
            if maximum is not None and (type(maximum) is not int or maximum <= 0):
                raise ValueError("max_steps must be a positive integer or null")
        if "stop_on_success" in task.backend_kwargs:
            if type(task.backend_kwargs["stop_on_success"]) is not bool:
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
        if task.device_bindings is not None:
            from roboenv.devices.bindings.spec import BindingSpec

            if not isinstance(task.device_bindings, dict) or not task.device_bindings:
                raise ValueError("device_bindings must be a nonempty device mapping")
            for name, binding in task.device_bindings.items():
                BindingSpec.parse(name, binding)
        if type(task.seed) is not int or task.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if task.action_abstraction not in {
            "devices",
            "cartesian-primitives",
            "symbolic-primitives",
        }:
            raise ValueError("unknown control abstraction")
        if task.execution_mode not in ("stepped", "realtime"):
            raise ValueError("execution_mode must be stepped or realtime")
        if task.budget_mode not in {"official", "practice"}:
            raise ValueError("budget_mode must be official or explicit practice")
        for name in ("id", "backend", "backend_task", "robot"):
            item = getattr(task, name)
            if name == "id" and isinstance(item, str) and "/" in item:
                parts = item.split("/")
                if len(parts) != 2 or parts[0] != task.backend:
                    raise ValueError("task id must use simulator/task_name with its backend")
                item = parts[1]
            if not isinstance(item, str):
                raise ValueError(f"invalid task {name}")
            # Robot references are relative catalog paths, including backend groups.
            parts = item.split("/") if name == "robot" else [item]
            if any(
                not part
                or not part[0].isalnum()
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for c in part
                )
                for part in parts
            ):
                raise ValueError(f"invalid task {name}")
        if not isinstance(task.instruction, str) or not task.instruction.strip():
            raise ValueError("instruction must be nonempty text")
        return task
