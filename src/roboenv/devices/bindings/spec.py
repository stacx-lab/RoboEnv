"""Declarative owner-only assembly: device facts + binding + existing entity."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from roboenv.devices.model import DeviceDefinition
from roboenv.validation import json_value, yaml_load

BINDING_CONTRACT = "roboenv/bindings/1"


def binding_resource(root, *, backend):
    """Return an owner-bundle plan or a packaged default, never catalog internals.

    The resource supports read_text/read_bytes/is_file, including in a wheel;
    callers must not assume it is a filesystem path.
    """
    root = Path(root)
    local = root / "bindings.yaml"
    if local.exists():
        return local
    return (
        files("roboenv.devices.bindings")
        .joinpath("defaults")
        .joinpath(_name(backend, "backend"))
        .joinpath(f"{_name(root.name, 'robot')}.yaml")
    )


def _name(value, label):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in value
        )
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def profile_path(root, relative):
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("device profile must be a relative path within the robot definition")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("device profile escapes the robot definition")
    return path


@dataclass(frozen=True)
class BindingSpec:
    name: str
    binding: str
    entity: str
    profile: str
    config: dict
    overrides: dict

    @classmethod
    def parse(cls, name, value):
        _name(name, "logical device")
        allowed = {"binding", "entity", "profile", "config", "overrides"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError(f"unknown binding fields for {name}")
        binding, entity = value.get("binding"), value.get("entity")
        if not isinstance(binding, str) or not binding:
            raise ValueError("binding must be a registered identifier")
        if not isinstance(entity, str) or not entity or len(entity) > 256:
            raise ValueError("entity must be a nonempty backend selector")
        config, overrides = value.get("config", {}), value.get("overrides", {})
        if not isinstance(config, dict) or not isinstance(overrides, dict):
            raise ValueError("binding config and overrides must be objects")
        if set(overrides) - {"intrinsics", "extrinsics"}:
            raise ValueError("overrides may only change device parameters, mounts and control mode")
        json_value([config, overrides])
        return cls(
            name,
            binding,
            entity,
            value.get("profile", f"devices/{name}/profile.yaml"),
            copy.deepcopy(config),
            copy.deepcopy(overrides),
        )

    def resolved_profile(self, root):
        value = yaml_load(profile_path(root, self.profile).read_text())
        if not isinstance(value, dict):
            raise ValueError("device profile must be an object")
        # Logical names and mounting/connection configuration are assembly facts,
        # not mutations of a shared hardware definition.
        value = copy.deepcopy(value)
        value["name"] = self.name
        for section in ("intrinsics", "extrinsics"):
            if section in self.overrides:
                if not isinstance(self.overrides[section], dict):
                    raise ValueError(f"{section} override must be an object")
                value[section] = _merge(value.get(section, {}), self.overrides[section])
        return value

    def definition(self, root):
        profile = self.resolved_profile(root)
        return DeviceDefinition(
            self.name,
            profile.get("type", "other"),
            profile.get("intrinsics", {}),
            profile.get("extrinsics", {}),
        )

    def as_dict(self):
        return {
            "binding": self.binding,
            "entity": self.entity,
            "profile": self.profile,
            **({"config": copy.deepcopy(self.config)} if self.config else {}),
            **({"overrides": copy.deepcopy(self.overrides)} if self.overrides else {}),
        }


@dataclass(frozen=True)
class BindingPlan:
    backend: str
    devices: tuple[BindingSpec, ...]
    embodiment: str | None = None

    @classmethod
    def load(cls, root, *, backend, devices=None):
        root = Path(root)
        resource = binding_resource(root, backend=backend) if devices is None else None
        if devices is not None or resource.is_file():
            value = (
                {"schema_version": BINDING_CONTRACT, "backend": backend, "devices": devices}
                if devices is not None
                else yaml_load(resource.read_text(encoding="utf-8"))
            )
            if (
                not isinstance(value, dict)
                or set(value) - {"schema_version", "backend", "embodiment", "devices"}
                or value.get("schema_version") != BINDING_CONTRACT
            ):
                raise ValueError("unsupported or malformed binding plan")
            if value.get("backend") != backend:
                raise ValueError(f"binding backend mismatch: expected {backend!r}")
            entries = value.get("devices")
            if not isinstance(entries, dict) or not entries:
                raise ValueError("binding plan needs a nonempty device mapping")
            embodiment = value.get("embodiment")
            if embodiment is not None:
                _name(embodiment, "embodiment")
            plan = cls(
                backend,
                tuple(BindingSpec.parse(name, entry) for name, entry in sorted(entries.items())),
                embodiment,
            )
        else:
            raise ValueError(
                f"No default bindings for {backend}/{root.name}; "
                "robot requires an explicit bindings.yaml"
            )
        for entry in plan.devices:
            entry.definition(root)  # validate parameters/paths before a simulator is started
        return plan

    def as_dict(self):
        return {
            "schema_version": BINDING_CONTRACT,
            "backend": self.backend,
            **({"embodiment": self.embodiment} if self.embodiment else {}),
            "devices": {entry.name: entry.as_dict() for entry in self.devices},
        }
