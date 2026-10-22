"""One clock for independently addressed instruments and a process-driven Isaac scene."""

import copy
import hashlib
import json
import math
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from roboenv.devices.model import DeviceError
from roboenv.provisioning.chemlab_assets import (
    REVISION,
    asset_root,
    catalog,
    resource,
    verify_assets,
)
from roboenv.validation import json_value

from roboenv_runtime.core.types import Budget, ScoreReport

from ...common.recording import source_identity
from ...common.spectator import SpectatorSource
from .instrument import Instrument


def value_at(observations, slot, path=()):
    value = observations[slot]["value"]
    for key in path:
        value = value[key]
    return value


def matches(observations, condition):
    try:
        if observations[condition["state"]].get("quality") in {"unavailable", "invalid", "error"}:
            return False
        value = value_at(observations, condition["state"], condition.get("path", []))
        if "equals" in condition:
            return type(value) is type(condition["equals"]) and value == condition["equals"]
        return (
            type(value) in (float, int)
            and math.isfinite(value)
            and condition.get("min", -math.inf) <= value <= condition.get("max", math.inf)
        )
    except (KeyError, TypeError, IndexError):
        return False


class ChemLabEnvironment:
    control_interval = 0.1
    stop_on_success = False

    def __init__(
        self,
        *,
        task_spec,
        asset_directory=None,
        device_profiles=None,
        max_steps=3000,
        execution_mode="stepped",
        render=True,
    ):
        json_value(task_spec)
        if not isinstance(task_spec, dict) or set(task_spec) - {"instruments", "goals", "dwell_s"}:
            raise ValueError("Invalid ChemLab task specification")
        if not task_spec.get("instruments") or not isinstance(task_spec["instruments"], dict):
            raise ValueError("At least one named instrument is required")
        if type(max_steps) is not int or max_steps < 1 or type(render) is not bool:
            raise ValueError("Invalid frame budget or rendering mode")
        self.implementation = source_identity("chemlab-instruments", Path(__file__).parent)
        self.spec = copy.deepcopy(task_spec)
        self.profiles = copy.deepcopy(device_profiles or {})
        self.depot = asset_root(asset_directory)
        self.asset_identity = verify_assets(self.depot)
        self.implementation["chemlab-runtimes.json"] = hashlib.sha256(
            files("roboenv").joinpath("assets/chemlab-runtimes.json").read_bytes()
        ).hexdigest()
        from roboenv.provisioning import chemlab_assets

        from roboenv_runtime.bindings.backends.isaaclab import chemlab as binding
        from roboenv_runtime.paths import repo_root

        for name, path in {
            "runtime/locks/chemlab.txt": repo_root() / "runtime/locks/chemlab.txt",
            "binding.py": Path(binding.__file__),
            "assets.py": Path(chemlab_assets.__file__),
        }.items():
            self.implementation[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.dependencies = {
            name: version(name) for name in ("numpy", "scipy", "Pillow", "CoolProp")
        }
        self.rows = catalog(self.depot)
        for name, spec in self.spec["instruments"].items():
            if (
                not name.isidentifier()
                or name == "world"
                or not isinstance(spec, dict)
                or set(spec) - {"asset", "initialize"}
            ):
                raise ValueError("Invalid instrument configuration")
            if not isinstance(spec.get("asset"), str) or spec["asset"] not in self.rows:
                raise ValueError("Unknown instrument asset")
            if not isinstance(spec.get("initialize", []), list):
                raise ValueError("Initialization must be a command list")
            for command in spec.get("initialize", []):
                if (
                    not isinstance(command, dict)
                    or set(command) != {"capability", "arguments"}
                    or not isinstance(command["capability"], str)
                    or not isinstance(command["arguments"], dict)
                ):
                    raise ValueError("Invalid initialization command")
        self.goals = self.spec.get("goals", [])
        if not isinstance(self.goals, list):
            raise ValueError("Goals must be a list")
        for goal in self.goals:
            if (
                not isinstance(goal, dict)
                or set(goal) - {"instrument", "state", "path", "equals", "min", "max"}
                or goal.get("instrument") not in self.spec["instruments"]
                or not isinstance(goal.get("state"), str)
                or not set(goal) & {"equals", "min", "max"}
            ):
                raise ValueError("Invalid owner goal")
            if (
                "equals" in goal
                and set(goal) & {"min", "max"}
                or not isinstance(goal.get("path", []), list)
                or any(type(item) not in (str, int) for item in goal.get("path", []))
                or any(type(goal[key]) not in (int, float) for key in ("min", "max") if key in goal)
                or goal.get("min", -math.inf) > goal.get("max", math.inf)
            ):
                raise ValueError("Invalid goal comparison")
            row = self.rows[self.spec["instruments"][goal["instrument"]]["asset"]]
            contract = json.loads(resource(self.depot, row["contract"]).read_text())
            if goal["state"] not in contract["state_slots"]:
                raise ValueError("Goal references an unknown state slot")
        self.dwell = self.spec.get("dwell_s", 1.0)
        if type(self.dwell) not in (int, float) or not math.isfinite(self.dwell) or self.dwell <= 0:
            raise ValueError("Goal dwell must be finite and positive")
        self.render_enabled = render
        self.native = None
        self.instruments = {}
        self.configured_max = max_steps
        self.configure_execution_mode(execution_mode)

    def configure_execution_mode(self, mode):
        if mode not in {"stepped", "realtime"}:
            raise ValueError("Unknown execution mode")
        self.execution_mode = mode
        self.max_steps = self.configured_max if mode == "stepped" else None

    def reset(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed, self.frame, self.held, self.completed = seed, 0, 0.0, False
        self.commands = []
        for instrument in self.instruments.values():
            instrument.close()
        self.instruments = {
            name: Instrument(self.depot, self.rows[spec["asset"]], name)
            for name, spec in self.spec["instruments"].items()
        }
        for name, spec in self.spec["instruments"].items():
            inst = self.instruments[name]
            for command in spec.get("initialize", []):
                args = copy.deepcopy(command.get("arguments", {}))
                if "noise_seed" in args:
                    args["noise_seed"] = seed
                response = inst.invoke(command["capability"], args, owner=True)
                if response.get("status") not in {"succeeded", "accepted"}:
                    raise ValueError(f"Instrument initialization failed: {name}: {response}")
        self.initial = self.owner_state()
        if self.render_enabled:
            if self.native is None:
                from .scene import InstrumentScene

                self.native = InstrumentScene(self)
            self.native.sync()

    def tick(self):
        if self.budget().exhausted or self.completed and self.stop_on_success:
            return
        for inst in self.instruments.values():
            inst.advance(self.control_interval)
        self.frame += 1
        state = self.owner_state()
        reached = bool(self.goals) and all(matches(state[g["instrument"]], g) for g in self.goals)
        self.held = self.held + self.control_interval if reached else 0.0
        self.completed |= self.held + 1e-9 >= self.dwell
        if self.native is not None:
            self.native.sync()

    def read_device(self, name):
        return {
            "observations": self.instruments[name].read(),
            "timestamp": self.frame * self.control_interval,
        }

    def invoke(self, name, capability, arguments):
        if self.completed and self.stop_on_success or self.budget().exhausted:
            raise DeviceError("episode_finished", "Instrument episode has ended")
        if capability in {"operation.poll", "operation.cancel"}:
            response = self.instruments[name].operation(
                capability.split(".")[1], arguments["operation_id"]
            )
        else:
            response = self.instruments[name].invoke(capability, arguments)
        self.commands.append(
            {
                "frame": self.frame,
                "instrument": name,
                "capability": capability,
                "arguments": copy.deepcopy(arguments),
                "status": response.get("status"),
            }
        )
        if self.native is not None:
            self.native.sync()
        return response

    def owner_state(self):
        return {name: inst.read(owner=True) for name, inst in self.instruments.items()}

    def score(self):
        return ScoreReport(
            reward=float(self.completed),
            raw_score=float(self.completed),
            completed=self.completed,
            detail={"dwell_s": self.held},
        )

    def budget(self):
        return Budget(self.frame, self.max_steps, "action_frames")

    def record(self):
        return {
            "schema": "roboenv/chemlab-record/1",
            "revision": REVISION,
            "seed": self.seed,
            "task_definition": self.spec,
            "state": self.owner_state(),
            "initial_state": self.initial,
            "commands": copy.deepcopy(self.commands),
            "budget": self.budget().as_dict(),
            "native_success": self.completed,
            "rendered": self.native is not None,
            "sources": self.implementation,
            "dependencies": self.dependencies,
            "device_profiles": self.profiles,
            "assets": self.asset_identity,
            "execution_mode": self.execution_mode,
            "max_steps": self.configured_max,
            "stop_on_success": self.stop_on_success,
            "scene_mappings": self.native.mapping_report if self.native else {},
        }

    def spectator_sources(self):
        if self.native is None:
            return {}
        return {"third_person": SpectatorSource(self.native.spectator_pose)}

    def close(self):
        for instrument in self.instruments.values():
            instrument.close()
        if self.native is not None:
            self.native.close()
            self.native = None
        self.instruments.clear()
