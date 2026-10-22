"""Adapt pinned upstream hosts without exposing owner identities or clock controls."""

import base64
import importlib
import inspect
import json
import sys
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from roboenv.devices.model import DeviceError
from roboenv.provisioning.chemlab_assets import resource


class Instrument:
    def __init__(self, root, row, instance):
        self.root, self.row, self.instance = root, row, instance
        self.contract = json.loads(resource(root, row["contract"]).read_text())
        registry = json.loads(files("roboenv").joinpath("assets/chemlab-runtimes.json").read_text())
        self.config = registry[row["id"]]
        # Only the release's pinned module/class table selects executable code.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        scripts = str(root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        module = importlib.import_module(self.config["module"])
        resource(root, str(module.__file__.removeprefix(str(root) + "/")))
        cls = getattr(module, self.config["cls"])
        self.time, self.sequence = 0.0, 0
        params = inspect.signature(cls).parameters
        kwargs = dict(self.config.get("ctor") or {})
        if "clock" in params:
            kwargs["clock"] = lambda: self.time
        if "scene_adapter" in params:
            kwargs["scene_adapter"] = lambda instance, updates: None
        if self.config.get("instance_kw"):
            kwargs[self.config["instance_kw"]] = instance
        self.host = (
            cls(instance, **kwargs) if self.config.get("instance_in_ctor") else cls(**kwargs)
        )
        self.capture_directory = TemporaryDirectory(prefix="chemlab-panel-")
        if hasattr(self.host, "panel_capture_dir"):
            self.host.panel_capture_dir = self.capture_directory.name
        create = self.config.get("create")
        if create == "type":
            self.host.create_instance(instance, row["id"])
        elif create == "plain":
            self.host.create_instance(instance)
        elif create == "scene":
            self.host.create_instance(instance, lambda instance, updates: None)
        self.agent = self._session("roboenv_agent", "agent")
        self.owner = self._session("roboenv_owner", "evaluator")
        self.capabilities = {
            key: value
            for key, value in self.contract["capabilities"].items()
            if value.get("visibility") == "agent"
            and value.get("kind") != "simulation"
            and not key.startswith(("sim.", "host.", "truth.", "plant.", "fault."))
        }

    def _public_result(self, value):
        if isinstance(value, list):
            return [self._public_result(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "path" in value and value.get("format") == "png":
            path = Path(value["path"]).resolve()
            if not path.is_relative_to(self.capture_directory.name) or not path.is_file():
                raise DeviceError("invalid_artifact", "Instrument returned an unowned panel image")
            return {
                key: self._public_result(item) for key, item in value.items() if key != "path"
            } | {
                "mime_type": "image/png",
                "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        return {key: self._public_result(item) for key, item in value.items()}

    def _session(self, name, visibility):
        if self.config.get("host_is_session"):
            return self.host
        if visibility == "agent" and self.config.get("agent_session_default"):
            return self.host.session()
        if visibility == "evaluator" and self.config.get("evaluator_register_positional"):
            self.host.register_principal(name)
        else:
            self.host.register_principal(name, visibility=visibility)
        return self.host.session(name)

    def invoke(self, capability, arguments, *, owner=False):
        if not owner and capability not in self.capabilities:
            raise DeviceError("permission_denied", "Capability is not exposed to agents")
        self.sequence += 1
        session = self.owner if owner else self.agent
        result = session.invoke(
            instance_id=self.instance,
            capability_id=capability,
            arguments=arguments,
            request_id=f"roboenv-{self.sequence}",
        )
        return result if owner else self._public_result(result)

    def operation(self, action, operation_id):
        method = getattr(self.agent, action, None)
        if method is None:
            raise DeviceError("unsupported_action", "Instrument has no asynchronous operation API")
        args = {"operation_id": operation_id}
        parameters = inspect.signature(method).parameters
        if "instance_id" in parameters:
            args["instance_id"] = self.instance
        if "request_id" in parameters:
            self.sequence += 1
            args["request_id"] = f"roboenv-{self.sequence}"
        return self._public_result(method(**args))

    def read(self, *, owner=False):
        session = self.owner if owner else self.agent
        allowed = [
            key
            for key, value in self.contract["state_slots"].items()
            if owner or value.get("visibility") == "agent"
        ]
        result = session.read(instance_id=self.instance, state_ids=None)
        # Keep only explicitly permitted slots; do not copy logs or raw host metadata.
        return {
            key: value for key, value in result.get("observations", {}).items() if key in allowed
        }

    def advance(self, dt):
        mode = self.config.get("advance")
        end = self.time + dt
        if mode == "step_dt":
            self.host.step(dt)
        elif mode == "host_step_dt":
            self.host.host_step(dt)
        elif mode == "host_step_dt_world":
            self.host.host_step(dt, world_time_s=end)
        elif mode == "step_dt_world":
            self.host.step(dt, world_time_s=end)
        elif mode == "host_step_abs":
            self.host.host_step(end)
        elif mode == "host_advance_abs":
            self.host.host_advance(end)
        elif mode == "host_set_clock_abs":
            self.host.host_set_clock(end)
        elif mode == "vici_set_time":
            self.host.host_set_time(world_time_s=end, request_id=f"clock-{end:.9f}")
        elif mode is not None:
            raise ValueError(f"Unknown pinned clock mode: {mode}")
        self.time = end

    def close(self):
        self.capture_directory.cleanup()
