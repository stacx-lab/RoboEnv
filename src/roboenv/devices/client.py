"""Small, typed device surface over the private wire client."""

import http.client
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class TaskInfo:
    id: str
    instruction: str
    context: object


class Environment:
    """An attached task. Closing this handle never shuts down its simulator."""

    def __init__(self, transport):
        self._transport = transport
        description = transport.describe()
        task = description.get("task", {"id": description.get("task_id", ""), "instruction": ""})
        self.task = TaskInfo(task["id"], task["instruction"], _freeze(task.get("context", {})))
        self._profiles = {p["name"]: p for p in transport.devices()["devices"]}
        self._closed = False
        self._artifacts = None

    def devices(self):
        return deepcopy(list(self._profiles.values()))

    def status(self):
        return self._transport.request("simulation.status")

    def stop(self):
        """Finalize and stop the assigned simulation, when its owner permits it."""
        return self._transport.request("simulation.stop")

    def device(self, name):
        if name not in self._profiles:
            raise KeyError(f"Unknown device: {name}")
        return Device(self, self._profiles[name])

    def _download(self, file):
        if not self._transport.endpoint:
            return file
        endpoint = urlsplit(self._transport.endpoint)
        connection = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )(endpoint.hostname, endpoint.port, timeout=30)
        try:
            name = Path(file).name
            connection.request(
                "GET", f"/artifacts/{self._transport.session}/{self._transport.generation}/{name}"
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Image download failed: HTTP {response.status}")
            if self._artifacts is None:
                self._artifacts = Path(tempfile.mkdtemp(prefix="roboenv-images-"))
            path = self._artifacts / name
            with path.open("wb") as stream:
                while chunk := response.read(65536):
                    stream.write(chunk)
            return str(path)
        finally:
            connection.close()

    def close(self):
        self._closed = True
        if self._artifacts is not None:
            shutil.rmtree(self._artifacts, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Device:
    """Only operations declared by this device's interface are exposed."""

    def __init__(self, environment, profile):
        self._environment, self._profile = environment, profile

    @property
    def profile(self):
        return deepcopy(self._profile)

    def __getattr__(self, action):
        spec = self._profile["actions"].get(action)
        if action.startswith("_") or not spec or not spec["supported"]:
            raise AttributeError(f"{self._profile['name']} does not provide {action}")

        def invoke(*values, **args):
            parameter = {
                "write": "actions"
                if self._profile["type"] in {"joint", "dexterous_hand"}
                else "target",
                "wait": "duration_s",
                "fk": "joints",
                "ik": "pose",
            }.get(action)
            if values:
                if len(values) != 1 or parameter is None or parameter in args:
                    raise TypeError(f"Invalid arguments to {action}")
                args[parameter] = values[0]
            env = self._environment
            if env._closed:
                raise RuntimeError("Environment client is closed")
            reply = env._transport.device(self._profile["name"]).call(action, args)
            data = reply["data"]
            if "simulation" in reply and isinstance(data, dict):
                data = {**data, "simulation": reply["simulation"]}
            if (
                self._profile["type"] in {"rgb", "rgbd"}
                and isinstance(data, dict)
                and data.get("file")
            ):
                data = {**data, "file": env._download(data["file"])}
                if self._profile["type"] == "rgbd" and data.get("depth_file"):
                    data["depth_file"] = env._download(data["depth_file"])
            return data

        return invoke
