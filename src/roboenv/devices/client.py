"""Small, typed device surface over the private wire client."""

import base64
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
        task = description["task"]
        self.task = TaskInfo(task["id"], task["instruction"], _freeze(task.get("context", {})))
        self._profiles = {p["name"]: p for p in transport.devices()["devices"]}
        self._closed = False
        self._artifacts = None

    def refresh_task(self):
        """Read the current public instruction, including native task transitions."""
        task = self._transport.describe()["task"]
        self.task = TaskInfo(task["id"], task["instruction"], _freeze(task.get("context", {})))
        return self.task

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

    def _download(self, file, *, content=False):
        if not self._transport.endpoint:
            return Path(file).read_bytes() if content else file
        endpoint = urlsplit(self._transport.endpoint)
        connection = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )(endpoint.hostname, endpoint.port, timeout=30)
        path = None
        try:
            name = Path(file).name
            connection.request(
                "GET", f"/artifacts/{self._transport.session}/{self._transport.generation}/{name}"
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"Image download failed: HTTP {response.status}")
            expected = response.length
            if content:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError("Camera content is too large; use output='file'")
                if expected is not None and len(raw) != expected:
                    raise ValueError("Incomplete image download")
                return raw
            if self._artifacts is None:
                self._artifacts = Path(tempfile.mkdtemp(prefix="roboenv-images-"))
            path = self._artifacts / name
            size = 0
            with path.open("wb") as stream:
                while chunk := response.read(65536):
                    size += stream.write(chunk)
            if expected is not None and size != expected:
                raise ValueError("Incomplete image download")
            return str(path)
        except BaseException:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
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
                if self._profile["type"] in {"joint", "dexterous_hand", "gripper", "suction"}
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
            camera = self._profile["type"] in {"rgb", "rgbd"} and action == "read"
            output = "file"
            if camera:
                from roboenv.assets._camera import output_mode

                output = output_mode(args.pop("output", "file"))
            reply = env._transport.device(self._profile["name"]).call(action, args)
            data = reply["data"]
            if "simulation" in reply and isinstance(data, dict):
                data = {**data, "simulation": reply["simulation"]}
            if camera and isinstance(data, dict) and data.get("file"):
                from roboenv.assets._camera import arrays, camera_read

                data = dict(data)
                for key, target in (("file", "color"), ("depth_file", "depth")):
                    if not data.get(key):
                        continue
                    value = env._download(data[key], content=output != "file")
                    if output == "file":
                        data[key] = value
                    else:
                        del data[key]
                        data[target] = base64.b64encode(value).decode("ascii")
                if output != "file":
                    camera_read(data, self._profile)
                if output == "array":
                    data = arrays(data, self._profile)
            return data

        return invoke
