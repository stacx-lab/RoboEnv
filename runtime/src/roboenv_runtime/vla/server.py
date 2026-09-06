"""Starting, reusing and stopping policy servers.

A policy server is a subprocess running in the policy's own interpreter (the
vendored openpi uv venv, a conda env for a torch policy ...) on a chosen GPU.
It is started on first use and kept for the life of the process — booting a
2-3B parameter policy takes 1-4 minutes — and one server per (policy, GPU)
serves every episode of that process.  Logs go under ``runs/vla_logs/``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from roboenv_runtime.paths import runs_root
from roboenv_runtime.vla.spec import PolicySpec

__all__ = ["PolicyServer", "current_gpu", "ensure_server", "stop_all"]

log = logging.getLogger(__name__)

_SERVERS: dict[tuple[str, str], PolicyServer] = {}


def current_gpu() -> str:
    """The (physical) GPU id the simulator is on: the first ``CUDA_VISIBLE_DEVICES`` entry."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return raw or "0"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(base: int) -> int:
    for port in range(base, base + 200):
        if _port_free(port):
            return port
    raise RuntimeError(f"no free port in {base}..{base + 200}")


class PolicyServer:
    """One policy-server subprocess and its address."""

    def __init__(
        self, spec: PolicySpec, gpu: str, *, port: int | None = None, log_dir: Path | None = None
    ):
        self.spec = spec
        self.gpu = str(gpu)
        try:
            preferred = spec.port_base + int(self.gpu)
        except ValueError:
            preferred = spec.port_base
        self.port = port or _pick_port(preferred)
        self.log_dir = Path(log_dir) if log_dir else runs_root() / "vla_logs"
        self.log_path = self.log_dir / f"{spec.name}_g{self.gpu}_{self.port}.log"
        self.process: subprocess.Popen[bytes] | None = None
        self.instance = uuid.uuid4().hex
        self.external = False  # started by another process; never stopped from here

    @property
    def url(self) -> str:
        """Websocket address of the server."""
        return f"ws://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        """Whether the subprocess this object started is still running."""
        return self.process is not None and self.process.poll() is None

    def start(self) -> PolicyServer:
        """Spawn the server and block until its port answers (or it dies)."""
        spec = self.spec
        if not spec.interpreter or not Path(spec.interpreter).exists():
            raise FileNotFoundError(f"{spec.name}: interpreter {spec.interpreter!r} not found")
        if spec.checkpoint and not Path(spec.checkpoint).exists():
            raise FileNotFoundError(
                f"{spec.name}: checkpoint {spec.checkpoint} missing; fetch it with: {spec.download}"
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(spec.env)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        env.setdefault("PYTHONUNBUFFERED", "1")
        cmd = spec.command(self.port)
        log.info("starting policy server %s on gpu %s: %s", spec.name, self.gpu, " ".join(cmd))
        with open(self.log_path, "ab") as logf:
            logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
            self.process = subprocess.Popen(
                cmd,
                cwd=spec.cwd or None,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._wait_ready()
        _write_marker(self)
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.spec.boot_timeout_s
        while time.monotonic() < deadline:
            if not self.alive():
                rc = self.process.returncode if self.process else "?"
                raise RuntimeError(
                    f"policy server {self.spec.name} exited rc={rc}"
                    f" (see {self.log_path}):\n{self.log_tail()}"
                )
            if not _port_free(self.port):
                return
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(
            f"policy server {self.spec.name} did not open port {self.port} within "
            f"{self.spec.boot_timeout_s:.0f}s (see {self.log_path}):\n{self.log_tail()}"
        )

    def log_tail(self, n_bytes: int = 2000) -> str:
        """The last ``n_bytes`` of the server log."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - n_bytes))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def stop(self) -> None:
        """Terminate the subprocess (whole process group); external servers are left alone."""
        proc = self.process
        if proc is None or self.external:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except Exception:
                with contextlib.suppress(Exception):
                    os.killpg(proc.pid, signal.SIGKILL)
        self.process = None
        with contextlib.suppress(OSError):
            _marker_path(self.port).unlink()


def _marker_path(port: int) -> Path:
    return Path(f"/tmp/roboenv-vla-{os.getuid()}") / f"{port}.json"


def _write_marker(server: PolicyServer) -> None:
    """Record who serves this port, so another checkout never adopts it by accident."""
    from roboenv_runtime.paths import repo_root

    try:
        path = _marker_path(server.port)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "repo": str(repo_root()),
                    "policy": server.spec.name,
                    "checkpoint": str(server.spec.checkpoint or ""),
                    "pid": server.process.pid if server.process else None,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _foreign(spec: PolicySpec, port: int) -> str | None:
    """Why the server on ``port`` must not be adopted, or ``None`` when it may be."""
    from roboenv_runtime.paths import repo_root

    try:
        marker = json.loads(_marker_path(port).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # started by something that keeps no marker: assume the old rule
    pid = marker.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                _marker_path(port).unlink()  # a dead server's marker is stale
            return None
    if marker.get("repo") != str(repo_root()):
        return f"it belongs to {marker.get('repo')}"
    if marker.get("policy") != spec.name:
        return f"it serves {marker.get('policy')}, not {spec.name}"
    ckpt = marker.get("checkpoint")
    if spec.checkpoint and ckpt and ckpt != str(spec.checkpoint):
        return f"it serves checkpoint {ckpt}"
    return None


def _attached(spec: PolicySpec, gpu: str, port: int) -> PolicyServer | None:
    """A server another process already runs on ``port`` (one per GPU is shared).

    The liveness probe has to speak the policy's own protocol: an openpi server
    greets a new connection with its metadata, while an XPolicyLab one says
    nothing until the client sends HELLO — probing the latter with the openpi
    client blocks on ``recv`` forever (measured 2026-08-28 against the
    InternVLA-A1.5 RoboTwin server).
    """
    if _port_free(port):
        return None
    why = _foreign(spec, port)
    if why is not None:
        # Measured 2026-08-28: a clean-room replay returned reward 1.0 against a server
        # belonging to the development tree, and again with a nonexistent interpreter and
        # checkpoint -- the pre-flight lives in start(), which attaching skips.
        raise RuntimeError(
            f"port {port} already serves a policy but {why}; stop it, or set "
            f"ROBOENV_VLA_PORT_SHIFT to move this checkout's servers to other ports"
        )
    try:
        if spec.protocol == "xpolicylab-ws":
            from roboenv_runtime.vla.client_xpl import XplPolicyClient

            client: Any = XplPolicyClient(
                f"ws://127.0.0.1:{port}",
                spec=spec,
                reset_on_connect=False,
                connect_timeout_s=5.0,
                max_connect_seconds=30.0,
            )
        else:
            from roboenv_runtime.vla.client import WebsocketPolicyClient

            client = WebsocketPolicyClient(f"ws://127.0.0.1:{port}", connect_timeout_s=5.0)
        client.close()
    except Exception:  # something else owns the port
        return None
    server = PolicyServer(spec, gpu, port=port)
    server.external = True
    return server


def ensure_server(spec: PolicySpec, gpu: str | None = None) -> PolicyServer:
    """The server for ``spec`` on ``gpu``.

    Reused if this process started it, attached if another process on this host serves
    it on the GPU's port, started otherwise.
    """
    gpu = str(gpu if gpu is not None else current_gpu())
    key = (spec.name, gpu)
    server = _SERVERS.get(key)
    if server is not None and (server.alive() or (server.external and not _port_free(server.port))):
        return server
    shift = int(os.environ.get("ROBOENV_VLA_PORT_SHIFT", "0") or 0)
    try:
        preferred = spec.port_base + int(gpu) + shift
    except ValueError:
        preferred = spec.port_base + shift
    server = _attached(spec, gpu, preferred)
    if server is None:
        server = PolicyServer(spec, gpu, port=preferred if _port_free(preferred) else None).start()
    _SERVERS[key] = server
    return server


def stop_all() -> None:
    """Stop every server this process started."""
    for server in list(_SERVERS.values()):
        server.stop()
    _SERVERS.clear()


atexit.register(stop_all)


def inventory():
    """Public evidence of which policy-server processes this episode actually owns."""
    return [{"policy": server.spec.name, "instance": server.instance,
             "owned": not server.external and server.process is not None,
             "pid": server.process.pid if server.process else None,
             "alive": server.alive()}
            for server in _SERVERS.values()]
