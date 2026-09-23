"""Thread-affine physics and RPC continue while another socket stays idle."""

import json
import socket
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.protocol import socket_address
from roboenv_runtime.runtime.worker import MainThreadWorkerUnixServer
from roboenv_runtime.runtime.world import WorldSession


class Native:
    name = "thread-affine-test"
    embodiment = "test"
    control_interval_s = 0.005
    owner_thread_required = True

    def __init__(self, maximum=None, fail=False):
        self.frames = 0
        self.maximum = maximum
        self.fail = fail
        self.threads = []
        self.closed = False
        self.progress = threading.Event()

    def budget(self):
        return Budget(used=self.frames, max=self.maximum)

    def advance_control(self):
        self.threads.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("physics failed")
        self.frames += 1
        if self.frames >= 5:
            self.progress.set()

    def close(self):
        self.closed = True


def test_idle_partial_connection_does_not_block_motion_or_other_rpc(tmp_path):
    native = Native()
    world = None
    rpc_threads = []

    def dispatch(op, args):
        rpc_threads.append(threading.get_ident())
        return {"op": op, "frames": native.frames}

    worker = SimpleNamespace(world=world, dispatch_async=dispatch, shutdown_requested=False)
    path = str(tmp_path / "worker.sock")
    server = MainThreadWorkerUnixServer(path, worker)
    ready = threading.Event()

    def serve():
        nonlocal world
        world = WorldSession(native)
        worker.world = world

        def motion():
            for _ in range(100):
                yield

        world.schedule(motion(), threading.Event())
        ready.set()
        try:
            server.serve_forever(poll_interval=0.01)
        finally:
            server.server_close()
            world.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(2)
    idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    active = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        idle.connect(socket_address(path))
        idle.sendall(b'{"id": "unfinished"')
        assert native.progress.wait(2)
        active.settimeout(2)
        active.connect(socket_address(path))
        stream = active.makefile("rb")
        try:
            for index in range(2):
                active.sendall(json.dumps({"id": index, "op": "read", "args": {}}).encode() + b"\n")
                response = json.loads(stream.readline())
                assert response["ok"] and response["result"]["frames"] >= 5
        finally:
            stream.close()
        assert rpc_threads == [thread.ident, thread.ident]
        assert set(native.threads) == {thread.ident}
    finally:
        idle.close()
        active.close()
        server.shutdown()
        thread.join(timeout=2)
    assert not thread.is_alive() and native.closed


def test_request_batch_is_bounded_and_still_services_motion(tmp_path):
    events = []
    worker = SimpleNamespace(
        world=SimpleNamespace(
            advance=lambda: events.append("tick"),
            poll_motions=lambda: None,
            pending=True,
            mode="stepped",
        ),
        dispatch_async=lambda op, args: events.append("request"),
    )
    server = MainThreadWorkerUnixServer(str(tmp_path / "batch.sock"), worker)
    futures = [Future() for _ in range(17)]
    try:
        for index, future in enumerate(futures):
            server._pending.put((future, {"id": index, "op": "read", "args": {}}))
        server.service_native_requests()
        assert events == ["request"] * 16 + ["tick"]
        assert all(future.done() for future in futures[:16]) and not futures[-1].done()
    finally:
        server.server_close()
    assert futures[-1].cancelled()


def test_new_connections_can_share_a_slow_native_frame(tmp_path):
    """Match WorkerClient's one-connection-per-request transport."""
    native = Native()
    original_tick = native.advance_control

    def slow_tick():
        time.sleep(0.05)
        original_tick()

    native.advance_control = slow_tick
    world = None
    worker = SimpleNamespace(
        world=world,
        dispatch_async=lambda op, args: native.frames,
        shutdown_requested=False,
    )
    path = str(tmp_path / "fresh-connections.sock")
    server = MainThreadWorkerUnixServer(path, worker)
    ready = threading.Event()

    def serve():
        nonlocal world
        world = WorldSession(native)
        worker.world = world

        def motion():
            for _ in range(100):
                yield

        world.schedule(motion(), threading.Event())
        ready.set()
        try:
            server.serve_forever(poll_interval=0.01)
        finally:
            server.server_close()
            world.close()

    thread = threading.Thread(target=serve)
    thread.start()
    frames = []
    try:
        assert ready.wait(2)
        assert native.progress.wait(2)
        for index in range(8):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2)
                client.connect(socket_address(path))
                client.sendall(json.dumps({"id": index, "op": "read", "args": {}}).encode() + b"\n")
                with client.makefile("rb") as stream:
                    frames.append(json.loads(stream.readline())["result"])
        assert len(set(frames)) < len(frames)
        assert set(native.threads) == {thread.ident}
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert not thread.is_alive() and native.closed


@pytest.mark.parametrize("backend", ["libero", "robocasa"])
def test_camera_rpc_uses_the_thread_that_created_the_native_renderer(tmp_path, backend):
    from concurrent.futures import ThreadPoolExecutor

    from roboenv_runtime.adapters.libero.adapter import LiberoAdapter
    from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
    from roboenv_runtime.runtime.worker import create_worker_server

    ready = Future()
    closed = []
    path = str(tmp_path / "camera.sock")

    def serve():
        owner = threading.get_ident()
        frames = [0]

        def capture():
            # A graphics context belongs to its creating thread, even if access
            # from a different thread is serialized with a mutex.
            if threading.get_ident() != owner:
                raise RuntimeError("rendering without the owning graphics context")
            return {"pixels": [23, 41, 97], "thread": owner}

        def close():
            closed.append(threading.get_ident())

        adapter_type = LiberoAdapter if backend == "libero" else RoboCasaAdapter
        adapter = adapter_type.__new__(adapter_type)
        adapter._env = SimpleNamespace(
            env=SimpleNamespace(control_freq=200),
            render=capture,
            tick=lambda: frames.__setitem__(0, frames[0] + 1),
            close=close,
            get_step_budget=lambda: {"used": frames[0], "max": None},
        )
        world = WorldSession(adapter)

        def dispatch(op, args):
            return adapter.capture()

        worker = SimpleNamespace(
            world=world,
            dispatch=dispatch,
            dispatch_async=dispatch,
            shutdown_requested=False,
        )
        server = create_worker_server(path, worker)
        ready.set_result(server)
        try:
            server.serve_forever(poll_interval=0.01)
        finally:
            server.server_close()
            world.close()

    thread = threading.Thread(target=serve)
    thread.start()
    server = ready.result(timeout=2)

    def read(index):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(socket_address(path))
            client.sendall(json.dumps({"id": index, "op": "capture"}).encode() + b"\n")
            with client.makefile("rb") as stream:
                response = json.loads(stream.readline())
            assert response["ok"], response
            assert response["result"] == {"pixels": [23, 41, 97], "thread": thread.ident}

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(read, range(8)))
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert closed == [thread.ident]
