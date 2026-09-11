"""Thread-affine physics and RPC continue while another socket stays idle."""

import json
import socket
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime import world as world_module
from roboenv_runtime.runtime.protocol import socket_address
from roboenv_runtime.runtime.worker import MainThreadWorkerUnixServer
from roboenv_runtime.runtime.world import WorldSession


class Native:
    name = "thread-affine-test"
    embodiment = "test"
    realtime_interval = 0.005
    realtime_on_main_thread = True

    def __init__(self, maximum=None, fail=False):
        self.frames = 0
        self.maximum = maximum
        self.fail = fail
        self.threads = []
        self.closed = False
        self.progress = threading.Event()

    def budget(self):
        return Budget(used=self.frames, max=self.maximum)

    def realtime_tick(self):
        self.threads.append(threading.get_ident())
        if self.fail:
            raise RuntimeError("physics failed")
        self.frames += 1
        if self.frames >= 5:
            self.progress.set()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_polled_clock_obeys_deadlines_freeze_and_failure(monkeypatch, fail):
    now = [0.0]
    monkeypatch.setattr(world_module.time, "monotonic", lambda: now[0])
    native = Native(maximum=2, fail=fail)
    world = WorldSession(native)
    world.start()
    assert world._thread is None
    with pytest.raises(RuntimeError, match="twice"):
        world.start()
    world.poll()
    assert native.frames == 0
    for _ in range(4):
        now[0] += 0.005
        world.poll()
    assert native.frames == (0 if fail else 2)
    assert world._halt.is_set()
    assert isinstance(world.failure, RuntimeError) if fail else world.failure is None
    assert set(native.threads) == {threading.get_ident()}
    world.close()
    world.poll()
    assert native.closed


def test_poll_rejects_a_different_thread():
    world = WorldSession(Native())
    world.start()
    errors = []

    def wrong_thread():
        try:
            world.poll()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=2)
    world.close()
    assert errors == ["native clock must run on its creating thread"]


def test_idle_partial_connection_does_not_block_clock_or_other_rpc(tmp_path):
    native = Native()
    world = WorldSession(native)
    rpc_threads = []

    def dispatch(op, args):
        rpc_threads.append(threading.get_ident())
        return {"op": op, "frames": native.frames}

    worker = SimpleNamespace(world=world, dispatch=dispatch, shutdown_requested=False)
    path = str(tmp_path / "worker.sock")
    server = MainThreadWorkerUnixServer(path, worker)
    ready = threading.Event()

    def serve():
        world.start()
        ready.set()
        try:
            server.serve_forever(poll_interval=world.poll_interval)
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


def test_request_batch_is_bounded_and_still_services_clock(tmp_path):
    events = []
    worker = SimpleNamespace(
        world=SimpleNamespace(poll=lambda: events.append("tick"), poll_interval=0.005),
        dispatch=lambda op, args: events.append("request"),
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
    original_tick = native.realtime_tick

    def slow_tick():
        time.sleep(0.05)
        original_tick()

    native.realtime_tick = slow_tick
    world = WorldSession(native)
    worker = SimpleNamespace(
        world=world, dispatch=lambda op, args: native.frames, shutdown_requested=False,
    )
    path = str(tmp_path / "fresh-connections.sock")
    server = MainThreadWorkerUnixServer(path, worker)
    ready = threading.Event()

    def serve():
        world.start()
        ready.set()
        try:
            server.serve_forever(poll_interval=world.poll_interval)
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
