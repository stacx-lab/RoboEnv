"""Safe socket setup and backpressure before worker dispatch."""

import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from roboenv_runtime.runtime.protocol import _remove_socket, socket_address
from roboenv_runtime.runtime.supervisor import Supervisor, SupervisorConfig


@pytest.fixture
def saturated_worker(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOENV_SOCKET_ROOT", raising=False)
    supervisor = Supervisor(SupervisorConfig(
        run_dir=tmp_path, task="test", seed=0, budget_mode="practice", session="test",
    ))
    monkeypatch.setattr(supervisor, "_refresh_worker_state", lambda: None)
    monkeypatch.setattr(supervisor, "_worker_alive", lambda: True)
    supervisor._worker_state = "ready"
    address = socket_address(supervisor.config.worker_socket)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        listener.listen(0)
        listener.settimeout(2)
        # Linux admits one pending connection with backlog zero. A second
        # timed connect gets EAGAIN until the server accepts the first.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as pending:
            pending.settimeout(2)
            pending.connect(address)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(2)
                with pytest.raises(BlockingIOError):
                    probe.connect(address)
            yield supervisor, listener


def test_worker_connection_waits_for_backlog_space(saturated_worker):
    supervisor, listener = saturated_worker
    with ThreadPoolExecutor(max_workers=1) as pool:
        connected = pool.submit(supervisor._connect_worker, time.monotonic() + 2)
        with pytest.raises(TimeoutError):
            connected.result(timeout=0.1)
        previous, _ = listener.accept()
        previous.close()
        with connected.result(timeout=2) as client:
            accepted, _ = listener.accept()
            with accepted:
                client.sendall(b"sent-once")
                assert accepted.recv(32) == b"sent-once"


def test_worker_connection_backpressure_respects_deadline(saturated_worker):
    supervisor, _ = saturated_worker
    with pytest.raises(TimeoutError, match="worker socket was not ready"):
        supervisor._connect_worker(time.monotonic() + 0.1)


def test_socket_cleanup_removes_only_the_socket(tmp_path):
    path = tmp_path / "stale.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(socket_address(path))
    _remove_socket(path)
    assert not path.exists()
    _remove_socket(path)  # A first launch has no old socket.


@pytest.mark.parametrize("symlink", [False, True])
def test_socket_cleanup_preserves_non_socket_paths(tmp_path, symlink):
    target = tmp_path / "retained"
    target.write_text("keep")
    path = tmp_path / "worker.sock" if symlink else target
    if symlink:
        path.symlink_to(target)
    with pytest.raises(RuntimeError, match="refusing to replace non-socket"):
        _remove_socket(path)
    assert target.read_text() == "keep"
    assert path.is_symlink() is symlink
