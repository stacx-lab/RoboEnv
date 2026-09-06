"""Public SDK and standalone verifier against a synthetic, journaled service."""

import importlib.resources
import json
import os
import socketserver
import subprocess
import sys
import threading
import uuid

import pytest
from roboenv.client import Client


@pytest.fixture
def service(tmp_path):
    # Abstract-free fixture: actual Unix connections, independent JSON client,
    # fresh state for each test, no installed simulator or agent package needed.
    journal = tmp_path / "journal/actions.jsonl"
    journal.parent.mkdir()
    state = {"position": 0.0, "used": 0, "submitted": False}
    cache = {}
    api = {
        "tools": {
            "push": {"parameters": ["distance"], "read_only": False},
            "get_position": {"parameters": ["name"], "read_only": True},
        }
    }

    def score():
        return {
            "reward": float(state["position"] >= 0.3),
            "completed": state["position"] >= 0.3,
            "submitted": state["submitted"],
            "budget": {"used": state["used"], "max": 100, "unit": "sim_steps"},
        }

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            key = request["request_id"]
            if key in cache:
                self.wfile.write(json.dumps(cache[key]).encode() + b"\n")
                return
            op = request["op"]
            if op == "health":
                result = {
                    "worker": {"state": "ready"},
                    "task": "mock.push",
                    "seed": 0,
                    "budget_mode": "official",
                }
            elif op == "api":
                result = api
            elif op == "call":
                if request["args"]["name"] == "push":
                    state["position"] += request["args"]["kwargs"]["distance"]
                    state["used"] += 10
                result = state["position"]
            elif op == "submit":
                state["submitted"] = True
                result = score()
            elif op == "score.peek":
                result = score()
            elif op == "observe":
                result = {"budget": score()["budget"], "position": state["position"]}
            else:
                raise AssertionError(op)
            response = {
                name: request[name] for name in ("protocol", "session", "generation", "request_id")
            }
            response.update(ok=True, result=result)
            if op != "health":
                with journal.open("a") as out:
                    for phase in ("intent", "commit"):
                        row = {"journal_version": 1, "phase": phase, "request": request}
                        if phase == "commit":
                            row["response"] = response
                        out.write(json.dumps(row) + "\n")
            cache[key] = response
            self.wfile.write(json.dumps(response).encode() + b"\n")

    # Filesystem aliases keep the standalone verifier compatible with long checkout paths.
    address = str(tmp_path / "s.sock")
    if len(address.encode()) >= 100:
        # The socket can be addressed through a short symlink in the refactor cache.
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / ".cache" / "s"
        root.mkdir(parents=True, exist_ok=True)
        alias = root / uuid.uuid4().hex[:8]
        alias.symlink_to(tmp_path, target_is_directory=True)
        address = str(alias / "s.sock")
    with socketserver.UnixStreamServer(address, Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield Client(address, session="harbor"), state, address, tmp_path
        finally:
            server.shutdown()
            thread.join()


def test_identity_cache_does_not_duplicate_an_action(service):
    client, state, _, _ = service
    for _ in range(2):
        client.request("call", {"name": "push", "kwargs": {"distance": 0.1}}, request_id="same")
    assert state["position"] == pytest.approx(0.1)
    assert state["used"] == 10
    assert client.call("get_position", name="puck") == pytest.approx(0.1)


def test_oracle_and_verifier_need_no_agent_runtime(service):
    _, _, address, temporary = service
    program = temporary / "solution.py"
    program.write_text('push(0.3)\nassert get_position(name="puck") >= 0.3\n')
    verifier = temporary / "verify.py"
    verifier.write_text(
        importlib.resources.files("roboenv").joinpath("assets/verify.py").read_text()
    )
    (temporary / "task.json").write_text(
        json.dumps({"task_id": "mock.push", "seed": 0, "session": "harbor"})
    )
    env = {
        **os.environ,
        "ROBOENV_SOCKET": address,
        "ROBOENV_RUN_DIR": str(temporary),
        "ROBOENV_VERIFIER_OUT": str(temporary / "verdict"),
    }
    replay = subprocess.run(
        [sys.executable, "-m", "roboenv.cli", "replay", str(program)],
        cwd=temporary,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)["program_completed"]
    checked = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=temporary,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads((temporary / "verdict/reward.json").read_text()) == {"reward": 1.0}


@pytest.mark.parametrize("operation", ["submit", "reset", "restore", "shutdown", "exec"])
def test_public_client_refuses_trial_control_operations(operation):
    with pytest.raises(ValueError, match="not an agent operation"):
        Client("/unused").request(operation)
