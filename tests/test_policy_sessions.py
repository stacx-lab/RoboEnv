import json
import sys
from pathlib import Path

import pytest
from roboenv.policy import run_with_callback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/src"))

from roboenv_runtime.adapters.fakeshell import FakeShell  # noqa: E402
from roboenv_runtime.runtime.worker import SimWorker  # noqa: E402


class PolicyWorld(FakeShell):
    def __init__(self):
        super().__init__()
        self.rollouts = 0
        self.actions = 0
        self.ended = 0

    def tools(self):
        return {**super().tools(), "vla_run": self.vla_run}

    def vla_run(self, n_actions=8, until=None):
        self.rollouts += 1
        try:
            for _ in range(n_actions):
                self.push(0.1)
                self.actions += 1
                if until and until():
                    break
        finally:
            self.ended += 1
        return {"actions_taken": self.actions, "rollouts": self.rollouts}


class Client:
    def __init__(self, worker):
        self.worker = worker
        self.requests = []

    def request(self, op, args=None):
        args = args or {}
        json.dumps(args)
        self.requests.append((op, args))
        return self.worker.dispatch(op, args)


def test_until_is_per_action_without_restarting_policy(tmp_path):
    world = PolicyWorld()
    worker = SimWorker(world, task_id="test.policy", seed=0, run_dir=tmp_path, session="test")
    client = Client(worker)
    seen = []

    def until():
        position = worker.dispatch("call", {"name": "get_object_pose", "kwargs": {"name": "puck"}})[
            0
        ][0]
        seen.append(position)
        return position >= 0.3

    result = run_with_callback(client, {"n_actions": 8}, until)
    assert seen == pytest.approx([0.1, 0.2, 0.3])
    assert result == {"actions_taken": 3, "rollouts": 1}
    assert world.ended == 1 and worker._policy_session is None
    assert all(op == "vla" for op, _ in client.requests)
    worker.close()


def test_callback_exception_stops_only_policy_and_releases_controller(tmp_path):
    world = PolicyWorld()
    worker = SimWorker(world, task_id="test.policy", seed=0, run_dir=tmp_path, session="test")

    def fail():
        raise ValueError("predicate failed")

    with pytest.raises(ValueError, match="predicate failed"):
        run_with_callback(Client(worker), {"n_actions": 8}, fail)
    assert world.actions == world.ended == 1 and worker._policy_session is None
    assert not worker.scorer.submitted
    worker.close()


def test_verifier_submit_stops_a_paused_policy_without_another_action(tmp_path):
    world = PolicyWorld()
    worker = SimWorker(world, task_id="test.policy", seed=0, run_dir=tmp_path, session="test")
    state = worker.dispatch("vla", {"_interactive": "start", "arguments": {"n_actions": 8}})
    assert state["status"] == "paused" and world.actions == 1
    worker.dispatch("submit", {})
    assert world.actions == world.ended == 1 and worker.scorer.submitted
    worker.close()
