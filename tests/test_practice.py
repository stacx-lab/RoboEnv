"""Practice reset remains available; removed rewind requests never execute."""

import json
import threading

import pytest
from roboenv.control.practice import PracticeClient
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError, handle_mapping


@pytest.mark.parametrize("budget_mode", ["official", "practice"])
def test_removed_rewind_requests_have_no_effects(tmp_path, monkeypatch, budget_mode):
    worker = SimWorker(FakeShell(), task_id="toy", seed=0, run_dir=tmp_path,
                       budget_mode=budget_mode, execution_mode="stepped", session="test")
    client = PracticeClient(tmp_path / "unused.sock", session="test", generation=0)
    exchanged = []

    def exchange(payload, *args):
        request = json.loads(payload)
        exchanged.append(request["op"])
        return (json.dumps(handle_mapping(worker, request)) + "\n").encode()

    monkeypatch.setattr(client, "_unix_exchange", exchange)
    try:
        worker.adapter.push(0.3)
        assert worker.scorer.peek()["latched"]
        pending = worker.world.wait(1)
        before = worker.observe()
        for op in ("checkpoint", "restore"):
            args = {} if op == "checkpoint" else {"token": "old-token"}
            with pytest.raises(ValueError, match="not an agent operation"):
                client.request(op, args)
            with pytest.raises(WorkerRPCError) as rejected:
                worker.dispatch(op, args)
            assert rejected.value.code == "unsupported_op"
            response = handle_mapping(worker, {
                "protocol": "roboenv/1", "session": "test", "generation": 0,
                "request_id": op, "deadline_s": 10, "op": op, "args": args,
            })
            assert response["ok"] is False
            assert response["error"]["code"] == "unsupported_op"
            assert worker.observe() == before
            assert worker.world.time_s == 0 and not pending.done()
            assert worker.state == "ready"
        assert exchanged == []
        if budget_mode == "practice":
            assert client.request("reset", {"seed": 1})["seed"] == 1
            assert exchanged == ["health", "reset"]
            assert pending.exception().code == "episode_finished"
            assert worker.adapter.used == 0 and not worker.scorer.latched
        else:
            with pytest.raises(ValueError, match="explicitly configured practice task"):
                client.request("reset", {"seed": 1})
            assert exchanged == ["health"]
            assert worker.observe() == before and not pending.done()
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
def test_reset_clears_clock_success_and_pending_work(tmp_path, monkeypatch, mode):
    class TickedToy(FakeShell):
        def advance_control(self):
            self.used += 1

    now = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    worker = SimWorker(TickedToy(), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode=mode)

    def motion():
        while True:
            yield

    try:
        completed = worker.world.wait(0.04)
        for _ in range(2):
            now[0] += 0.021
            worker.world.advance()
        assert completed.result()["waited_s"] == pytest.approx(0.04)
        worker.adapter.pos[0] = worker.adapter.goal_x
        assert worker.scorer.peek()["latched"]
        waiting = worker.world.wait(10)
        moving = worker.world.schedule(motion(), threading.Event())
        now[0] = 200.0
        worker.dispatch("reset", {"seed": 1})
        assert waiting.exception().code == "episode_finished"
        assert moving.exception().code == "operation_interrupted"
        assert worker.world.time_s == 0 and worker.adapter.used == 0
        assert worker.seed == worker.adapter.seed == 1
        assert not worker.scorer.latched and not worker.scorer.submitted
        assert not worker.world.pending and not worker.world._frozen
        assert worker.world.failure is None and worker.world._episode_error is None
        assert worker.world.next_tick == now[0]
        resumed = worker.world.wait(0.02)
        now[0] += 0.021
        worker.world.advance()
        assert resumed.result()["waited_s"] == pytest.approx(0.02)
        assert worker.adapter.used == 1
    finally:
        worker.close()
