"""Budget termination keeps real worker responses and journals scoreable."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import BoundDevice, DeviceDefinition
from roboenv.devices.service import DeviceService
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.bindings.world import clock_actions
from roboenv_runtime.runtime.atomic import AtomicJournal
from roboenv_runtime.runtime.worker import SimWorker, handle_mapping

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / "src/roboenv/assets/verify.py"))


class TickedToy(FakeShell):
    def configure_execution_mode(self, mode):
        super().configure_execution_mode(mode)
        self.bounded_episodes = False  # Like native adapters without a terminal callback.

    def advance_control(self):
        self.used += 1
        if getattr(self, "complete_at_horizon", False) and self.budget().exhausted:
            self.pos[0] = self.goal_x


@pytest.mark.parametrize("duration", [0.06, 0.2])
@pytest.mark.parametrize("completed", [False, True])
def test_worker_budget_retains_verdict_and_certifiable_journal(tmp_path, duration, completed):
    worker = SimWorker(TickedToy(max_budget=3), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode="stepped", session="test")
    worker.adapter.complete_at_horizon = completed
    context = BindingContext("fakeshell", "general", worker.adapter, world=worker.world)
    clock = BoundDevice(DeviceDefinition("world", "world", {}), clock_actions(context),
                        implementation="clock")
    worker.devices = DeviceService([clock], execution_lock=worker.world.lock,
                                   resolve=worker.world.resolve)
    journal = AtomicJournal(tmp_path / "journal.jsonl")

    def call(ident, op, args):
        request = {"protocol": "roboenv/1", "session": "test", "generation": 0,
                   "request_id": ident, "deadline_s": 10, "op": op, "args": args}
        journal.intent(request)
        response = handle_mapping(worker, request)
        (journal.commit if response["ok"] else journal.error)(request, response)
        return response

    try:
        response = call("wait", "devices.call", {"device": "world", "action": "wait",
                                                "args": {"duration_s": duration}})
        assert response["ok"] is (duration == 0.06)
        if not response["ok"]:
            assert response["error"]["code"] == "episode_finished"
            assert response["error"]["details"]["effects"] == "partial"
        assert worker.world.time_s == pytest.approx(0.06)
        assert worker.world.failure is None
        result = call("submit", "submit", {})["result"]
        assert result["budget"]["used"] == result["budget"]["max"] == 3
        assert result["completed"] is completed
        assert worker.simulation_status(include_verdict=True)["reason"] == (
            "success" if completed else "step_limit"
        )
        assert worker.dispatch("submit", {}) == result
        assert worker.dispatch("observe", {})["budget"] == result["budget"]
        assert worker.dispatch("simulation.stop", {})["terminal"]
        assert VERIFY["validate_journal"](journal.path.read_bytes(), session="test",
                                          submit_id="submit", verdict=result)["requests"] == 2
        assert VERIFY["validate_score"](result) is False
        worker.dispatch("reset", {"seed": 1})
        worker.world.resolve(worker.world.wait(0.02))
        assert worker.adapter.used == 1 and worker.world.failure is None
    finally:
        worker.close()


def history(error):
    identity = {"protocol": "roboenv/1", "session": "s", "generation": 0}
    write = {**identity, "request_id": "w", "op": "devices.call", "deadline_s": 10,
             "args": {"device": "arm", "action": "write", "args": {}}}
    submit = {**identity, "request_id": "s", "op": "submit", "deadline_s": 10, "args": {}}
    verdict = {"completed": False, "reward": 0.0}
    events = [
        {"journal_version": 1, "phase": "intent", "request": write},
        {"journal_version": 1, "phase": "error", "request": write,
         "response": {**identity, "request_id": "w", "ok": False, "error": error}},
        {"journal_version": 1, "phase": "intent", "request": submit},
        {"journal_version": 1, "phase": "commit", "request": submit,
         "response": {**identity, "request_id": "s", "ok": True, "result": verdict}},
    ]
    return SimpleNamespace(events=events, verdict=verdict)


def validate(history):
    raw = "".join(json.dumps(event) + "\n" for event in history.events).encode()
    return VERIFY["validate_journal"](raw, session="s", submit_id="s", verdict=history.verdict)


@pytest.mark.parametrize("effects", ["none", "partial"])
def test_explicit_episode_effects_are_certifiable(effects):
    assert validate(history({"code": "episode_finished", "details": {"effects": effects}}))


@pytest.mark.parametrize("error", [
    {"code": "episode_finished"},
    {"code": "episode_finished", "details": {}},
    {"code": "episode_finished", "details": None},
    {"code": "episode_finished", "details": []},
    {"code": "episode_finished", "details": {"effects": "unknown"}},
    {"code": "episode_finished", "details": {"effects": "read_only"}},
    {"code": "episode_finished", "details": {"effects": ["partial"]}},
    {"code": "motion_failed", "details": {"effects": "partial"}},
    {"code": "stop_failed", "details": {"effects": "unknown"}},
])
def test_uncertain_or_unrelated_errors_still_fail(error):
    with pytest.raises(ValueError, match="indeterminate"):
        validate(history(error))


@pytest.mark.parametrize("change", ["identity", "missing_submit", "after_submit"])
def test_episode_error_does_not_bypass_journal_invariants(change):
    value = history({"code": "episode_finished", "details": {"effects": "partial"}})
    if change == "identity":
        value.events[1]["response"]["session"] = "wrong"
    elif change == "missing_submit":
        value.events = value.events[:2]
    else:
        value.events = copy.deepcopy(value.events[2:] + value.events[:2])
    with pytest.raises(ValueError):
        validate(value)


def test_native_terminal_callback_seals_after_buffer_cleanup(tmp_path):
    import threading

    class AutoToy(TickedToy):
        def configure_execution_mode(self, mode):
            super().configure_execution_mode(mode)
            self.bounded_episodes = True
            self.cleaned = False

        def set_step_callback(self, callback):
            self.callback = callback

        def advance_control(self):
            super().advance_control()
            self.callback()

        def observe(self):
            return {**super().observe(), "cleaned": self.cleaned}

    worker = SimWorker(AutoToy(max_budget=3), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode="stepped")

    def pending():
        try:
            for _ in range(10):
                yield
        finally:
            worker.adapter.cleaned = True

    try:
        worker.world.schedule(pending(), threading.Event())
        with pytest.raises(RuntimeError, match="budget exhausted"):
            worker.world.resolve(worker.world.wait(1))
        assert worker._final_result is not None
        assert worker.observe()["cleaned"]
        assert worker.world.time_s == pytest.approx(0.06)
        assert worker.dispatch("simulation.stop", {})["reason"] == "step_limit"
    finally:
        worker.close()
