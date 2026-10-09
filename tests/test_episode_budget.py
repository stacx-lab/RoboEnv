"""Budget termination keeps real worker responses and journals scoreable."""

import copy
import json
import runpy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from motion_fixtures import Engine, bind_devices
from roboenv.devices.bindings.registry import BindingContext
from roboenv.devices.model import BoundDevice, DeviceDefinition
from roboenv.devices.service import DeviceService
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.bindings.world import clock_actions
from roboenv_runtime.core.types import ScoreReport
from roboenv_runtime.runtime.atomic import AtomicJournal
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError, handle_mapping

ROOT = Path(__file__).resolve().parents[1]
VERIFY = runpy.run_path(str(ROOT / "src/roboenv/verification.py"))


def journal_call(worker, journal, ident, op, args):
    request = {"protocol": "roboenv/1", "session": "test", "generation": 0,
               "request_id": ident, "deadline_s": 10, "op": op, "args": args}
    journal.intent(request)
    response = handle_mapping(worker, request)
    (journal.commit if response["ok"] else journal.error)(request, response)
    return response


class TickedToy(FakeShell):
    def advance_control(self):
        self.used += 1
        if getattr(self, "complete_at_horizon", False) and self.budget().exhausted:
            self.pos[0] = self.goal_x


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("signal", ["after_interval", "no_effects_error"])
def test_native_failure_seals_before_another_request(tmp_path, mode, signal):
    from roboenv.devices.model import DeviceError

    class NativeEnd(TickedToy):
        def advance_control(self):
            if signal == "no_effects_error":
                raise DeviceError("episode_finished", "object_dropped",
                                  details={"effects": "none"})
            super().advance_control()
            self.episode_done = True

        def observe(self):
            return {**super().observe(), "native_failure": "object_dropped"}

    worker = SimWorker(NativeEnd(), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode=mode)
    try:
        pending = worker.world.wait(0.02)
        worker.world.advance()
        assert pending.done()
        status = worker.simulation_status(include_verdict=True)
        assert status["terminal"] and not status["success"]
        assert status["reason"] == "native_done"
        assert worker.world.time_s == (0.02 if signal == "after_interval" else 0)
        assert worker.world.failure is None
        assert worker.observe()["native_failure"] == "object_dropped"
        assert json.loads((tmp_path / "terminal.json").read_text())["status"] == status
        worker.world.advance()
        assert worker.simulation_status(include_verdict=True) == status
    finally:
        worker.close()


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

    try:
        response = journal_call(worker, journal, "wait", "devices.call", {
            "device": "world", "action": "wait", "args": {"duration_s": duration},
        })
        assert response["ok"] is (duration == 0.06)
        if not response["ok"]:
            assert response["error"]["code"] == "episode_finished"
            assert response["error"]["details"]["effects"] == "partial"
        assert worker.world.time_s == pytest.approx(0.06)
        assert worker.world.failure is None
        assert worker.simulation_status()["terminal"]
        cancelled = journal_call(worker, journal, "cancel", "devices.cancel",
                                 {"execution_id": "ended-buffer"})
        assert not cancelled["ok"] and cancelled["error"]["code"] == "episode_finished"
        assert cancelled["error"]["details"]["effects"] == "none"
        result = journal_call(worker, journal, "submit", "submit", {})["result"]
        assert result["budget"]["used"] == result["budget"]["max"] == 3
        assert result["completed"] is completed
        assert worker.simulation_status(include_verdict=True)["reason"] == (
            "success" if completed else "step_limit"
        )
        assert worker.dispatch("submit", {}) == result
        assert worker.dispatch("observe", {})["budget"] == result["budget"]
        assert worker.dispatch("simulation.stop", {})["terminal"]
        saved = json.loads((tmp_path / "final-observation.json").read_text())
        assert saved == worker.observe()
        assert json.loads((tmp_path / "terminal.json").read_text())["evidence"]["complete"]
        assert VERIFY["validate_journal"](journal.path.read_bytes(), session="test",
                                          submit_id="submit", verdict=result)["requests"] == 3
        assert VERIFY["validate_score"](result) is False
        worker.dispatch("reset", {"seed": 1})
        worker.world.resolve(worker.world.wait(0.02))
        assert worker.adapter.used == 1 and worker.world.failure is None
    finally:
        worker.close()


def history(error, *, op="devices.call"):
    identity = {"protocol": "roboenv/1", "session": "s", "generation": 0}
    write = {**identity, "request_id": "w", "op": op, "deadline_s": 10,
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


@pytest.mark.parametrize("effects", [None, "partial", "unknown"])
def test_cancellation_with_uncertain_effects_remains_uncertifiable(effects):
    value = history({"code": "episode_finished", "details": {"effects": effects}},
                    op="devices.cancel")
    with pytest.raises(ValueError, match="indeterminate"):
        validate(value)


def test_no_effects_does_not_allow_forbidden_operations():
    value = history({"code": "episode_finished", "details": {"effects": "none"}}, op="reset")
    with pytest.raises(ValueError, match="operation forbidden"):
        validate(value)


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


def test_budget_seals_after_reentrant_buffer_cleanup(tmp_path):
    class CleanupToy(TickedToy):
        def observe(self):
            return {**super().observe(), "cleaned": list(cleaned)}

    cleaned = []
    worker = SimWorker(CleanupToy(max_budget=3), task_id="toy", seed=0, run_dir=tmp_path,
                       execution_mode="stepped")

    def pending(name):
        try:
            for _ in range(10):
                yield
        finally:
            cleaned.append(name)

    try:
        for name in ("left", "right"):
            future = worker.world.schedule(pending(name), threading.Event())
            future.add_done_callback(lambda _: worker._check_terminal())
        with pytest.raises(RuntimeError, match="budget exhausted"):
            worker.world.resolve(worker.world.wait(1))
        assert worker._final_result is not None
        assert worker.observe()["cleaned"] == ["left", "right"]
        assert worker.world.time_s == pytest.approx(0.06)
        assert worker.simulation_status(include_verdict=True)["reason"] == "step_limit"
    finally:
        worker.close()


class BufferedToy(Engine):
    stop_on_success = True

    def score(self):
        done = self.frame >= 3
        return ScoreReport(reward=float(done), raw_score=float(done), completed=done)

    def observe(self):
        return {"targets": dict(self.targets), "positions": dict(self.positions)}


def buffered_worker(tmp_path, *, stop_failure=False):
    native = BufferedToy()
    native.stop_failure = stop_failure
    worker = SimWorker(native, task_id="buffered", seed=0, run_dir=tmp_path,
                       execution_mode="stepped", reset_adapter=False, session="test")
    world = worker.world
    clock = BoundDevice(
        DeviceDefinition("world", "world", {}),
        clock_actions(BindingContext("fixture", "fixture", native, world=world)),
        implementation="clock",
    )
    motions = []

    def schedule(motion, cancel):
        future = world.schedule(motion, cancel)
        motions.append(future)
        return future

    worker.devices = DeviceService(
        [*bind_devices(native, world), clock], execution_lock=world.lock, motion_scheduler=schedule,
        resolve=world.resolve, flush=world.poll_motions,
    )
    return worker, motions


@pytest.mark.parametrize("duration", [0.03, 0.1])
def test_success_cleans_concurrent_buffers_before_sealing_and_certifies_journal(
    tmp_path, duration,
):
    worker, motions = buffered_worker(tmp_path)
    journal = AtomicJournal(tmp_path / "journal.jsonl")

    try:
        for name in ("left", "right"):
            response = journal_call(worker, journal, name, "devices.call", {
                "device": name, "action": "write", "args": {"actions": [[1]], "duration_s": 0.1},
            })
            assert response["ok"] and worker.adapter.frame == 0
        for motion in motions:
            motion.add_done_callback(lambda _: worker._check_terminal())
        # A second waiter observes only the fully cleaned, sealed terminal state.
        peer = worker.world.wait(0.2)
        snapshots = []
        peer.add_done_callback(lambda _: snapshots.append(worker.observe()))
        response = journal_call(worker, journal, "wait", "devices.call", {
            "device": "world", "action": "wait", "args": {"duration_s": duration},
        })
        assert response["ok"] is (duration == 0.03)
        if not response["ok"]:
            assert response["error"]["code"] == "episode_finished"
            assert response["error"]["details"]["effects"] == "partial"
        assert peer.exception().code == "episode_finished"
        assert worker.simulation_status(include_verdict=True)["reason"] == "success"
        assert worker.adapter.frame == 3 and worker.world.failure is None
        assert not worker.world.pending
        assert not worker.devices._active and not worker.devices._faults
        for motion in motions:
            assert motion.exception().code == "episode_finished"
            assert motion.exception().details["effects"] == "partial"
        snapshot = worker.observe()
        assert snapshot["targets"] == snapshot["positions"] == {"left": 0.03, "right": 0.03}
        assert snapshots == [snapshot]
        verdict = journal_call(worker, journal, "submit", "submit", {})["result"]
        assert VERIFY["validate_journal"](journal.path.read_bytes(), session="test",
                                          submit_id="submit", verdict=verdict)["requests"] == 4
        assert VERIFY["validate_score"](verdict)
        worker.adapter.targets["left"] = 2
        assert worker.observe() == snapshot
        assert worker.dispatch("submit", {}) == verdict
        assert worker.dispatch("simulation.stop", {})["success"]
        assert worker.dispatch("devices.call", {"device": "left", "action": "read", "args": {}})
        with pytest.raises(WorkerRPCError) as rejected:
            worker.dispatch("devices.call", {"device": "left", "action": "write",
                                              "args": {"actions": [[1]], "duration_s": 0.1}})
        assert rejected.value.code == "episode_finished"
        assert rejected.value.details["effects"] == "none"
        assert worker.adapter.frame == 3
    finally:
        worker.close()


def test_success_cleanup_failure_cannot_publish_a_successful_verdict(tmp_path):
    worker, motions = buffered_worker(tmp_path, stop_failure=True)
    try:
        worker.dispatch("devices.call", {"device": "left", "action": "write",
                                          "args": {"actions": [[1]], "duration_s": 0.1}})
        wait = worker.world.wait(0.1)
        with pytest.raises(WorkerRPCError, match="cleanup failed"):
            worker.world.resolve(wait)
        assert wait.exception().code == "stop_failed"
        assert wait.exception().details["effects"] == "unknown"
        assert motions[0].exception().code == "stop_failed"
        assert worker.world.failure is not None and worker.adapter.frame == 3
        assert worker._final_result is None and not worker.scorer.submitted
        assert not (tmp_path / "terminal.json").exists()
        with pytest.raises(WorkerRPCError, match="owner recovery"):
            worker.submit()
    finally:
        worker.close()


def test_success_at_buffer_boundary_preserves_normal_completion(tmp_path):
    worker, motions = buffered_worker(tmp_path)
    try:
        worker.dispatch("devices.call", {"device": "left", "action": "write",
                                          "args": {"actions": [[1]], "duration_s": 0.03}})
        assert worker.world.resolve(worker.world.wait(0.03))["time_s"] == pytest.approx(0.03)
        assert motions[0].result() is None
        assert worker.simulation_status(include_verdict=True)["success"]
        assert not any(event[2] == "stop" for event in worker.adapter.events)
        assert worker.adapter.targets["left"] == 1
    finally:
        worker.close()
