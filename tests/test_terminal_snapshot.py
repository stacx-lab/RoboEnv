"""Submitted trials retain one state and every pre-submit accounting charge."""

import copy
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from roboenv.devices.model import Action, BoundDevice, DeviceDefinition
from roboenv.devices.service import DeviceService
from roboenv_runtime.adapters.fakeshell import FakeShell
from roboenv_runtime.core.types import Budget
from roboenv_runtime.runtime.scoring import SimpleScorer
from roboenv_runtime.runtime.worker import SimWorker, WorkerRPCError

ROOT = Path(__file__).resolve().parents[1]


class CountedObservationWorld(FakeShell):
    """Seven charged observation reads exercise terminal budget stability."""

    def __init__(self, max_budget=60_000):
        super().__init__(max_budget=max_budget)
        self.calls = self.observations = self.evaluations = self.motions = 0

    def reset(self, seed):
        super().reset(seed)
        self.calls = self.observations = self.evaluations = self.motions = 0

    def get_object_pose(self, name):
        self.calls += 1
        return super().get_object_pose(name)

    def push(self, dx=0.1):
        self.calls += 1
        self.motions += 1
        return super().push(dx)

    def observe(self):
        self.observations += 1
        self.calls += 7
        return super().observe()

    def score(self):
        self.evaluations += 1
        return super().score()

    def budget(self):
        return Budget(used=self.calls, max=self.max_budget, unit="api_calls")


def worker(tmp_path, **kwargs):
    return SimWorker(
        CountedObservationWorld(**kwargs),
        task_id="counted-observation",
        seed=0,
        run_dir=tmp_path,
        session="test",
        execution_mode="stepped",
        device_root=ROOT / "tests/fixtures/native-toy/catalog/robots/toy-puck",
    )


def test_submit_seals_observation_after_preserving_all_internal_read_charges(tmp_path):
    instance = worker(tmp_path)
    try:
        instance.adapter.calls = 56
        instance.dispatch(
            "devices.call", {"device": "controller", "action": "write", "args": {"target": [0.3]}}
        )
        assert instance.adapter.budget().used == 57
        result = instance.dispatch("submit", {})
        assert result["budget"] == {"used": 64, "max": 60_000, "unit": "api_calls"}
        assert result["completed"] and result["submitted"] and result["reward"] == 1
        for _ in range(3):
            assert instance.dispatch("submit", {}) == result
            assert instance.dispatch("score.peek", {}) == result
            observation = instance.dispatch("observe", {})
            assert observation["budget"] == result["budget"]
            assert observation["submitted"] is True
        assert instance.adapter.calls == 64
        assert instance.adapter.observations == instance.adapter.evaluations == 1
        verifier = runpy.run_path(str(ROOT / "src/roboenv/verification.py"))
        assert verifier["validate_score"](verifier["attach_budget"](result, observation))
    finally:
        instance.close()


def test_terminal_snapshots_are_deeply_detached_from_return_values(tmp_path):
    instance = worker(tmp_path)
    try:
        original = copy.deepcopy(instance.dispatch("submit", {}))
        reply = instance.dispatch("submit", {})
        reply["budget"]["used"] = -100
        observed = instance.dispatch("observe", {})
        observed["objects"]["puck"]["position"][0] = 99
        observed["budget"]["used"] = -100
        assert instance.dispatch("submit", {}) == original
        assert instance.dispatch("score.peek", {}) == original
        assert instance.dispatch("observe", {})["objects"]["puck"]["position"][0] == 0
        scorer_reply = instance.scorer.submit()
        scorer_reply["budget"]["used"] = -100
        assert instance.scorer.submit()["budget"] == original["budget"]
    finally:
        instance.close()


def test_post_submit_device_reads_cannot_change_authority_or_enable_motion(tmp_path):
    instance = worker(tmp_path)
    try:
        instance.devices = DeviceService(
            [
                BoundDevice(
                    DeviceDefinition("arm", "arm"),
                    {
                        "read": Action(
                            lambda args, context: instance.adapter.get_object_pose("puck")
                        ),
                        "write": Action(lambda args, context: instance.adapter.push(dx=0.3)),
                    },
                    implementation="test-counted-binding",
                )
            ],
            execution_lock=instance.world.lock,
        )
        result = instance.dispatch("submit", {})
        for _ in range(3):
            response = instance.dispatch(
                "devices.call", {"device": "arm", "action": "read", "args": {}}
            )
            assert response["budget"] == result["budget"]
        # Diagnostic queries keep their native accounting counter; they cannot
        # reopen or rewrite the already finished benchmark episode.
        assert instance.adapter.calls == result["budget"]["used"] + 3
        assert instance.dispatch("observe", {})["budget"] == result["budget"]
        assert instance.dispatch("submit", {}) == result
        for operation, arguments in [
            ("devices.call", {"device": "arm", "action": "write", "args": {}}),
        ]:
            with pytest.raises(WorkerRPCError, match="frozen|Episode finished"):
                instance.dispatch(operation, arguments)
        assert instance.adapter.motions == 0
    finally:
        instance.close()


def test_final_observation_cost_cannot_be_refunded_at_the_budget_boundary(tmp_path):
    instance = worker(tmp_path, max_budget=8)
    try:
        instance.dispatch(
            "devices.call", {"device": "controller", "action": "write", "args": {"target": [0.3]}}
        )
        result = instance.dispatch("submit", {})
        assert result["reward"] == 1
        assert result["budget"]["used"] == result["budget"]["max"] == 8
        verifier = runpy.run_path(str(ROOT / "src/roboenv/verification.py"))
        assert verifier["validate_score"](result) is False  # strict used < cap is unchanged
    finally:
        instance.close()


def test_reset_starts_a_new_snapshot_and_budget_epoch(tmp_path):
    instance = worker(tmp_path)
    try:
        instance.dispatch("submit", {})
        instance.dispatch("reset", {"seed": 1})
        observation = instance.dispatch("observe", {})
        assert observation["seed"] == 1
        assert observation["submitted"] is False
        assert instance._final_result is None and instance._final_observation is None
    finally:
        instance.close()


def test_failed_final_evaluation_is_never_retried(tmp_path):
    instance = worker(tmp_path)
    calls = []

    def fail():
        calls.append("evaluation")
        raise RuntimeError("failed final evaluation")

    instance.adapter.score = fail
    try:
        with pytest.raises(RuntimeError, match="failed final evaluation"):
            instance.dispatch("submit", {})
        assert instance.state == "failed"
        with pytest.raises(WorkerRPCError, match="owner recovery"):
            instance.dispatch("submit", {})
        with pytest.raises(RuntimeError, match="previously failed"):
            instance.scorer.submit()
        with pytest.raises(RuntimeError, match="previously failed"):
            instance.scorer.peek()
        assert calls == ["evaluation"]
    finally:
        instance.close()


def test_scorer_without_budget_is_paired_with_independent_final_native_budget(tmp_path):
    instance = worker(tmp_path)

    class BudgetlessAuthority(SimpleScorer):
        def submit(self):
            result = super().submit()
            result.pop("budget")
            return result

    instance.scorer = BudgetlessAuthority(instance.adapter)
    try:
        result = instance.dispatch("submit", {})
        assert result["budget"] == instance.adapter.budget().as_dict()
        assert instance.dispatch("observe", {})["budget"] == result["budget"]
    finally:
        instance.close()


def test_final_budget_mismatch_is_rejected_instead_of_overwritten(tmp_path):
    instance = worker(tmp_path)

    class IncorrectBudgetAuthority(SimpleScorer):
        def submit(self):
            result = super().submit()
            result["budget"]["used"] -= 1
            return result

    instance.scorer = IncorrectBudgetAuthority(instance.adapter)
    try:
        with pytest.raises(WorkerRPCError, match="final native budget disagree"):
            instance.dispatch("submit", {})
        assert instance.state == "failed"
        assert instance._final_result is None
    finally:
        instance.close()


def test_motion_queued_during_final_capture_cannot_escape_submit_freeze(tmp_path):
    instance = worker(tmp_path)
    captured, release = threading.Event(), threading.Event()
    native_observe = instance.adapter.observe

    def paused_observe():
        captured.set()
        assert release.wait(5)
        return native_observe()

    instance.adapter.observe = paused_observe
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            final = executor.submit(instance.dispatch, "submit", {})
            assert captured.wait(5)
            motion = executor.submit(
                instance.dispatch,
                "devices.call",
                {"device": "controller", "action": "write", "args": {"target": [0.3]}},
            )
            release.set()
            assert final.result(timeout=5)["submitted"] is True
            with pytest.raises(WorkerRPCError, match="frozen|Episode finished"):
                motion.result(timeout=5)
        assert instance.adapter.motions == 0
    finally:
        release.set()
        instance.close()


def test_unbounded_budget_verification_keeps_other_success_requirements():
    validate = runpy.run_path(str(ROOT / "src/roboenv/verification.py"))["validate_score"]
    result = {
        "reward": 1,
        "completed": True,
        "submitted": True,
        "budget": {"used": 100_000, "max": None, "unit": "action_frames"},
    }
    assert validate(result)
    assert not validate({**result, "completed": False})
    assert not validate({**result, "reward": 0})
    with pytest.raises(ValueError, match="budget fields"):
        validate({**result, "budget": {**result["budget"], "official_cap": 1000}})
    for maximum in (False, -1, 0, "unlimited"):
        with pytest.raises(ValueError):
            validate({**result, "budget": {**result["budget"], "max": maximum}})
    with pytest.raises(ValueError):
        validate({**result, "budget": {"used": 1, "unit": "action_frames"}})


@pytest.mark.parametrize("mode", ["stepped", "realtime"])
@pytest.mark.parametrize("latched", [False, True])
def test_restore_rewinds_clock_score_history_and_discards_pending_work(
    tmp_path, monkeypatch, mode, latched,
):
    class TickedSnapshot(FakeShell):
        def advance_control(self):
            self.used += 1

    now = [100.0]
    monkeypatch.setattr("roboenv_runtime.runtime.world.time.monotonic", lambda: now[0])
    instance = SimWorker(TickedSnapshot(), task_id="checkpoint", seed=0,
                         run_dir=tmp_path, execution_mode=mode)

    def advance(frames):
        waiting = instance.world.wait(frames * 0.02)
        for _ in range(frames):
            now[0] += 0.021
            instance.world.advance()
        assert waiting.done() and waiting.result()["waited_s"] == pytest.approx(frames * 0.02)

    def motion():
        while True:
            yield

    try:
        advance(2)
        instance.adapter.pos[0] = instance.adapter.goal_x if latched else 0
        assert instance.scorer.peek()["latched"] is latched
        token = instance.checkpoint()
        advance(3)
        instance.adapter.pos[0] = instance.adapter.goal_x
        assert instance.scorer.peek()["latched"]
        pending_wait = instance.world.wait(10)
        pending_motion = instance.world.schedule(motion(), threading.Event())
        now[0] = 200.0
        instance.restore({"token": token})
        assert instance.world.time_s == pytest.approx(0.04)
        assert instance.adapter.used == 2
        assert instance.scorer.peek()["latched"] is latched
        assert not instance.world.pending and not instance.world._frozen
        assert instance.world._episode_error is None and instance.world.failure is None
        assert instance.world.next_tick == now[0]
        assert pending_wait.exception().code == "episode_finished"
        assert pending_motion.exception().code == "operation_interrupted"
        advance(1)
        assert instance.world.time_s == pytest.approx(0.06)
        assert instance.adapter.used == 3
    finally:
        instance.close()


@pytest.mark.parametrize("seed", [0, 1])
def test_reset_invalidates_old_checkpoints_before_native_restore(tmp_path, seed):
    instance = SimWorker(FakeShell(), task_id="checkpoint", seed=0,
                         run_dir=tmp_path, execution_mode="stepped")
    try:
        token = instance.checkpoint()
        instance.reset({"seed": seed})
        with pytest.raises(WorkerRPCError, match="unknown checkpoint") as rejected:
            instance.restore({"token": token})
        assert rejected.value.code == "checkpoint_identity"
        assert instance.seed == instance.adapter.seed == seed
        assert instance.world.time_s == 0 and instance.state == "ready"
        fresh = instance.checkpoint()
        assert instance.restore({"token": fresh}) == {"restored": True}
    finally:
        instance.close()
