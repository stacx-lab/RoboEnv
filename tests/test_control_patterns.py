"""The SDK owns the single simulator-independent control helper implementation."""

import pytest
from roboenv.control.patterns import supervised_chunks, with_fallback


def test_chunking_respects_budget_and_preserves_instruction():
    calls = []

    def policy(**args):
        calls.append(args)
        return {"actions_taken": args["n_actions"], "reward_after": 0}

    result = supervised_chunks(policy, chunk=2, budget=5, instruction="move")
    assert calls == [{"n_actions": n, "instruction": "move"} for n in (2, 2, 1)]
    assert result.stopped_because == "frame budget spent"


def test_chunking_stops_on_success_without_another_call():
    calls = []

    def policy(**args):
        calls.append(args)
        return {"actions_taken": 1, "reward_after": 1}

    assert supervised_chunks(policy).solved
    assert len(calls) == 1


def test_invalid_chunk_budget_fails_before_policy_execution():
    def policy(**args):
        pytest.fail("policy must not run")

    with pytest.raises(ValueError, match="positive"):
        supervised_chunks(policy, budget=0)


def test_fallback_runs_after_stage_failure():
    def stage():
        raise ValueError("unavailable")

    assert with_fallback(stage, lambda: "recovered") == ("recovered", "fallback")
