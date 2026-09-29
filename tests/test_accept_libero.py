"""Acceptance enables required cameras and selects the packaged Node smoke."""

from types import SimpleNamespace

import pytest
from helpers import load_tool


@pytest.mark.parametrize("exit_code", [0, 1])
def test_required_acceptance_gates_are_enabled_and_failures_propagate(
    tmp_path, monkeypatch, exit_code
):
    module = load_tool("accept_libero")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(module.subprocess, "run", run)
    args = SimpleNamespace(out=tmp_path / "acceptance", python=tmp_path / "python",
                           kinex=tmp_path / "kinex", node="selected-node")
    report = module.run(args)
    assert report["passed"] is (exit_code == 0)
    assert len(calls) == (4 if exit_code == 0 else 1)
    for _, kwargs in calls:
        assert kwargs["env"]["ROBOENV_LIBERO_RENDER_TESTS"] == "1"
        assert kwargs["env"]["ROBOENV_LIBERO_ACCEPTANCE"] == "1"
        assert kwargs["env"]["NODE"] == "selected-node"
    if exit_code == 0:
        assert calls[-1][0][-2:] == ["--node", "selected-node"]
