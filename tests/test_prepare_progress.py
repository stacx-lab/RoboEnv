"""Preparation reports progress without polluting JSON or leaking child output."""

import json
import threading

import pytest
from roboenv.cli.main import main
from roboenv.provisioning import setup
from roboenv.provisioning.progress import stage


@pytest.mark.parametrize("machine", [False, True])
def test_cli_reports_progress_on_stderr(monkeypatch, capsys, machine):
    def prepare(*args, progress, **kwargs):
        progress("Installing dependencies...")

    monkeypatch.setattr(setup, "prepare", prepare)
    monkeypatch.setattr(setup, "doctor", lambda *args, **kwargs: {"install_preflight_passed": True})
    assert main(["prepare", "robotwin", *(["--json"] if machine else [])]) == 0
    captured = capsys.readouterr()
    assert captured.err == "[prepare] Installing dependencies...\n"
    assert json.loads(captured.out) == {"install_preflight_passed": True}


def test_slow_stage_reports_heartbeat_and_stops_thread():
    seen = threading.Event()
    messages = []

    def report(message):
        messages.append(message)
        if "still running" in message:
            seen.set()

    with stage(report, "Installing", interval=0.01):
        assert seen.wait(2)
    assert messages[0] == "Installing..."
    assert messages[-1].startswith("Installing: done")


def test_stage_failure_is_reported_without_exception_details():
    messages = []
    with pytest.raises(ValueError, match="private detail"):
        with stage(messages.append, "Checking"):
            raise ValueError("private detail")
    assert "failed or interrupted" in messages[-1]
    assert "private detail" not in "\n".join(messages)


def test_quiet_api_stage_propagates_failure():
    with pytest.raises(ValueError):
        with stage(None, "Checking"):
            raise ValueError("failed")


def test_cli_prepare_reports_asset_verification(monkeypatch, capsys):
    monkeypatch.setattr(setup, "prepare", lambda *a, **kw: {"downloaded_assets_verified": True})
    monkeypatch.setattr(setup, "doctor", lambda *a, **kw: {"install_preflight_passed": True})
    assert main(["prepare", "robocasa", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["downloaded_assets_verified"] is True
