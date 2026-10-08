"""Go2 replay identities track Isaac inputs without relabeling old evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import load_tool
from roboenv_runtime.adapters.isaaclab.robots.go2_d1.environment import Go2Env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def identity_inputs(tmp_path, monkeypatch):
    lock = json.loads((ROOT / "upstreams.lock.json").read_text())
    # Exercise dependency traversal as well as the selected backend itself.
    lock["sources"]["isaaclab_native"]["requires"] = ["robosuite"]
    path = tmp_path / "upstreams.lock.json"
    path.write_text(json.dumps(lock))
    dependencies = tmp_path / "runtime/locks/isaaclab-resolved.txt"
    dependencies.parent.mkdir(parents=True)
    dependencies.write_bytes((ROOT / "runtime/locks/isaaclab-resolved.txt").read_bytes())
    monkeypatch.setenv("ROBOENV_ROOT", str(tmp_path))
    return lock, path, dependencies


@pytest.mark.parametrize("change, matches", [
    ("unrelated_revision", True),
    ("unrelated_patch", True),
    ("formatting", True),
    ("isaac_revision", False),
    ("isaac_patch", False),
    ("dependency_revision", False),
    ("dependency_selection", False),
    ("dependency_lock", False),
])
def test_source_identity_tracks_only_isaac_inputs(identity_inputs, change, matches):
    lock, path, dependencies = identity_inputs
    original = Go2Env.source_identity()
    if change == "unrelated_revision":
        lock["sources"]["behavior"]["revision"] = "0" * 40
    elif change == "unrelated_patch":
        lock["sources"]["behavior"]["patches"][0]["sha256"] = "0" * 64
    elif change == "isaac_revision":
        lock["sources"]["isaaclab_native"]["revision"] = "0" * 40
    elif change == "isaac_patch":
        lock["sources"]["isaaclab_native"]["patches"][0]["sha256"] = "0" * 64
    elif change == "dependency_revision":
        lock["sources"]["robosuite"]["revision"] = "0" * 40
    elif change == "dependency_selection":
        lock["sources"]["isaaclab_native"]["requires"].append("libero")
    elif change == "dependency_lock":
        dependencies.write_bytes(dependencies.read_bytes() + b"\n# changed dependency lock\n")
    path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n")
    assert (Go2Env.source_identity() == original) is matches


def test_source_identity_rejects_unknown_lock_schema(identity_inputs):
    lock, path, _ = identity_inputs
    lock["schema"] = "roboenv/upstreams/unknown"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Unsupported upstream lock"):
        Go2Env.source_identity()


@pytest.mark.parametrize("relative,changes_reach", [
    ("robots/go2_d1/controllers.py", True),
    ("common/observations.py", True),
    ("tasks/go2_d1_reach/scoring.py", True),
    ("tasks/go2_d1_pick_place/scoring.py", False),
])
def test_replay_identity_covers_shared_and_selected_task_sources(
    identity_inputs, monkeypatch, relative, changes_reach,
):
    original = Go2Env.source_identity("go2-d1-reach")
    read_bytes = Path.read_bytes

    def changed(path):
        content = read_bytes(path)
        if path.as_posix().endswith(relative):
            return content + b"\n# changed implementation\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert (Go2Env.source_identity("go2-d1-reach") != original) is changes_reach


@pytest.mark.parametrize("legacy", [False, True])
def test_replay_rejects_changed_or_legacy_sources_before_advancing(
    identity_inputs, tmp_path, monkeypatch, legacy,
):
    sources = Go2Env.source_identity()
    current = {
        "schema": "roboenv/go2-d1-record/1", "seed": 0,
        "asset_sha256": "fixture", "task_definition": {}, "device_profiles": {},
        "sources": sources,
    }
    recorded = copy.deepcopy(current)
    key = "upstreams.lock.json#isaaclab"
    if legacy:
        del recorded["sources"][key]
        key = "upstreams.lock.json"
    recorded["sources"][key] = "0" * 64
    calls = []
    world = SimpleNamespace(
        reset=lambda seed: calls.append("reset"),
        record=lambda: current,
        close=lambda: calls.append("close"),
        advance=lambda: pytest.fail("Identity mismatch must not advance physics"),
    )
    replay_tool = load_tool("replay_go2_d1")
    monkeypatch.setattr(replay_tool, "Go2Env", lambda **kwargs: world)
    output = tmp_path / "replay.json"
    with pytest.raises(ValueError, match="Replay identity mismatch: sources"):
        replay_tool.replay(recorded, "unused", output)
    assert calls == ["reset", "close"]
    assert not output.exists()
