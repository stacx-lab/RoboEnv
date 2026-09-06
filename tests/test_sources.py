"""Source locks are public, pinned, selective, and independent of checkout state."""

import hashlib
from pathlib import Path

import pytest
from roboenv.provisioning.sources import (
    git,
    load_lock,
    managed_tree_matches,
    selected_sources,
    status,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_and_patch_is_pinned_and_public():
    root, lock = load_lock(ROOT)
    for item in lock["sources"].values():
        assert len(item["revision"]) == 40
        assert item["url"].startswith("https://github.com/")
        for patch in item.get("patches", []):
            assert (
                hashlib.sha256((root / patch["path"]).read_bytes()).hexdigest() == patch["sha256"]
            )
    assert set(selected_sources(lock, ["robocasa"])) == {"robocasa", "robosuite"}
    assert set(selected_sources(lock, ["libero"])) == {"capx", "libero_pro", "libero_robosuite"}
    assert "libero_pro" not in selected_sources(lock, ["robotwin"])
    with pytest.raises(ValueError, match="unknown"):
        selected_sources(lock, ["not-a-backend"])


def test_status_does_not_claim_readiness_without_checkout(tmp_path, monkeypatch):
    import roboenv.provisioning.sources as upstreams

    lock = {
        "backends": ["test"],
        "sources": {
            "test": {
                "path": "third_party/test",
                "revision": "a" * 40,
                "url": "https://github.com/example/test.git",
                "backends": ["test"],
            }
        },
    }
    monkeypatch.setattr(upstreams, "load_lock", lambda project: (tmp_path, lock))
    report = status(tmp_path)
    assert report["ready"] is False
    assert report["sources"][0]["present"] is False
    assert str(tmp_path) not in str(report)


def test_source_cli_exits_unsuccessfully_when_not_ready(monkeypatch, capsys):
    import roboenv.provisioning.sources as upstreams
    from roboenv.cli.main import main

    monkeypatch.setattr(upstreams, "status", lambda *args: {"ready": False, "sources": []})
    assert main(["sources", "status"]) == 1
    assert '"ready": false' in capsys.readouterr().out


def test_exact_managed_tree_check_preserves_user_index(tmp_path):
    checkout = tmp_path / "source"
    checkout.mkdir()
    git(checkout, "init", "--initial-branch=main")
    (checkout / "controller.py").write_text("value = 1\n")
    git(checkout, "add", "controller.py")
    git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    original_index = (checkout / ".git/index").read_bytes()
    assert managed_tree_matches(tmp_path, checkout, [])
    assert (checkout / ".git/index").read_bytes() == original_index
    (checkout / "controller.py").write_text("value = 2\n")
    patch = tmp_path / "change.patch"
    patch.write_text(git(checkout, "diff") + "\n")
    patches = [{"path": "change.patch"}]
    assert managed_tree_matches(tmp_path, checkout, patches)
    assert (checkout / ".git/index").read_bytes() == original_index
    (checkout / "controller.py").write_text("value = 2\nextra = True\n")
    assert not managed_tree_matches(tmp_path, checkout, patches)
    (checkout / "controller.py").write_text("value = 2\n")
    (checkout / "unexpected.py").write_text("pass\n")
    assert not managed_tree_matches(tmp_path, checkout, patches)
    assert (checkout / ".git/index").read_bytes() == original_index
