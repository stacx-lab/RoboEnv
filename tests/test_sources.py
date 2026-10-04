"""Source locks are public, pinned, selective, and independent of checkout state."""

import hashlib
import json
import subprocess
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


@pytest.mark.parametrize("current", [True, False])
def test_check_updates_queries_selected_remotes_without_mutation(tmp_path, monkeypatch, current):
    from roboenv.provisioning import sources

    revision = "a" * 40
    tip = revision if current else "b" * 40
    lock = {
        "backends": ["selected", "other"],
        "sources": {
            name: {
                "revision": revision,
                "url": f"https://github.com/example/{name}",
                "backends": [name],
            }
            for name in ("selected", "other")
        },
    }
    monkeypatch.setattr(sources, "load_lock", lambda project: (tmp_path, lock))
    calls = []

    def remote(root, *args, **kwargs):
        calls.append(args)
        return f"ref: refs/heads/main\tHEAD\n{tip}\tHEAD"

    monkeypatch.setattr(sources, "git", remote)
    report = sources.check_updates(tmp_path, ["selected"])
    assert report["up_to_date"] is current
    assert report["sources"][0]["branch"] == "refs/heads/main"
    assert report["sources"][0]["upstream"] == tip
    assert calls == [("ls-remote", "--symref", "https://github.com/example/selected", "HEAD")]
    assert list(tmp_path.iterdir()) == []


def test_check_updates_rejects_unresolved_remote(monkeypatch):
    from roboenv.provisioning import sources

    monkeypatch.setattr(sources, "git", lambda *args, **kwargs: "")
    with pytest.raises(ValueError, match="Cannot resolve"):
        sources.check_updates(ROOT, ["robotwin"])


@pytest.mark.parametrize("current", [True, False])
def test_check_updates_cli_exit_status(monkeypatch, capsys, current):
    from roboenv.cli.main import main
    from roboenv.provisioning import sources

    monkeypatch.setattr(sources, "check_updates", lambda *args: {"up_to_date": current})
    assert main(["sources", "check-updates", "--backends", "robotwin"]) == (0 if current else 1)
    assert '"up_to_date":' in capsys.readouterr().out


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
    assert set(selected_sources(lock, ["libero"])) == {"libero"}
    assert set(selected_sources(lock, ["robotwin"])) == {"robotwin"}
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


@pytest.fixture
def disposable_sources(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").touch()
    (root / "runtime").mkdir()
    (root / "runtime/pyproject.toml").touch()
    (root / "src/roboenv").mkdir(parents=True)
    git(root, "init", "--initial-branch=main")
    checkout = root / "third_party/test"
    checkout.mkdir(parents=True)
    git(checkout, "init", "--initial-branch=main")
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.invalid")
    git(checkout, "remote", "add", "origin", "https://github.com/example/test")
    (checkout / "controller.py").write_text("value = 1\n")
    (checkout / ".gitignore").write_text("assets/\n.cache/\nrecordings/\n.venv/\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "upstream")
    revision = git(checkout, "rev-parse", "HEAD")
    patch = root / "third_party/patches/change.patch"
    patch.parent.mkdir()
    patch.write_text(
        "--- a/controller.py\n+++ b/controller.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    item = {
        "path": "third_party/test", "url": "https://github.com/example/test",
        "revision": revision, "backends": ["test"],
        "patches": [{"path": str(patch.relative_to(root)),
                     "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}],
    }
    lock = {"schema": "roboenv/upstreams/1", "backends": ["test"], "sources": {"test": item}}
    (root / "upstreams.lock.json").write_text(json.dumps(lock))
    git(root, "update-index", "--add", "--cacheinfo", "160000", revision, item["path"])
    return root, checkout, patch, lock


def test_refresh_discards_edits_and_reapplies_on_every_call(disposable_sources):
    from roboenv.provisioning.sources import sync

    root, checkout, _, _ = disposable_sources
    assert sync(root, ["test"], reset=True)["ready"]
    (checkout / "controller.py").write_text("unrecorded = True\n")
    git(checkout, "add", "controller.py")
    (checkout / "old-patch-helper.py").write_text("obsolete = True\n")
    (checkout / " leading-space.py").write_text("obsolete = True\n")
    for name in ("assets", ".cache", "recordings", ".venv"):
        (checkout / name).mkdir()
        (checkout / name / "keep").write_text("preserve")
    assert sync(root, ["test"], reset=True)["ready"]
    assert (checkout / "controller.py").read_text() == "value = 2\n"
    assert not (checkout / "old-patch-helper.py").exists()
    assert not (checkout / " leading-space.py").exists()
    assert not git(checkout, "diff", "--cached")
    for name in ("assets", ".cache", "recordings", ".venv"):
        assert (checkout / name / "keep").read_text() == "preserve"
    assert sync(root, ["test"], reset=True)["ready"]


@pytest.mark.parametrize("invalid", ["checksum", "application"])
def test_refresh_validates_before_discarding_edits(disposable_sources, invalid):
    from roboenv.provisioning.sources import sync

    root, checkout, patch, lock = disposable_sources
    (checkout / "controller.py").write_text("keep until preflight succeeds\n")
    (checkout / "untracked.py").write_text("keep")
    patch.write_text(patch.read_text().replace("value = 1", "absent = 1"))
    if invalid == "application":
        lock["sources"]["test"]["patches"][0]["sha256"] = hashlib.sha256(
            patch.read_bytes()
        ).hexdigest()
        (root / "upstreams.lock.json").write_text(json.dumps(lock))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        sync(root, ["test"], reset=True)
    assert (checkout / "controller.py").read_text() == "keep until preflight succeeds\n"
    assert (checkout / "untracked.py").read_text() == "keep"


def test_refresh_changed_pin_and_removed_patch(disposable_sources):
    from roboenv.provisioning.sources import sync

    root, checkout, _, lock = disposable_sources
    (checkout / "controller.py").write_text("value = 3\n")
    git(checkout, "add", "controller.py")
    git(checkout, "commit", "-m", "new upstream")
    revision = git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "--detach", lock["sources"]["test"]["revision"])
    assert sync(root, ["test"])["ready"]
    lock["sources"]["test"].update(revision=revision, patches=[])
    (root / "upstreams.lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="dirty upstream"):
        sync(root, ["test"])
    assert sync(root, ["test"], reset=True)["ready"]
    assert git(checkout, "rev-parse", "HEAD") == revision
    assert (checkout / "controller.py").read_text() == "value = 3\n"


def test_refresh_preserves_nested_checkout_and_asset_link(disposable_sources):
    from roboenv.provisioning.sources import _discard_source_edits

    _, checkout, _, _ = disposable_sources
    child = checkout / "nested"
    child.mkdir()
    git(child, "init", "--initial-branch=main")
    (child / "keep.py").write_text("preserve")
    depot = checkout.parent / "depot"
    depot.mkdir()
    (depot / "keep").write_text("preserve")
    (checkout / "assets").symlink_to(depot, target_is_directory=True)
    (checkout / "controller.py").write_text("discard\n")
    _discard_source_edits(checkout, [child])
    assert (child / "keep.py").read_text() == "preserve"
    assert (checkout / "assets/keep").read_text() == "preserve"
    assert (checkout / "controller.py").read_text() == "value = 1\n"


def test_refresh_does_not_overwrite_ignored_asset_collision(disposable_sources):
    from roboenv.provisioning.sources import sync

    root, checkout, _, lock = disposable_sources
    asset = checkout / "assets/keep"
    asset.parent.mkdir()
    asset.write_text("new upstream file")
    git(checkout, "add", "-f", "assets/keep")
    git(checkout, "commit", "-m", "upstream introduces conflicting path")
    revision = git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "--detach", lock["sources"]["test"]["revision"])
    asset.parent.mkdir(exist_ok=True)
    asset.write_text("preserve acquired asset")
    lock["sources"]["test"]["revision"] = revision
    (root / "upstreams.lock.json").write_text(json.dumps(lock))
    with pytest.raises(subprocess.CalledProcessError):
        sync(root, ["test"], reset=True)
    assert asset.read_text() == "preserve acquired asset"


def test_packaged_root_reports_sources_from_the_build_receipt(tmp_path, monkeypatch):
    """A standalone image copies sources without git; the runtime checks them against the receipt
    the builder wrote from a verified checkout, and refuses a receipt made for another lock."""
    import roboenv.provisioning.sources as upstreams
    from roboenv.project import PACKAGED_RECEIPT, project_root

    patch = tmp_path / "third_party/patches/test.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("--- a\n+++ b\n")
    lock = {
        "schema": "roboenv/upstreams/1",
        "backends": ["test"],
        "sources": {
            "test": {
                "path": "third_party/test",
                "revision": "a" * 40,
                "url": "https://github.com/example/test",
                "backends": ["test"],
                "patches": [
                    {
                        "path": "third_party/patches/test.patch",
                        "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    (tmp_path / "upstreams.lock.json").write_text(json.dumps(lock))
    (tmp_path / "third_party/test").mkdir()
    verified = {
        "name": "test",
        "path": "third_party/test",
        "present": True,
        "expected": "a" * 40,
        "actual": "a" * 40,
        "pin_matches": True,
        "origin_matches": True,
        "patches_ready": True,
        "managed_tree_matches": True,
    }
    # The builder's checkout passes the live check (a git checkout in reality); the receipt
    # records it for the copy, which has no .git.
    with monkeypatch.context() as context:
        context.setattr(upstreams, "status", lambda *args: {"ready": True, "sources": [verified]})
        receipt = upstreams.write_receipt(tmp_path, ["test"], tmp_path / PACKAGED_RECEIPT)
    assert receipt["sources"]["test"]["managed_tree_matches"] is True
    with monkeypatch.context() as context:
        context.setattr(upstreams, "load_lock", lambda project: (tmp_path, lock))
        report = status(tmp_path)
        assert report["ready"] is True
        assert report["sources"][0]["packaged"] is True
        # A patch file that changed after the build is not what the receipt vouched for.
        patch.write_text("--- a\n+++ c\n")
        assert status(tmp_path)["ready"] is False
        patch.write_text("--- a\n+++ b\n")
        # A receipt made for another lock is no receipt.
        other = {**lock, "backends": ["test", "x"]}
        (tmp_path / "upstreams.lock.json").write_text(json.dumps(other))
        with pytest.raises(ValueError, match="does not match"):
            status(tmp_path)
    # The packaged root is a project: found through the runtime root a launch hands its processes
    # when no source project is at hand (a working directory outside any checkout).
    (tmp_path / "upstreams.lock.json").write_text(json.dumps(lock))
    monkeypatch.delenv("ROBOENV_PROJECT", raising=False)
    monkeypatch.setenv("ROBOENV_ROOT", str(tmp_path))
    monkeypatch.chdir("/")
    assert project_root() == tmp_path.resolve()


def test_receipt_is_refused_for_sources_that_are_not_ready(tmp_path, monkeypatch):
    import roboenv.provisioning.sources as upstreams

    monkeypatch.setattr(upstreams, "status", lambda *args: {"ready": False, "sources": []})
    with pytest.raises(ValueError, match="not ready"):
        upstreams.write_receipt(tmp_path, ["test"], tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()
