"""Pinned, selectively initialized Git submodules and verified managed patches."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from roboenv.project import project_root


def git(root, *args, timeout=600, extra_env=None):
    output = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            **(extra_env or {}),
        },
    ).stdout
    return output if "-z" in args else output.strip()


def load_lock(project=None):
    root = project_root(project)
    lock = json.loads((root / "upstreams.lock.json").read_text())
    if lock.get("schema") != "roboenv/upstreams/1":
        raise ValueError("unsupported upstream lock")
    for name, item in lock["sources"].items():
        path = Path(item["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] != "third_party"
            or not (root / path).resolve().is_relative_to(root)
        ):
            raise ValueError(f"invalid upstream path for {name}")
        if len(item["revision"]) != 40 or any(
            c not in "0123456789abcdef" for c in item["revision"]
        ):
            raise ValueError(f"unresolved revision for {name}")
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", item["url"]):
            raise ValueError("upstream URLs must be explicit public GitHub HTTPS URLs")
        for patch in item.get("patches", []):
            patch_path = Path(patch["path"])
            if (
                patch_path.is_absolute()
                or ".." in patch_path.parts
                or patch_path.parts[:2] != ("third_party", "patches")
                or not re.fullmatch(r"[0-9a-f]{64}", patch["sha256"])
                or not (root / patch_path).resolve().is_relative_to(root)
            ):
                raise ValueError(f"invalid managed patch for {name}")
        dependencies = item.get("requires", []) + ([item["parent"]] if item.get("parent") else [])
        if any(dependency not in lock["sources"] for dependency in dependencies):
            raise ValueError(f"unknown source dependency for {name}")
        if item.get("parent"):
            parent = Path(lock["sources"][item["parent"]]["path"])
            if not path.is_relative_to(parent) or path == parent:
                raise ValueError(f"invalid parent source for {name}")
    return root, lock


def selected_sources(lock, backends):
    backends = set(backends)
    known = {backend for item in lock["sources"].values() for backend in item["backends"]}
    if backends - known:
        raise ValueError(f"unknown backend selections: {sorted(backends - known)}")
    selected = {name for name, item in lock["sources"].items() if backends & set(item["backends"])}
    ordered = []

    def visit(name, active):
        if name in ordered:
            return
        if name in active:
            raise ValueError("upstream dependency cycle")
        item = lock["sources"][name]
        for dependency in item.get("requires", []) + (
            [item["parent"]] if item.get("parent") else []
        ):
            visit(dependency, active | {name})
        ordered.append(name)

    for name in sorted(selected):
        visit(name, set())
    return ordered


def managed_tree_matches(root, checkout, patches):
    """Compare files to HEAD plus locked patches without touching the user's index."""
    with tempfile.TemporaryDirectory(prefix="roboenv-source-index-") as temporary:
        environment = {"GIT_INDEX_FILE": str(Path(temporary) / "index")}
        try:
            git(checkout, "read-tree", "HEAD", extra_env=environment)
            for patch in patches:
                git(
                    checkout,
                    "apply",
                    "--cached",
                    str(root / patch["path"]),
                    extra_env=environment,
                )
            git(checkout, "diff", "--quiet", "--ignore-submodules", extra_env=environment)
            untracked = git(
                checkout, "ls-files", "--others", "--exclude-standard", extra_env=environment
            )
            return not untracked
        except subprocess.CalledProcessError:
            return False


def status(project=None, backends=None):
    root, lock = load_lock(project)
    names = selected_sources(lock, backends or lock["backends"])
    result = []
    for name in names:
        item = lock["sources"][name]
        checkout = root / item["path"]
        present = (checkout / ".git").exists()
        revision = git(checkout, "rev-parse", "HEAD") if present else None
        origin_matches = False
        patches_ready = True
        if present:
            origin = git(checkout, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
            origin_matches = origin == item["url"].rstrip("/").removesuffix(".git")
        for patch in item.get("patches", []):
            path = root / patch["path"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]
            ):
                patches_ready = False
                continue
            if present:
                try:
                    git(checkout, "apply", "--reverse", "--check", str(path))
                except subprocess.CalledProcessError:
                    patches_ready = False
            else:
                patches_ready = False
        result.append(
            {
                "name": name,
                "path": item["path"],
                "present": present,
                "expected": item["revision"],
                "actual": revision,
                "pin_matches": revision == item["revision"],
                "origin_matches": origin_matches,
                "patches_ready": patches_ready,
                "managed_tree_matches": (
                    present
                    and patches_ready
                    and managed_tree_matches(root, checkout, item.get("patches", []))
                ),
            }
        )
    return {
        "project": root.name,
        "sources": result,
        "ready": all(
            item["pin_matches"]
            and item["origin_matches"]
            and item["patches_ready"]
            and item["managed_tree_matches"]
            for item in result
        ),
    }


def check_updates(project=None, backends=None):
    """Compare pins to live upstream default-branch tips without changing checkouts."""
    root, lock = load_lock(project)
    results = []
    for name in selected_sources(lock, backends or lock["backends"]):
        item = lock["sources"][name]
        lines = git(root, "ls-remote", "--symref", item["url"], "HEAD", timeout=60).splitlines()
        revisions = [
            line.split()[0] for line in lines if re.fullmatch(r"[0-9a-f]{40}\s+HEAD", line)
        ]
        if len(revisions) != 1:
            raise ValueError(f"Cannot resolve upstream default-branch tip for {name}")
        branch = next(
            (line.split()[1] for line in lines if line.startswith("ref: ")),
            None,
        )
        results.append(
            {
                "name": name,
                "pinned": item["revision"],
                "upstream": revisions[0],
                "branch": branch,
                "up_to_date": item["revision"] == revisions[0],
            }
        )
    return {"sources": results, "up_to_date": all(item["up_to_date"] for item in results)}


def _initialize_source(root, lock, name):
    item = lock["sources"][name]
    checkout = root / item["path"]
    if not (checkout / ".git").exists():
        if item.get("parent"):
            parent = root / lock["sources"][item["parent"]]["path"]
            relative = str(checkout.relative_to(parent))
            registrations = git(parent, "config", "-f", ".gitmodules", "--get-regexp",
                                r"submodule\..*\.path").splitlines()
            keys = [line.split()[0].removesuffix(".path") for line in registrations
                    if line.split(maxsplit=1)[1] == relative]
            if len(keys) != 1:
                raise ValueError(f"Expected one parent submodule registration for {name}")
            git(
                parent,
                "-c", keys[0] + ".url=" + item["url"],
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
                "--",
                str(checkout.relative_to(parent)),
            )
        else:
            modules = root / ".gitmodules"
            registered = (
                modules.is_file()
                and item["path"]
                in git(
                    root, "config", "-f", ".gitmodules", "--get-regexp", r"submodule\..*\.path"
                ).split()
            )
            if registered:
                git(root, "submodule", "update", "--init", "--depth", "1", "--", item["path"])
            else:
                git(
                    root,
                    "submodule",
                    "add",
                    "--name",
                    name,
                    "--depth",
                    "1",
                    item["url"],
                    item["path"],
                )


def _check_patch_sequence(root, checkout, item):
    """Validate the entire target tree without changing the checkout or real index."""
    with tempfile.TemporaryDirectory(prefix="roboenv-patch-index-") as temporary:
        environment = {"GIT_INDEX_FILE": str(Path(temporary) / "index")}
        git(checkout, "read-tree", item["revision"], extra_env=environment)
        for patch in item.get("patches", []):
            git(checkout, "apply", "--cached", str(root / patch["path"]),
                extra_env=environment)


def _discard_source_edits(checkout, children):
    # Capture nonignored source files BEFORE restoring .gitignore: managed patches
    # may protect asset depots. Never clean ignored data or recurse into submodules.
    untracked = git(checkout, "ls-files", "--others", "--exclude-standard", "-z")
    for relative in filter(None, untracked.split("\0")):
        path = checkout / relative
        if any(path == child or child in path.parents for child in children):
            continue
        if path.is_symlink() and path.is_dir():
            continue  # external asset/depot link
        if path.is_file() or path.is_symlink():
            path.unlink()
    git(checkout, "reset", "--hard", "--no-recurse-submodules", "HEAD")


def sync(project=None, backends=None, *, reset=False):
    """Sync exact managed sources; prepare explicitly discards disposable source edits."""
    root, lock = load_lock(project)
    names = selected_sources(lock, backends or lock["backends"])
    # Check every selected patch before initialization or destructive refresh.
    for name in names:
        item = lock["sources"][name]
        for patch in item.get("patches", []):
            path = root / patch["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
                raise ValueError(f"patch checksum mismatch: {path.name}")
        checkout = root / item["path"]
        if (checkout / ".git").exists():
            origin = git(checkout, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
            if origin != item["url"].removesuffix(".git").rstrip("/"):
                raise ValueError(f"origin mismatch for {name}")
    if not (root / ".git").exists():
        git(root, "init", "--initial-branch=main")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Refusing to register submodules in an unrelated parent repository")
    for name in names:
        _initialize_source(root, lock, name)
        item = lock["sources"][name]
        checkout = root / item["path"]
        origin = git(checkout, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
        if origin != item["url"].removesuffix(".git").rstrip("/"):
            raise ValueError(f"origin mismatch for {name}")
        try:
            git(checkout, "cat-file", "-e", item["revision"] + "^{commit}")
        except subprocess.CalledProcessError:
            git(checkout, "fetch", "--depth", "1", "origin", item["revision"])
        if reset:
            _check_patch_sequence(root, checkout, item)
    # All target patch sequences are valid before any existing source edits are discarded.
    for name in names:
        item = lock["sources"][name]
        checkout = root / item["path"]
        if reset:
            children = [root / source["path"] for source in lock["sources"].values()
                        if source.get("parent") == name]
            _discard_source_edits(checkout, children)
        if git(checkout, "rev-parse", "HEAD") != item["revision"]:
            if git(
                checkout, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules"
            ):
                raise ValueError(f"refusing to change dirty upstream {name}")
            git(checkout, "checkout", "--detach", "--no-overwrite-ignore", item["revision"])
        for patch in item.get("patches", []):
            path = root / patch["path"]
            try:
                git(checkout, "apply", "--reverse", "--check", str(path))
            except subprocess.CalledProcessError:
                git(checkout, "apply", "--check", str(path))
                git(checkout, "apply", str(path))
        if not item.get("parent"):
            git(root, "add", "--", item["path"])
    report = status(root, backends)
    if reset and not report["ready"]:
        raise ValueError("Source refresh did not produce the exact managed trees")
    return report
