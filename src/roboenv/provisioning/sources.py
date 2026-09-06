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
    return subprocess.run(
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
    ).stdout.strip()


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


def sync(project=None, backends=None):
    root, lock = load_lock(project)
    if not (root / ".git").exists():
        git(root, "init", "--initial-branch=main")
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Refusing to register submodules in an unrelated parent repository")
    names = selected_sources(lock, backends or lock["backends"])
    for name in names:
        item = lock["sources"][name]
        checkout = root / item["path"]
        if not (checkout / ".git").exists():
            if item.get("parent"):
                parent = root / lock["sources"][item["parent"]]["path"]
                git(
                    parent,
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
        actual_url = git(checkout, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
        if actual_url != item["url"].removesuffix(".git").rstrip("/"):
            raise ValueError(f"origin mismatch for {name}")
        if git(checkout, "rev-parse", "HEAD") != item["revision"]:
            if git(
                checkout, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules"
            ):
                raise ValueError(f"refusing to change dirty upstream {name}")
            git(checkout, "fetch", "--depth", "1", "origin", item["revision"])
            git(checkout, "checkout", "--detach", item["revision"])
        for patch in item.get("patches", []):
            path = root / patch["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
                raise ValueError(f"patch checksum mismatch: {path.name}")
            try:
                git(checkout, "apply", "--reverse", "--check", str(path))
            except subprocess.CalledProcessError:
                git(checkout, "apply", "--check", str(path))
                git(checkout, "apply", str(path))
        if not item.get("parent"):
            git(root, "add", "--", item["path"])
    return status(root, backends)
