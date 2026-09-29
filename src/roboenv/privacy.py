"""Audit publishable files for development-machine identifiers and private paths."""

import os
import re
import socket
from pathlib import Path

SKIP = {
    ".git",
    ".cache",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "generated",
    "jobs",
}
HOST_ROOTS = ("localscratch", "nethome", "scratch")
LEGACY_MARKER = "".join(("one", "world"))


def _patterns():
    roots = "|".join(HOST_ROOTS)
    result = [
        ("host_path", re.compile(r"/(?:" + roots + r")/[^\s\"']*")),
        ("user_home", re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")),
        ("legacy_identifier", re.compile(re.escape(LEGACY_MARKER), re.I)),
    ]
    for value in {socket.gethostname(), os.environ.get("USER", "")}:
        if value and value not in {"root", "user", "runner"}:
            result.append(("local_identity", re.compile(r"\b" + re.escape(value) + r"\b")))
    return result


def validate_public_text(content, label):
    """Reject known private identifiers without echoing the matching value."""
    for code, pattern in _patterns():
        if pattern.search(content) or pattern.search(label):
            raise ValueError(f"publishable content rejected by privacy check: {code}")


def audit(root):
    root = Path(root).resolve()
    findings = []
    patterns = _patterns()
    for path in owned_files(root):
        relative = path.relative_to(root)
        if any(
            part in SKIP or part.startswith(".venv") or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.parts[0] == "third_party" and relative.parts[1:2] != ("patches",):
            continue  # upstream-owned source is not rewritten or republished as our own
        if not path.is_file() or path.is_symlink():
            continue
        try:
            content = path.read_text()
        except UnicodeError:
            continue
        for code, pattern in patterns:
            if pattern.search(str(relative)) or pattern.search(content):
                safe = str(relative)
                for _, secret_pattern in patterns:
                    safe = secret_pattern.sub("REDACTED", safe)
                findings.append({"file": safe, "kind": code})
    return {
        "passed": not findings,
        "findings": findings,
        "scope": "owned-source; excludes git metadata, vendor checkouts and local artifacts",
    }


def owned_files(root):
    """Prune external trees before walking them; do not traverse every simulator asset."""
    for directory, children, names in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        children[:] = sorted(
            name
            for name in children
            if name not in SKIP
            and not name.startswith(".venv")
            and not name.endswith(".egg-info")
            and not (relative.parts == ("third_party",) and name != "patches")
            and not (relative.parts == ("benchmarks",) and name == "manipulation")
            and not (relative.parts == ("deployments",) and name == "local")
        )
        for name in sorted(names):
            if name not in SKIP:
                yield Path(directory) / name
