"""Synchronize checked-in package metadata from VERSION; never build or publish."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {"roboenv": "", "roboenv-runtime": "runtime"}


def read_version(root):
    value = (root / "VERSION").read_text().strip()
    if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
        raise ValueError("VERSION must contain MAJOR.MINOR.PATCH")
    return value


def version_updates(root):
    version = read_version(root)
    updates, drift = {}, []

    def replace(text, pattern, value, label):
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if len(matches) != 1:
            raise ValueError(f"{label}: expected exactly one version field")
        match = matches[0]
        if match.group(1) != value:
            drift.append(f"{label} must be {value}")
        return text[:match.start(1)] + value + text[match.end(1):]

    for name, directory in PROJECTS.items():
        manifest = Path(directory) / "pyproject.toml"
        text = (root / manifest).read_text()
        project = tomllib.loads(text)["project"]
        if project["name"] != name:
            raise ValueError(f"{manifest}: unexpected project name")
        section = re.search(r"(?ms)^\[project\][ \t]*\n(.*?)(?=^\[|\Z)", text)
        if section is None:
            raise ValueError(f"{manifest}: missing project section")
        content = replace(section.group(1), r'^version\s*=\s*[\"\']([^\"\']*)[\"\']',
                          version, f"{manifest}: project.version")
        if name == "roboenv-runtime":
            content = replace(content, r'[\"\']roboenv\[server\]==([^\"\']*)[\"\']',
                              version, f"{manifest}: dependency roboenv[server]")
        updates[manifest] = text[:section.start(1)] + content + text[section.end(1):]
        module = Path(directory) / "src" / name.replace("-", "_") / "__init__.py"
        updates[module] = replace(
            (root / module).read_text(), r'^__version__\s*=\s*[\"\']([^\"\']*)[\"\']',
            version, f"{module}: __version__",
        )
    return updates, drift


def check_version(root=ROOT):
    _, drift = version_updates(root)
    if drift:
        raise ValueError("\n".join([*drift, "Run python tools/version.py --sync"]))


def sync_version(root=ROOT):
    updates, _ = version_updates(root)
    changed = []
    for relative, content in updates.items():
        path = root / relative
        if path.read_text() != content:
            path.write_text(content)
            changed.append(relative.as_posix())
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--sync", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.sync:
        for path in sync_version():
            print(f"Updated {path}")
    else:
        check_version()
        print(f"Versions match {read_version(ROOT)}")


if __name__ == "__main__":
    main()
