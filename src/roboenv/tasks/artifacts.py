"""Conflict-safe, content-verified task and catalog artifacts."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml
from roboenv.privacy import validate_public_text
from roboenv.validation import json_loads, json_value


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Bundle:
    """Preflight all writes, preserve unrelated files and refuse manual-edit conflicts."""

    def __init__(self, root):
        self.root = Path(root)
        self.contents = {}

    def add(self, name, content, mode=0o644):
        if (
            not isinstance(name, str)
            or not name
            or name in {".", "bundle.json"}
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or "\\" in name
            or Path(name).as_posix() != name
        ):
            raise ValueError("bundle entries must be safe relative paths")
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise ValueError("bundle permissions must be ordinary file mode bits")
        if isinstance(content, str):
            validate_public_text(content, name)
        elif not name.endswith(".whl"):
            validate_public_text(content.decode("utf-8"), name)
        self.contents[name] = (content.encode() if isinstance(content, str) else content, mode)

    def add_yaml(self, name, content):
        self.add(name, yaml.safe_dump(content, sort_keys=False))

    def preflight(self):
        manifest = self.root / "bundle.json"
        if manifest.is_symlink():
            raise ValueError("bundle manifest must not be a symlink")
        if manifest.exists() and not manifest.is_file():
            raise ValueError("bundle manifest must be a regular file")
        saved = json_loads(manifest.read_text()) if manifest.exists() else {}
        previous = saved.get("files", {}) if isinstance(saved, dict) else None
        if not isinstance(previous, dict) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in previous.values()
        ):
            raise ValueError("bundle manifest must contain file SHA-256 hashes")
        if set(previous) - set(self.contents):
            raise ValueError(
                "generation would leave stale managed files; use a fresh output directory"
            )
        for name, (data, _) in self.contents.items():
            path = self.root / name
            if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError("bundle entry escapes output directory")
            for parent in path.parents:
                if parent == self.root:
                    break
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise ValueError("bundle output ancestors must be real directories")
            if path.exists() and not path.is_file():
                raise ValueError("bundle output entries must be regular files")
            if (
                path.exists()
                and path.read_bytes() != data
                and sha(path.read_bytes()) != previous.get(name)
            ):
                raise ValueError(f"refusing to overwrite edited/unmanaged output: {name}")
        return previous

    def commit(self, metadata):
        json_value(metadata)
        validate_public_text(json.dumps(metadata), "bundle.json")
        self.preflight()
        for name, (data, mode) in self.contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            self._replace(path, data, mode)
        metadata = {
            **metadata,
            "files": {name: sha(data) for name, (data, _) in self.contents.items()},
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._replace(
            self.root / "bundle.json",
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
            0o644,
        )
        return metadata

    @staticmethod
    def _replace(path, data, mode):
        # Replacing a directory entry never truncates an unrelated hard-linked
        # file. The manifest is published last; partial writes remain detectable.
        descriptor, temporary = tempfile.mkstemp(prefix=".roboenv-write-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), mode)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
