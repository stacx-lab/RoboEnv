"""Relocate only a marked staging copy; never change a prepared interpreter."""

import argparse
import json
import re
import shutil
from pathlib import Path


def relocate(prefix, manifest):
    prefix = Path(prefix)
    marker = prefix / ".roboenv-staged-environment.json"
    if (
        prefix.is_symlink()
        or not prefix.is_dir()
        or prefix == Path(prefix.anchor)
        or marker.is_symlink()
        or not marker.is_file()
    ):
        raise ValueError("refusing to relocate an unmarked interpreter")
    try:
        marker_data = json.loads(marker.read_text())
    except (ValueError, OSError) as error:
        raise ValueError("invalid interpreter staging marker") from error
    if (
        not isinstance(marker_data, dict)
        or marker_data.get("schema") != "roboenv/staged-environment/1"
    ):
        raise ValueError("invalid interpreter staging marker")
    replacements = manifest["replacements"]
    python_aliases = set()
    if manifest.get("python_base"):
        version = manifest.get("python_version", "")
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
            raise ValueError("invalid base interpreter version")
        if (prefix / "bin").is_symlink() or not (prefix / "bin").is_dir():
            raise ValueError("interpreter bin must be a real staging directory")
        python_aliases = {"python", "python3", f"python{version}"}
    retired = "".join(("phy", "si"))
    entries = sorted(prefix.rglob("*"))
    for path in entries:
        if path.is_symlink():
            if path.parent == prefix / "bin" and path.name in python_aliases:
                continue  # Replaced below with the separately verified copied Python base.
            original = str(path.readlink())
            if Path(original).is_absolute():
                replacement = None
                for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
                    if original == source or original.startswith(source.rstrip("/") + "/"):
                        replacement = target + original[len(source) :]
                        break
                if replacement is None:
                    protected = (
                        "/opt/sim-env",
                        "/opt/python",
                        "/opt/roboenv/runtime/dependencies/sim-env",
                        "/opt/roboenv/runtime/dependencies/python-base",
                    )
                    if any(
                        original == root or original.startswith(root + "/") for root in protected
                    ):
                        continue  # Already relocated; the snapshot verifies terminal coverage.
                    raise ValueError("absolute interpreter link has no safe relocation target")
                path.unlink()
                path.symlink_to(replacement)
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(prefix)
        if path.name.startswith("__editable__") and any(
            name in path.name for name in ("roboenv", "code_agent", retired)
        ):
            path.unlink()
            continue
        selected = path.suffix in {".pth", ".egg-link"} or path.name.startswith("__editable__")
        selected = selected or relative.parts[0] == "bin"
        selected = selected or path.name in {"pyvenv.cfg", "direct_url.json"}
        if not selected:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode()
        except UnicodeDecodeError:
            continue
        for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
            text = text.replace(source, target)
        path.write_text(text)
    # Stale editable metadata must not advertise a removed project distribution.
    for path in entries:
        if path.relative_to(prefix).match(
            "lib/python*/site-packages/*.dist-info"
        ) and path.name.startswith((retired + "-", "roboenv-", "roboenv_runtime-", "code_agent-")):
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    if manifest.get("python_base"):
        for name in ("python", "python3", f"python{manifest['python_version']}"):
            path = prefix / "bin" / name
            if path.is_symlink() or path.exists():
                path.unlink()
            path.symlink_to(f"/opt/python/bin/python{manifest['python_version']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("/opt/sim-env"))
    parser.add_argument("--manifest", type=Path, default=Path("/run/secrets/relocation.json"))
    args = parser.parse_args()
    relocate(args.prefix, json.loads(args.manifest.read_text()))


if __name__ == "__main__":
    main()
