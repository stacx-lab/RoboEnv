"""Fail-closed packaging checks; development mode reports outstanding release blockers.

Use the Python 3.12 development environment. This never publishes, signs, uploads,
changes Git state, or chooses a project license.
"""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import stat
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import tomllib
from packaging.licenses import canonicalize_license_expression
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from roboenv.privacy import audit, owned_files, validate_public_text
from roboenv.provisioning.sources import load_lock, status

PACKAGES = {"roboenv": ".", "roboenv-runtime": "runtime"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "cloud_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential_url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def local_path(root, value):
    """Resolve a declared relative artifact without following an escaping symlink."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError("unsafe relative artifact path")
    candidate = root.joinpath(*path.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes its declared root")
    return candidate


def secret_findings(content, label):
    return [
        {"file": label, "kind": code}
        for code, pattern in SECRET_PATTERNS.items()
        if pattern.search(content)
    ]


def safe_subject(value):
    label = str(value)
    for pattern in SECRET_PATTERNS.values():
        label = pattern.sub("REDACTED", label)
    try:
        validate_public_text(label, "")
    except ValueError:
        return "REDACTED"
    return label


def scan_source(root):
    findings = audit(root)["findings"]
    for path in owned_files(root):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            content = path.read_text()
        except UnicodeError:
            continue
        label = path.relative_to(root).as_posix()
        findings.extend(secret_findings(content + "\n" + label, safe_subject(label)))
    return findings


def read_projects(root, report):
    projects = {}
    for expected_name, folder in PACKAGES.items():
        directory = root / folder
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        projects[expected_name] = project
        if project["name"] != expected_name:
            report("metadata.name", expected_name)
        Version(project["version"])
        readme = project.get("readme")
        if not isinstance(readme, str) or not local_path(directory, readme).is_file():
            report("metadata.readme", expected_name)
        package = expected_name.replace("-", "_")
        tree = ast.parse((directory / "src" / package / "__init__.py").read_text())
        versions = [
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        ]
        if versions != [project["version"]]:
            report("metadata.module_version", expected_name)
        license_file = directory / "LICENSE"
        expression = project.get("license")
        if not isinstance(expression, str) or not expression.strip():
            report("license.decision_missing", expected_name, blocker=True)
        else:
            try:
                canonicalize_license_expression(expression)
            except ValueError:
                report("license.invalid_spdx", expected_name)
            if "LicenseRef-" in expression:
                report("license.custom_review_required", expected_name, blocker=True)
        if not license_file.is_file() or len(license_file.read_bytes()) < 100:
            report("license.text_missing", expected_name, blocker=True)
        elif re.search(r"\b(?:TODO|TBD|CHOOSE A LICENSE)\b", license_file.read_text(), re.I):
            report("license.placeholder", expected_name, blocker=True)
    if len({project["version"] for project in projects.values()}) != 1:
        report("metadata.version_mismatch", "package versions must match")
    dependencies = [
        Requirement(value) for value in projects["roboenv-runtime"].get("dependencies", [])
    ]
    sdk = [value for value in dependencies if canonicalize_name(value.name) == "roboenv"]
    if (
        len(sdk) != 1
        or str(sdk[0].specifier) != "==" + projects["roboenv"]["version"]
        or sdk[0].marker is not None
        or sdk[0].url is not None
        or sdk[0].extras != {"server"}
    ):
        report("metadata.sdk_dependency", "runtime must require the matching SDK server extra")
    if (root / "LICENSE").is_file() and (root / "runtime/LICENSE").is_file():
        if (root / "LICENSE").read_bytes() != (root / "runtime/LICENSE").read_bytes():
            report("license.package_mismatch", "package license texts must match")
    return projects


def check_sources(root, report, verify_upstreams):
    _, lock = load_lock(root)
    notices = json.loads((root / "docs/source-notices.json").read_text())
    if notices.get("schema") != "roboenv/source-notices/1":
        report("notices.schema", "source notice inventory")
    if set(notices["sources"]) != set(lock["sources"]):
        report("notices.coverage", "inventory must cover every locked source")
    modules = configparser.ConfigParser()
    modules.read(root / ".gitmodules")
    registered = {
        modules[section]["path"]: modules[section]["url"] for section in modules.sections()
    }
    for name, source in lock["sources"].items():
        if not source.get("parent") and registered.get(source["path"]) != source["url"]:
            report("sources.registration", name)
        for patch in source.get("patches", []):
            path = local_path(root, patch["path"])
            if not path.is_file() or digest(path.read_bytes()) != patch["sha256"]:
                report("sources.patch_checksum", name)
        item = notices["sources"].get(name, {})
        if item.get("revision") != source["revision"]:
            report("notices.revision", name)
        if not item.get("notices"):
            report("notices.empty", name)
        for notice in item.get("notices", []):
            path = local_path(root / source["path"], notice["path"])
            if not re.fullmatch(r"[0-9a-f]{64}", notice["sha256"]):
                report("notices.checksum", name)
            if path.exists() and digest(path.read_bytes()) != notice["sha256"]:
                report("notices.changed", name)
            if verify_upstreams and not path.is_file():
                report("notices.unavailable", name)
    if verify_upstreams:
        if not status(root)["ready"]:
            report("sources.checkout", "source pins, origins, and patches must match")
    else:
        report("sources.checkout_unverified", "rerun with --verify-upstreams", blocker=True)


def check_record(package):
    """Validate every member, not just the archive-level checksum."""
    names = package.namelist()
    if len(set(names)) != len(names):
        raise ValueError("duplicate archive member")
    records = [name for name in names if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("wheel must contain exactly one RECORD")
    rows = list(csv.reader(io.StringIO(package.read(records[0]).decode())))
    if any(len(row) != 3 for row in rows):
        raise ValueError("malformed wheel RECORD")
    if len({row[0] for row in rows}) != len(rows) or {row[0] for row in rows} != set(names):
        raise ValueError("RECORD does not enumerate the wheel exactly")
    for name, checksum, size in rows:
        path = PurePosixPath(name)
        info = package.getinfo(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe wheel member")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("symlink in wheel")
        if name == records[0]:
            if checksum or size:
                raise ValueError("RECORD must not hash itself")
            continue
        data = package.read(name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if checksum != "sha256=" + encoded or size != str(len(data)):
            raise ValueError("wheel RECORD checksum mismatch")


def check_wheels(root, directory, projects, report):
    artifacts = []
    found = set()
    for wheel in sorted(directory.glob("*.whl")):
        with zipfile.ZipFile(wheel) as package:
            check_record(package)
            filename_name, filename_version, _, _ = parse_wheel_filename(wheel.name)
            metadata_files = [
                name for name in package.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError("wheel must contain exactly one METADATA")
            metadata = BytesParser().parsebytes(package.read(metadata_files[0]))
            name = canonicalize_name(metadata["Name"])
            if name not in projects or name in found:
                report("wheel.unexpected", wheel.name)
                continue
            found.add(name)
            project = projects[name]
            expected_metadata = (
                name.replace("-", "_") + "-" + project["version"] + ".dist-info/METADATA"
            )
            if metadata_files[0] != expected_metadata:
                report("wheel.metadata_location", name)
            if filename_name != name or str(filename_version) != metadata["Version"]:
                report("wheel.filename", name)
            if metadata["Version"] != project["version"]:
                report("wheel.version", name)
            wheel_metadata = metadata_files[0].removesuffix("METADATA") + "WHEEL"
            wheel_info = BytesParser().parsebytes(package.read(wheel_metadata))
            if (
                wheel_info["Wheel-Version"] != "1.0"
                or wheel_info["Root-Is-Purelib"] != "true"
                or wheel_info.get_all("Tag") != ["py3-none-any"]
            ):
                report("wheel.format", name)
            if metadata["Requires-Python"] != project["requires-python"]:
                report("wheel.python", name)
            declared = {str(Requirement(value)) for value in project.get("dependencies", [])}
            actual = {
                str(Requirement(value))
                for value in metadata.get_all("Requires-Dist", [])
                if "extra ==" not in value
            }
            if actual != declared:
                report("wheel.dependencies", name)
            if (
                metadata["Description-Content-Type"] != "text/markdown"
                or len(metadata.get_payload().strip()) < 100
            ):
                report("wheel.description", name)
            if project.get("license") and metadata["License-Expression"] != project["license"]:
                report("wheel.license_expression", name)
            license_file = root / PACKAGES[name] / "LICENSE"
            if license_file.is_file() and not any(
                member.endswith("/licenses/LICENSE")
                and package.read(member) == license_file.read_bytes()
                for member in package.namelist()
            ):
                report("wheel.license_text", name, blocker=True)
            for member in package.namelist():
                if ".dist-info/" in member and not member.startswith(
                    expected_metadata.removesuffix("METADATA")
                ):
                    report("wheel.metadata_location", name)
                if any(
                    part in {"tests", "third_party", ".env", "__pycache__"}
                    for part in PurePosixPath(member).parts
                ) or member.endswith("solution.py"):
                    report("wheel.private_payload", name)
                data = package.read(member)
                if ".dist-info/" not in member:
                    source = local_path(root / PACKAGES[name] / "src", member)
                    if not source.is_file() or source.read_bytes() != data:
                        report("wheel.source_mismatch", name)
                try:
                    content = data.decode()
                except UnicodeError:
                    continue
                try:
                    validate_public_text(content, member)
                except ValueError:
                    report("wheel.privacy", name)
                if secret_findings(content, member):
                    report("wheel.secret", name)
            for source in (root / PACKAGES[name] / "src").rglob("*.py"):
                member = source.relative_to(root / PACKAGES[name] / "src").as_posix()
                if "__pycache__" not in source.parts and member not in package.namelist():
                    report("wheel.source_missing", name)
        artifacts.append({"name": wheel.name, "sha256": digest(wheel.read_bytes())})
    if found != set(projects):
        report("wheel.missing", "build exactly one SDK and one runtime wheel")
    return artifacts


def inspect(root, *, wheels=None, verify_upstreams=False, mode="release"):
    issues = []

    def report(code, subject, *, blocker=False):
        issues.append(
            {
                "code": code,
                "subject": safe_subject(subject),
                "severity": "release_blocker" if blocker else "error",
            }
        )

    artifacts = []
    try:
        projects = read_projects(root, report)
        check_sources(root, report, verify_upstreams)
        for finding in scan_source(root):
            report("source." + finding["kind"], finding["file"])
        if wheels is None:
            report("wheel.unverified", "provide --wheels", blocker=True)
        else:
            artifacts = check_wheels(root, wheels, projects, report)
    except Exception as error:
        # Do not print exceptions containing private filesystem locations or data.
        report("preflight.invalid_input", type(error).__name__)
    errors = [issue for issue in issues if issue["severity"] == "error"]
    preview_blockers = [
        issue
        for issue in issues
        if issue["severity"] == "release_blocker"
        and not issue["code"].startswith(("license.", "wheel."))
    ]
    passed = not issues if mode == "release" else not errors
    if mode == "source-preview":
        passed = not errors and not preview_blockers
    return {
        "schema": "roboenv/release-preflight/1",
        "mode": mode,
        "passed": passed,
        "packaging_ready": not issues,
        "private_source_preview_ready": passed if mode == "source-preview" else False,
        "scope": "packaging and source prerequisites; not GPU, security, or release approval",
        "issues": issues,
        "artifacts": artifacts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--mode", choices=("development", "source-preview", "release"), default="release"
    )
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--verify-upstreams", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect(
        args.project.resolve(),
        wheels=args.wheels,
        verify_upstreams=args.verify_upstreams,
        mode=args.mode,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
