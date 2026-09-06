"""Publishing fails closed without legal decisions or trustworthy build contents."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_preflight", ROOT / "tools/release_preflight.py"
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


@pytest.fixture
def project(tmp_path):
    paths = [
        "pyproject.toml",
        "README.md",
        "src/roboenv/__init__.py",
        "runtime/pyproject.toml",
        "runtime/README.md",
        "runtime/src/roboenv_runtime/__init__.py",
        ".gitmodules",
        "upstreams.lock.json",
        "docs/source-notices.json",
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copytree(ROOT / "third_party/patches", tmp_path / "third_party/patches")
    return tmp_path


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def wheel_archive(members, *, corrupt=False, symlink=None, record="example-0.1.dist-info/RECORD"):
    rows = []
    for name, data in members.items():
        checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([name, "sha256=" + checksum, str(len(data))])
    rows.append([record, "", ""])
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        for name, data in members.items():
            if name == symlink:
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                package.writestr(member, data)
            else:
                package.writestr(name, data + (b"tampered" if corrupt else b""))
        package.writestr(record, text.getvalue())
    output.seek(0)
    return zipfile.ZipFile(output)


def build_fixture_wheels(project, directory):
    projects = PREFLIGHT.read_projects(project, lambda *args, **kwargs: None)
    directory.mkdir()
    for name, item in projects.items():
        module = name.replace("-", "_")
        prefix = f"{module}-{item['version']}.dist-info/"
        dependencies = "".join(f"Requires-Dist: {dep}\n" for dep in item["dependencies"])
        metadata = (
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {item['version']}\n"
            f"Requires-Python: {item['requires-python']}\n"
            f"Description-Content-Type: text/markdown\n{dependencies}\n"
            + "Release checker synthetic fixture documentation.\n"
            * 5
        )
        source = project / PREFLIGHT.PACKAGES[name] / "src" / module / "__init__.py"
        members = {
            prefix + "METADATA": metadata.encode(),
            prefix + "WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
            module + "/__init__.py": source.read_bytes(),
        }
        with wheel_archive(members, record=prefix + "RECORD") as package:
            (directory / f"{module}-{item['version']}-py3-none-any.whl").write_bytes(
                package.fp.getvalue()
            )
    return projects


def test_development_is_not_release_approval(project):
    result = PREFLIGHT.inspect(project, mode="development")
    assert result["passed"] is True
    assert result["packaging_ready"] is False
    assert {
        "license.decision_missing",
        "license.text_missing",
        "wheel.unverified",
        "sources.checkout_unverified",
    } <= codes(result)
    strict = PREFLIGHT.inspect(project)
    assert strict["passed"] is False
    assert str(project) not in json.dumps(strict)


def test_module_version_drift_fails_even_development(project):
    module = project / "src/roboenv/__init__.py"
    module.write_text('__version__ = "99.0.0"\n')
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert "metadata.module_version" in codes(result)


def test_runtime_must_depend_on_exact_sdk_server_extra(project):
    metadata = project / "runtime/pyproject.toml"
    metadata.write_text(metadata.read_text().replace("roboenv[server]==", "roboenv>="))
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert "metadata.sdk_dependency" in codes(result)


def test_source_notice_revision_and_patch_digest_are_enforced(project):
    inventory = project / "docs/source-notices.json"
    content = json.loads(inventory.read_text())
    content["sources"]["capx"]["revision"] = "0" * 40
    inventory.write_text(json.dumps(content))
    patch = project / "third_party/patches/capx-runtime.patch"
    patch.write_bytes(patch.read_bytes() + b"\n")
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert {"notices.revision", "sources.patch_checksum"} <= codes(result)


def test_secret_report_never_echoes_secret_contents(project):
    token = "ghp_" + "a" * 36
    (project / "leak.txt").write_text(token)
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert "source.github_token" in codes(result)
    assert token not in json.dumps(result)


@pytest.mark.parametrize("value", ["../escape", "/outside/file", r"..\escape", ""])
def test_artifact_path_escape_is_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        PREFLIGHT.local_path(tmp_path, value)


def test_artifact_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "out").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        PREFLIGHT.local_path(root, "out/private")


def test_record_checks_each_wheel_member():
    with wheel_archive({"example/__init__.py": b"ok"}) as package:
        PREFLIGHT.check_record(package)
    with wheel_archive({"example/__init__.py": b"ok"}, corrupt=True) as package:
        with pytest.raises(ValueError, match="checksum"):
            PREFLIGHT.check_record(package)


def test_record_rejects_escape_and_symlink():
    with wheel_archive({"../escape": b"ok"}) as package:
        with pytest.raises(ValueError, match="unsafe"):
            PREFLIGHT.check_record(package)
    with wheel_archive({"example/link": b"target"}, symlink="example/link") as package:
        with pytest.raises(ValueError, match="symlink"):
            PREFLIGHT.check_record(package)


def test_malformed_input_becomes_sanitized_failure(project):
    (project / "upstreams.lock.json").write_text("invalid json")
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert "preflight.invalid_input" in codes(result)
    assert str(project) not in json.dumps(result)


def test_cli_writes_machine_readable_blockers(project, capsys):
    report = project / "build/preflight.json"
    assert (
        PREFLIGHT.main(
            [
                "--project",
                str(project),
                "--mode",
                "development",
                "--output",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text())["packaging_ready"] is False
    assert str(project) not in capsys.readouterr().out


def test_actual_wheel_structure_and_sources_are_checked(project):
    directory = project / "wheels"
    projects = build_fixture_wheels(project, directory)
    issues = []
    artifacts = PREFLIGHT.check_wheels(
        project, directory, projects, lambda code, *args, **kwargs: issues.append(code)
    )
    assert not issues
    assert len(artifacts) == 2
    module = project / "src/roboenv/__init__.py"
    module.write_text(module.read_text() + "\n# Changed after the candidate was built.\n")
    (module.parent / "missing.py").write_text('"""New unbuilt module."""\n')
    PREFLIGHT.check_wheels(
        project, directory, projects, lambda code, *args, **kwargs: issues.append(code)
    )
    assert {"wheel.source_mismatch", "wheel.source_missing"} <= set(issues)


def test_missing_package_wheel_blocks_even_development(project):
    wheels = project / "build/wheels"
    wheels.mkdir(parents=True)
    result = PREFLIGHT.inspect(project, wheels=wheels, mode="development")
    assert not result["passed"]
    assert "wheel.missing" in codes(result)


def test_secret_in_filename_is_reported_without_echoing_the_name(project):
    token = "ghp_" + "b" * 36
    (project / token).write_text("content")
    result = PREFLIGHT.inspect(project, mode="development")
    assert not result["passed"]
    assert token not in json.dumps(result)
