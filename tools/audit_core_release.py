"""Audit installed candidate SDK/runtime dependencies and emit a validated core SBOM.

Only public dependency names/versions are sent to the PyPI advisory service.
Install hash-locked tools separately; no automatic remediation or upload occurs.
"""

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from roboenv.privacy import validate_public_text

ROOTS = {"roboenv", "roboenv-runtime"}
TOOLS = {"pip-audit": "2.10.1", "cyclonedx-bom": "7.3.1"}


def run(command, *, timeout=600):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def inventory(python):
    code = (
        "import importlib.metadata as m,json,sys;"
        "print(json.dumps({'python':list(sys.version_info[:2]),'packages':["
        "{'name':d.metadata['Name'],'version':d.version,'requires':d.requires or []}"
        " for d in m.distributions()]}))"
    )
    process = run([str(python), "-I", "-c", code])
    if process.returncode:
        raise ValueError("environment_inventory_failed")
    result = json.loads(process.stdout)
    if result["python"] != [3, 12]:
        raise ValueError("audit_requires_python_3_12")
    return {canonicalize_name(item["name"]): item for item in result["packages"]}


def dependency_closure(packages):
    """Reject dev tools or unrelated installed packages from a claimed core audit."""
    pending = [(name, "") for name in ROOTS]
    visited = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in visited:
            continue
        if name not in packages:
            raise ValueError("core_dependency_missing")
        visited.add((name, extra))
        for value in packages[name]["requires"]:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate(
                {
                    **default_environment(),
                    "extra": extra,
                }
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if requirement.url or dependency not in packages:
                raise ValueError("core_dependency_unresolved")
            if packages[dependency]["version"] not in requirement.specifier:
                raise ValueError("core_dependency_version_mismatch")
            pending.extend((dependency, item) for item in ("", *requirement.extras))
    names = {name for name, _ in visited}
    if names != set(packages):
        raise ValueError("environment_contains_non_core_packages")
    return sorted(names - ROOTS)


def wheel_identities(directory, packages, python):
    artifacts = {}
    for wheel in directory.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            member = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = BytesParser().parsebytes(archive.read(member))
            expected = {
                entry: hashlib.sha256(archive.read(entry)).hexdigest()
                for entry in archive.namelist()
                if not entry.endswith(".dist-info/RECORD")
            }
        name = canonicalize_name(metadata["Name"])
        if name not in ROOTS or name in artifacts:
            raise ValueError("unexpected_core_wheel")
        if metadata["Version"] != packages[name]["version"]:
            raise ValueError("installed_candidate_version_mismatch")
        verification = run(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata as m,json,hashlib,sys;"
                "d=m.distribution(sys.argv[1]);"
                "print(all(hashlib.sha256(d.locate_file(p).read_bytes()).hexdigest()==h"
                " for p,h in json.loads(sys.argv[2]).items()))",
                name,
                json.dumps(expected),
            ]
        )
        if verification.returncode or verification.stdout.strip() != "True":
            raise ValueError("installed_candidate_payload_mismatch")
        artifacts[name] = {
            "name": wheel.name,
            "version": metadata["Version"],
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    if set(artifacts) != ROOTS:
        raise ValueError("candidate_wheel_missing")
    return artifacts


def sanitize_bom(bom, artifacts):
    """Replace only local candidate URLs with actual immutable wheel identities."""
    for component in bom.get("components", []):
        name = canonicalize_name(component["name"])
        if name not in artifacts:
            continue
        references = component.get("externalReferences", [])
        component["externalReferences"] = [
            reference for reference in references if not reference["url"].startswith("file:")
        ]
        component["hashes"] = [{"alg": "SHA-256", "content": artifacts[name]["sha256"]}]
        component.setdefault("properties", []).append(
            {
                "name": "roboenv:wheel",
                "value": artifacts[name]["name"],
            }
        )
    validate_public_text(json.dumps(bom), "core.cdx.json")
    return bom


def audit(packages, names, artifacts, tool_python, temporary, out):
    requirements = "\n".join(f"{name}=={packages[name]['version']}" for name in names) + "\n"
    requirement_file = temporary / "requirements.txt"
    requirement_file.write_text(requirements)
    raw_audit = temporary / "audit.json"
    process = run(
        [
            str(tool_python),
            "-I",
            "-m",
            "pip_audit",
            "-r",
            str(requirement_file),
            "--strict",
            "--disable-pip",
            "--no-deps",
            "--vulnerability-service",
            "pypi",
            "--timeout",
            "30",
            "--progress-spinner",
            "off",
            "--desc",
            "off",
            "--format",
            "json",
            "--output",
            str(raw_audit),
            "--cache-dir",
            str(temporary / "advisory-cache"),
        ]
    )
    if process.returncode not in (0, 1) or not raw_audit.is_file():
        raise ValueError("advisory_query_failed")
    result = json.loads(raw_audit.read_text())
    audited = result.get("dependencies", [])
    if {canonicalize_name(item["name"]) for item in audited} != set(names):
        raise ValueError("advisory_coverage_incomplete")
    if any(item.get("skip_reason") or "vulns" not in item for item in audited):
        raise ValueError("advisory_package_skipped")
    findings = [
        {
            "name": canonicalize_name(item["name"]),
            "version": item["version"],
            "vulnerabilities": [
                {
                    "id": vulnerability["id"],
                    "fix_versions": vulnerability.get("fix_versions", []),
                }
                for vulnerability in item["vulns"]
            ],
        }
        for item in audited
    ]
    vulnerability_count = sum(len(item["vulnerabilities"]) for item in findings)
    if process.returncode and not vulnerability_count:
        raise ValueError("advisory_query_failed")
    (out / "core-requirements.txt").write_text(requirements)
    return {
        "schema": "roboenv/core-audit/1",
        "passed": vulnerability_count == 0,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator": {"name": "pip-audit", "version": TOOLS["pip-audit"], "service": "pypi"},
        "scope": "candidate SDK/runtime and server dependencies; excludes simulators and agents",
        "local_packages_not_queried": sorted(ROOTS),
        "candidate_wheels": artifacts,
        "audited_dependencies": findings,
        "vulnerability_count": vulnerability_count,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path, help="Minimal candidate environment")
    parser.add_argument(
        "--tool-python", required=True, type=Path, help="Hash-locked tool environment"
    )
    parser.add_argument("--wheels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        packages = inventory(args.python)
        tool_packages = inventory(args.tool_python)
        if any(tool_packages[name]["version"] != version for name, version in TOOLS.items()):
            raise ValueError("audit_tool_version_mismatch")
        tool_lock = Path(__file__).resolve().parents[1] / "locks/release-tools.txt"
        locked_tools = {
            canonicalize_name(name): version
            for name, version in re.findall(
                r"^([A-Za-z0-9_.-]+)==([^\s]+)", tool_lock.read_text(), re.M
            )
        }
        if {name: item["version"] for name, item in tool_packages.items()} != locked_tools:
            raise ValueError("audit_tool_environment_does_not_match_lock")
        names = dependency_closure(packages)
        artifacts = wheel_identities(args.wheels, packages, args.python)
        with tempfile.TemporaryDirectory(prefix="core-audit-") as directory:
            temporary = Path(directory)
            result = audit(packages, names, artifacts, args.tool_python, temporary, args.out)
            result["tool_lock_sha256"] = hashlib.sha256(tool_lock.read_bytes()).hexdigest()
            raw_bom = temporary / "core.raw.cdx.json"
            process = run(
                [
                    str(args.tool_python),
                    "-I",
                    "-m",
                    "cyclonedx_py",
                    "environment",
                    str(args.python.absolute()),
                    "--sv",
                    "1.6",
                    "--short-PURLs",
                    "--output-reproducible",
                    "--validate",
                    "--of",
                    "JSON",
                    "-o",
                    str(raw_bom),
                ]
            )
            if process.returncode or not raw_bom.is_file():
                raise ValueError("sbom_generation_failed")
            bom = sanitize_bom(json.loads(raw_bom.read_text()), artifacts)
            if len(bom.get("components", [])) != len(packages):
                raise ValueError("sbom_component_coverage_incomplete")
            bom_file = temporary / "core.cdx.json"
            bom_file.write_text(json.dumps(bom, indent=2) + "\n")
            validation = run(
                [
                    str(args.tool_python),
                    "-I",
                    "-c",
                    "from pathlib import Path; import sys;"
                    "from cyclonedx.validation.json import JsonStrictValidator;"
                    "from cyclonedx.schema import SchemaVersion;"
                    "assert JsonStrictValidator(SchemaVersion.V1_6).validate_str("
                    "Path(sys.argv[1]).read_text(),all_errors=True) is None",
                    str(bom_file),
                ]
            )
            if validation.returncode:
                raise ValueError("sbom_schema_validation_failed")
            (args.out / "core.cdx.json").write_bytes(bom_file.read_bytes())
            result["sbom"] = {
                "name": "core.cdx.json",
                "sha256": hashlib.sha256(bom_file.read_bytes()).hexdigest(),
                "format": "CycloneDX 1.6",
                "validated": True,
                "generator": {"name": "cyclonedx-bom", "version": TOOLS["cyclonedx-bom"]},
                "components": len(packages),
            }
    except Exception as error:
        # Raw tool output may contain installation paths; never publish it.
        reason = str(error)
        result = {
            "passed": False,
            "error": (
                reason if re.fullmatch(r"[a-z][a-z0-9_]{2,80}", reason) else type(error).__name__
            ),
            "scope": "core release audit",
        }
    rendered = json.dumps(result, indent=2) + "\n"
    validate_public_text(rendered, "core-audit.json")
    (args.out / "core-audit.json").write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
