"""Summarize Trivy results without publishing source snippets or host paths."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(dependencies, source):
    for report in (dependencies, source):
        if (
            report.get("SchemaVersion") != 2
            or report.get("ArtifactType") != "filesystem"
            or not isinstance(report.get("ArtifactName"), str)
            or not isinstance(report.get("Results", []), list)
        ):
            raise ValueError("missing or unsupported scanner report")
    package_results = [
        item
        for item in dependencies.get("Results", [])
        if item.get("Class") == "lang-pkgs" and item.get("Packages")
    ]
    if not package_results:
        raise ValueError("dependency scanner did not inventory any packages")
    vulnerabilities = Counter(
        finding.get("Severity", "UNKNOWN")
        for item in dependencies.get("Results", [])
        for finding in item.get("Vulnerabilities", [])
    )
    secrets = sum(len(item.get("Secrets", [])) for item in source.get("Results", []))
    return {
        "schema": "roboenv/security-summary/1",
        "passed": not secrets
        and not (
            vulnerabilities["HIGH"] or vulnerabilities["CRITICAL"] or vulnerabilities["UNKNOWN"]
        ),
        "scope": "owned source secrets and development lock vulnerabilities; not simulator images",
        "inventoried_packages": sum(len(item["Packages"]) for item in package_results),
        "vulnerabilities_by_severity": dict(sorted(vulnerabilities.items())),
        "secret_findings": secrets,
        "raw_reports_published": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(
            json.loads(args.dependencies.read_text()), json.loads(args.source.read_text())
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        result = {"passed": False, "error": "scanner_report_invalid"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
