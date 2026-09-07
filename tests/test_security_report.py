"""Scanner summaries fail closed and never copy source snippets into artifacts."""

import json
from pathlib import Path

import pytest
from helpers import load_tool

ROOT = Path(__file__).resolve().parents[1]
SCANNER = load_tool("check_security_report")


def clean_reports():
    source = {"SchemaVersion": 2, "ArtifactName": ".", "ArtifactType": "filesystem"}
    dependencies = {
        **source,
        "Results": [
            {
                "Class": "lang-pkgs",
                "Packages": [{"Name": "example", "Version": "1.0"}],
            }
        ],
    }
    return dependencies, source


def test_clean_scan_requires_a_real_package_inventory():
    dependencies, source = clean_reports()
    result = SCANNER.summarize(dependencies, source)
    assert result["passed"]
    assert result["inventoried_packages"] == 1
    with pytest.raises(ValueError, match="inventory"):
        SCANNER.summarize(source, source)


@pytest.mark.parametrize("severity", ["HIGH", "CRITICAL", "UNKNOWN"])
def test_high_or_unclassified_vulnerabilities_block(severity):
    dependencies, source = clean_reports()
    dependencies["Results"][0]["Vulnerabilities"] = [{"Severity": severity}]
    assert not SCANNER.summarize(dependencies, source)["passed"]


def test_secret_at_any_severity_blocks_without_disclosing_snippet():
    dependencies, source = clean_reports()
    source["Results"] = [
        {
            "Target": "private-location",
            "Secrets": [{"Severity": "LOW", "Match": "private-value", "Code": "private-code"}],
        }
    ]
    result = SCANNER.summarize(dependencies, source)
    assert not result["passed"]
    assert result["secret_findings"] == 1
    assert "private-" not in json.dumps(result)


@pytest.mark.parametrize("malformed", [{}, {"SchemaVersion": 2}, {"Results": []}])
def test_missing_or_truncated_reports_do_not_pass(malformed):
    dependencies, _ = clean_reports()
    with pytest.raises(ValueError):
        SCANNER.summarize(dependencies, malformed)


def test_invalid_json_produces_sanitized_failure(tmp_path, capsys):
    raw = tmp_path / "scanner.json"
    raw.write_text("broken json")
    output = tmp_path / "summary.json"
    assert (
        SCANNER.main(
            [
                "--dependencies",
                str(raw),
                "--source",
                str(raw),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads(output.read_text()) == {"passed": False, "error": "scanner_report_invalid"}
    assert str(tmp_path) not in capsys.readouterr().out
