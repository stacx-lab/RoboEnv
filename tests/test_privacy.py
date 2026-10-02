from pathlib import Path

import pytest
from roboenv.privacy import audit, validate_public_text
from roboenv.tasks.artifacts import Bundle

ROOT = Path(__file__).resolve().parents[1]


def test_export_rejects_private_paths_without_echoing_them(tmp_path):
    private = "/" + "home/" + "example-owner/" + "asset"
    with pytest.raises(ValueError, match="privacy check") as error:
        Bundle(tmp_path).add("task.yaml", "asset: " + private)
    assert private not in str(error.value)
    assert not (tmp_path / "task.yaml").exists()


def test_container_paths_are_not_host_identifiers(tmp_path):
    bundle = Bundle(tmp_path)
    bundle.add("instruction.md", "Work in /app; read /env; write /logs/agent.")
    bundle.commit({})


def test_publishable_source_has_no_machine_identifiers():
    result = audit(ROOT)
    assert result["passed"], result["findings"]


def test_local_notes_are_excluded_without_hiding_owned_source(tmp_path):
    private = "/" + "home/" + "example-owner/" + "asset"
    local = tmp_path / ".local"
    local.mkdir()
    (local / "report.md").write_text(private)
    assert audit(tmp_path)["passed"]
    (tmp_path / "source.py").write_text(private)
    assert audit(tmp_path)["findings"] == [{"file": "source.py", "kind": "user_home"}]


@pytest.mark.parametrize("prefix", ["https://example.com", "http://example.com/docs"])
@pytest.mark.parametrize("root", ["home", "Users", "scratch", "nethome", "localscratch"])
def test_public_url_paths_are_not_host_paths(tmp_path, prefix, root):
    content = prefix + "/" + root + "/example-owner/documentation"
    validate_public_text(content, "")
    (tmp_path / "README.md").write_text(content)
    assert audit(tmp_path)["passed"]


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "file://",
        "https://example.com/?path=",
        "https://example.com/#",
        "https://example.com/home/en/docs ",
    ],
)
@pytest.mark.parametrize("root", ["home", "Users", "scratch", "nethome", "localscratch"])
def test_private_paths_remain_rejected(tmp_path, prefix, root):
    content = prefix + "/" + root + "/example-owner/asset"
    with pytest.raises(ValueError, match="privacy check"):
        validate_public_text(content, "")
    with pytest.raises(ValueError, match="privacy check"):
        validate_public_text("", content)
    (tmp_path / "README.md").write_text(content)
    assert not audit(tmp_path)["passed"]
