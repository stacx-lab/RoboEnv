from pathlib import Path

import pytest
from roboenv.privacy import audit
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
