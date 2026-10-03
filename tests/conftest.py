"""Self-contained Roboenv tests; no parent workspace or agent package is required."""

import os
import sys
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT / "src", PROJECT / "runtime/src"):
    sys.path.insert(0, str(path))

pytest_plugins = ["service_fixtures"]


def pytest_configure(config):
    if config.option.basetemp is None:
        root = PROJECT / ".cache/pytest"
        root.mkdir(parents=True, exist_ok=True)
        # Let pytest number, lock, and prune its own temporary directories.
        os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(root))


@pytest.fixture(scope="session")
def catalog_profiles():
    """Read-only parsed catalog facts shared by exhaustive catalog checks."""
    return tuple((path, yaml.safe_load(path.read_text()))
                 for path in sorted((PROJECT / "catalog/robots").rglob("profile.yaml")))
