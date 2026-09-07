"""Self-contained Roboenv tests; no parent workspace or agent package is required."""

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT / "src", PROJECT / "runtime/src"):
    sys.path.insert(0, str(path))

pytest_plugins = ["service_fixtures"]


def pytest_configure(config):
    if config.option.basetemp is None:
        root = PROJECT / ".cache/pytest"
        root.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = tempfile.mkdtemp(prefix="run-", dir=root)
