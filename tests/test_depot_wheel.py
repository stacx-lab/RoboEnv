"""Exercise the depot Dockerfile's wheel selection without a Docker daemon."""

import shlex
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_depot_requires_exactly_one_runtime_wheel(tmp_path, count):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    output = tmp_path / "runtime"
    for index in range(count):
        filename = wheels / f"roboenv_runtime-0.{index}.0-py3-none-any.whl"
        with zipfile.ZipFile(filename, "w") as wheel:
            wheel.writestr("roboenv_runtime/__init__.py", "# extracted runtime\n")
    # An SDK wheel must never be selected as the runtime.
    (wheels / "roboenv-0.10.0-py3-none-any.whl").write_bytes(b"unrelated")
    dockerfile = (ROOT / "runtime/packaging/Dockerfile.depot").read_text()
    command = "set --" + dockerfile.split("&& set --", 1)[1].split("\nENV", 1)[0]
    command = command.replace("/opt/wheels/", shlex.quote(str(wheels)) + "/")
    command = command.replace("/opt/roboenv-runtime", shlex.quote(str(output)))
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    assert (result.returncode == 0) == (count == 1), result.stderr
    assert (output / "roboenv_runtime/__init__.py").is_file() == (count == 1)
