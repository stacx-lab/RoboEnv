"""Check distributable contents independently of editable source imports."""

import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
wheels = [root / f"{name}-0.2.0-py3-none-any.whl" for name in ("roboenv", "roboenv_runtime")]
for wheel in wheels:
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        if any("/devices/" in name and name.startswith("roboenv_runtime/") for name in names):
            raise SystemExit("retired device package found in runtime wheel")
        if any("/tests/" in name or "solution.py" in name for name in names):
            raise SystemExit("test/reference data leaked into distribution")
code = (
    "import sys;"
    f"sys.path[:0] = {[str(w.resolve()) for w in wheels]!r};"
    "import roboenv, roboenv_runtime.runtime.worker;"
    "assert '.whl/' in roboenv.__file__;"
    "assert not {'torch','mujoco','harbor','roborl','complex'} & sys.modules.keys()"
)
subprocess.run([sys.executable, "-I", "-c", code], check=True)
print("Wheel contents and lightweight isolated imports passed.")
