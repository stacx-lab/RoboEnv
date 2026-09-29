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
        distribution = wheel.name.split("-")[0]
        required = (
            "roboenv/assets/isaaclab-g1.json"
            if distribution == "roboenv"
            else "roboenv_runtime/adapters/isaaclab/g1.json"
        )
        if required not in names:
            raise SystemExit(f"G1 resource missing from wheel: {required}")
        if distribution == "roboenv":
            defaults = Path("src/roboenv/devices/bindings/defaults")
            expected_bindings = {p.relative_to("src").as_posix() for p in defaults.rglob("*.yaml")}
            actual_bindings = {
                n for n in names if n.startswith("roboenv/devices/bindings/defaults/")
            }
            if actual_bindings != expected_bindings:
                raise SystemExit("Binding defaults missing or stale in SDK wheel")
        source = Path("src") if distribution == "roboenv" else Path("runtime/src")
        expected = {
            path.relative_to(source).as_posix() for path in (source / distribution).rglob("*.py")
        }
        actual = {name for name in names if name.endswith(".py")}
        if actual != expected:
            raise SystemExit(
                f"wheel/source mismatch: missing={sorted(expected - actual)}, "
                f"stale={sorted(actual - expected)}; use a fresh build directory"
            )
        if any("/devices/" in name and name.startswith("roboenv_runtime/") for name in names):
            raise SystemExit("retired device package found in runtime wheel")
        if any("/tests/" in name or "solution.py" in name for name in names):
            raise SystemExit("test/reference data leaked into distribution")
code = (
    "import importlib, sys;"
    f"sys.path[:0] = {[str(w.resolve()) for w in wheels]!r};"
    "import roboenv, roboenv_runtime.runtime.worker;"
    "from roboenv.devices.bindings.spec import binding_resource;"
    "from roboenv.validation import yaml_load;"
    "resource = binding_resource('libero-panda-kitchen', backend='libero');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['devices']['camera_wrist']['config']['camera'] == 'robot0_eye_in_hand';"
    "from roboenv_runtime.bindings import default_registry; default_registry();"
    "assert '.whl/' in roboenv.__file__;"
    "assert not {'torch','mujoco','harbor','roborl','kinex'} & sys.modules.keys()"
)
subprocess.run([sys.executable, "-I", "-c", code], check=True)
print("Wheel contents and lightweight isolated imports passed.")
