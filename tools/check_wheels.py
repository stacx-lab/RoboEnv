"""Check distributable contents independently of editable source imports."""

import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
wheels = []
for name in ("roboenv", "roboenv_runtime"):
    # One freshly built wheel of each, whatever the version: no copy of it to bump by hand.
    found = sorted(root.glob(f"{name}-*-py3-none-any.whl"))
    if len(found) != 1:
        raise SystemExit(f"Expected one {name} wheel in {root}, found {len(found)}")
    wheels.append(found[0])
for wheel in wheels:
    with zipfile.ZipFile(wheel) as package:
        names = package.namelist()
        distribution = wheel.name.split("-")[0]
        if distribution == "roboenv":
            defaults = Path("src/roboenv/devices/bindings/defaults")
            expected_bindings = {p.relative_to("src").as_posix() for p in defaults.rglob("*.yaml")}
            actual_bindings = {
                n for n in names if n.startswith("roboenv/devices/bindings/defaults/")
            }
            if actual_bindings != expected_bindings:
                raise SystemExit("Binding defaults missing or stale in SDK wheel")
        retired = {
            "roboenv/assets/main.Dockerfile", "roboenv/assets/serve-native.sh",
            "roboenv/assets/verify.py",
        }
        if retired.intersection(names) or any(n.startswith("roboenv/harbor/") for n in names):
            raise SystemExit("Removed orchestration files leaked into distribution")
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        if any(
            line.lower().startswith(("requires-dist: harbor", "provides-extra: harbor"))
            for line in package.read(metadata).decode().splitlines()
        ):
            raise SystemExit("Removed orchestration dependency leaked into metadata")
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
version = Path("VERSION").read_text().strip()
code = (
    "import importlib.metadata, sys;"
    f"sys.path[:0] = {[str(w.resolve()) for w in wheels]!r};"
    "import roboenv, roboenv_runtime.runtime.worker;"
    f"assert roboenv.__version__ == roboenv_runtime.__version__ == {version!r};"
    "assert importlib.metadata.version('roboenv') == roboenv.__version__;"
    "assert importlib.metadata.version('roboenv-runtime') == roboenv_runtime.__version__;"
    "from roboenv.devices.bindings.spec import binding_resource;"
    "from roboenv.validation import yaml_load;"
    "resource = binding_resource('libero-panda-kitchen', backend='libero');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['devices']['camera_wrist']['config']['camera'] == 'robot0_eye_in_hand';"
    "resource = binding_resource('isaaclab-go2-d1', backend='isaaclab');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['embodiment'] == 'go2_d1';"
    "assert plan['devices']['arm']['binding'] == 'isaaclab.go2-d1/1';"
    "from roboenv_runtime.adapters.isaaclab.registry import TASKS, get_task;"
    "tasks = [get_task(name) for name in TASKS];"
    "assert len(tasks) == 8 and {task.embodiment for task in tasks} == "
    "{'go2_d1', 'yam', 'roboballet_eight_panda', 'booster_t2'};"
    "assert all('.whl/' in sys.modules[task.__module__].__file__ for task in tasks);"
    "resource = binding_resource('isaaclab-yam-calibrate-wrist-cameras', backend='isaaclab');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['devices']['camera_left']['binding'] == 'isaaclab.yam/1';"
    "resource = binding_resource('isaaclab-roboballet-eight-panda', backend='isaaclab');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['embodiment'] == 'roboballet_eight_panda';"
    "assert plan['devices']['arm1']['binding'] == 'isaaclab.roboballet-eight-panda/1';"
    "resource = binding_resource('isaaclab-booster-t2', backend='isaaclab');"
    "assert '.whl/' in str(resource);"
    "plan = yaml_load(resource.read_text());"
    "assert plan['devices']['left_hand']['binding'] == 'isaaclab.booster-t2/1';"
    "from importlib.resources import files;"
    "package = files('roboenv_runtime.adapters.isaaclab.robots.booster_t2');"
    "assert package.joinpath('controller/configs/locomanip_policy.yaml').is_file();"
    "assert package.joinpath('controller/configs/t2_revo2.yaml').is_file();"
    "assert package.joinpath('scenes/kitchen/scene.yaml').is_file();"
    "from roboenv_runtime.bindings import default_registry; default_registry();"
    "assert '.whl/' in roboenv.__file__;"
    "assert not {'torch','mujoco','isaaclab','isaacsim','harbor','roborl','kinex'}"
    " & sys.modules.keys()"
)
subprocess.run([sys.executable, "-I", "-c", code], check=True)
print("Wheel contents and lightweight isolated imports passed.")
