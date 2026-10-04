"""Native rig camera profiles agree with the geometry the runtime derives from their intrinsics."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from roboenv_runtime.bindings.backends.robocasa.camera import intrinsics

ROOT = Path(__file__).resolve().parents[1]
# The rigs whose camera facts come from the shared helper; RoboDojo derives its own from Isaac's
# focal length and aperture (adapters/robodojo/rig.py).
NATIVE_RIGS = ("robotwin", "behavior")


def native_cameras():
    for backend in NATIVE_RIGS:
        for profile in sorted((ROOT / "catalog/robots" / backend).glob("*/devices/*/profile.yaml")):
            spec = yaml.safe_load(profile.read_text())
            introspected = spec["intrinsics"].get("source") == "introspected"
            if spec.get("type") in {"rgb", "rgbd"} and introspected:
                yield pytest.param(profile, id=f"{backend}/{profile.parent.name}")


@pytest.mark.parametrize("profile", list(native_cameras()))
def test_introspected_camera_fov_follows_the_shared_camera_geometry(profile):
    """The runtime derives a native camera's field of view from its focal length with the shared
    helper and refuses the catalog when the two differ (same_facts, 1e-10); a change to that
    helper has to come with regenerated profiles."""
    color = yaml.safe_load(profile.read_text())["intrinsics"]["color"]
    projection = color["projection"]
    vertical = math.degrees(2 * math.atan(color["height"] / (2 * projection["fy"])))
    expected = intrinsics(color["width"], color["height"], vertical, True)["color"]["fov"]
    for key in ("horizontal_deg", "vertical_deg"):
        assert np.isclose(color["fov"][key], expected[key], atol=1e-10, rtol=0), (profile, key)
