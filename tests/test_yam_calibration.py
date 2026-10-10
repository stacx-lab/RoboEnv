"""Calibration practice contracts, exercised without importing Isaac or starting a GPU."""

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from helpers import load_tool
from roboenv.devices.documentation import render_documentation
from roboenv.tasks.resolve import resolve_task
from roboenv_runtime.adapters.isaaclab.adapter import IsaacLabAdapter
from roboenv_runtime.adapters.isaaclab.registry import get_task
from roboenv_runtime.adapters.isaaclab.robots.yam import assets, native
from roboenv_runtime.adapters.isaaclab.tasks.yam_calibration.scene import (
    FIDUCIALS,
    black_rectangles,
)
from roboenv_runtime.adapters.isaaclab.tasks.yam_calibration.task import TASKS, load_profiles
from roboenv_runtime.core.types import ScoreReport
from roboenv_runtime.runtime.world import WorldSession

ROOT = Path(__file__).resolve().parents[1]


def physical_profiles():
    profiles = {
        p.parent.name: {
            k: v
            for k, v in yaml.safe_load(p.read_text()).items()
            if k in {"type", "intrinsics", "extrinsics"}
        }
        for p in (ROOT / "catalog/robots/abc/abc-yam/devices").glob("*/profile.yaml")
    }

    return assets.rgb_profiles(profiles)


@pytest.mark.parametrize("name", TASKS)
def test_task_profiles_mask_only_targets_everywhere(name):
    resolved = resolve_task("isaaclab/" + name, project=ROOT)
    profiles = resolved.task.backend_kwargs["device_profiles"]
    assert resolved.task.budget_mode == "practice"
    assert "max_steps" not in resolved.task.backend_kwargs
    assert not resolved.task.stop_on_success
    unknown = {
        n
        for n, p in profiles.items()
        if p.get("extrinsics", {}).get("mount", {}).get("translation_m") == "???"
    }
    assert unknown == set(TASKS[name])
    for spec in resolved.bindings.devices:
        raw = yaml.safe_load((resolved.robot / spec.profile).read_text())
        definition = spec.definition(resolved.robot)
        assert definition.extrinsics == raw.get("extrinsics", {})
        assert "???" not in json.dumps(raw["access"])
        assert "???" not in json.dumps(raw["intrinsics"])
        if spec.name in unknown:
            assert definition.extrinsics["mount"]["rotation_xyzw"] == "???"
            assert definition.extrinsics["source"] == "unknown"
            assert definition.extrinsics["calibrated_at"] is None
            assert definition.extrinsics["needs_calibration"]
    if name.endswith("dual-arm"):
        mount = profiles["right_arm"]["extrinsics"]["mount"]
        assert (mount["parent"], mount["parent_frame"]) == ("left_arm", "left_arm")
        assert {"left_contact", "right_contact"} <= profiles.keys()
    # The documentation receives the same public definitions, never an owner manifest.
    devices = [
        {
            "name": s.name,
            "type": profiles[s.name]["type"],
            "intrinsics": profiles[s.name]["intrinsics"],
            "extrinsics": profiles[s.name].get("extrinsics", {}),
            "actions": {},
        }
        for s in resolved.bindings.devices
    ]
    docs = render_documentation({}, devices, {"endpoint": "http://127.0.0.1:8123"})
    assert "???" in docs
    assert not list(resolved.robot.rglob("assets.json"))


def test_catalog_generation_is_deterministic(tmp_path):
    generator = load_tool("generate_yam_calibration")
    catalog, bindings = tmp_path / "catalog", tmp_path / "bindings"
    generator.generate(catalog, bindings)
    for path in catalog.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (ROOT / "catalog" / path.relative_to(catalog)).read_bytes()
    for path in bindings.rglob("*.yaml"):
        original = ROOT / "src/roboenv/devices/bindings/defaults" / path.relative_to(bindings)
        assert path.read_bytes() == original.read_bytes()


@pytest.mark.parametrize("name", TASKS)
def test_geometry_does_not_depend_on_calibration_estimates(monkeypatch, tmp_path, name):
    physical = physical_profiles()
    manifest = tmp_path / "assets.json"
    manifest.write_text("owner identity")
    monkeypatch.setattr(native, "load_manifest", lambda path: {"physical_profiles": physical})
    closed = []
    from roboenv_runtime.adapters import gpu

    monkeypatch.setattr(
        gpu,
        "launch_isaaclab",
        lambda *a, **kw: SimpleNamespace(close=lambda **kwargs: closed.append(True)),
    )
    monkeypatch.setattr(native.Native, "_create", lambda *a: None)
    public = load_profiles(name)
    for target in TASKS[name]:
        public[target]["extrinsics"]["mount"].update(
            translation_m=[8, 9, 10], rotation_xyzw=[0, 0, 0, 1]
        )
    world = native.Native(manifest, get_task(name), public)
    try:
        assert world.physical == physical
        public["camera_left"]["extrinsics"]["mount"]["translation_m"] = [99, 99, 99]
        assert world.physical == physical
        assert world.physical is not physical
    finally:
        world.close()
    assert closed == [True]
    public["camera_top"]["intrinsics"]["color"]["width"] = 1
    with pytest.raises(ValueError, match="hardware profile"):
        native.Native(manifest, get_task(name), public)


class FakeNative:
    """Only replaces physics, leaving the actual environment and shared clock active."""

    def __init__(self, manifest, task_name, profiles):
        self.home = {side: [0.0] * 6 for side in ("left", "right")}
        self.asset_sha = "test"
        self.calls = []

    def reset(self, seed):
        self.steps, self.seed = 0, seed

    def advance(self, targets):
        self.calls.append(copy.deepcopy(targets))
        self.steps += 1

    def read_device(self, name):
        return {"positions": [0.5] * 6, "faults": [], "timestamp": self.steps * 0.034}

    def close(self):
        pass


@pytest.mark.parametrize("name", TASKS)
def test_practice_clock_has_no_verdict_or_horizon(monkeypatch, name):
    monkeypatch.setattr(native, "Native", FakeNative)
    adapter = IsaacLabAdapter(task_name=name, stop_on_success=True)
    env = adapter.environment
    try:
        session = WorldSession(adapter, "stepped")
        env.set_target("left_gripper", [0.25])
        env.stop("left_gripper")
        assert env.targets["left_gripper"] == [0.25]
        env.set_target("right_arm", [0.1] * 6)
        env.read_device("right_arm")
        assert env.native.steps == 0
        waiting = session.wait(0.068)
        session.advance()
        session.advance()
        assert waiting.result()["time_s"] == pytest.approx(0.068)
        env.stop("right_arm")
        assert env.targets["right_arm"] == [0.5] * 6
        assert adapter.score() == ScoreReport()
        assert adapter.budget().max is None
        assert adapter.budget().used == 2
        adapter.configure_execution_mode("realtime")
        assert adapter.budget().max is None
        before = copy.deepcopy(env.targets)
        for bad in ([float("nan")] * 6, [True] * 6, [1.0], "invalid"):
            with pytest.raises(ValueError):
                env.set_target("right_arm", bad)
        with pytest.raises(ValueError):
            env.set_target("left_gripper", [1.1])
        assert env.targets == before
        record = env.record()
        assert not {"physical_profiles", "completed", "calibration_error"} & record.keys()
        assert "upstreams.lock.json#isaaclab+abc" in record["sources"]
    finally:
        adapter.close()


def minimal_source(tmp_path):
    mesh = tmp_path / "meshes"
    mesh.mkdir()
    (mesh / "arm.stl").write_bytes(b"fixture mesh")
    source = tmp_path / "source.xml"
    source.write_text("""<mujoco><compiler meshdir="meshes"/><asset>
      <mesh file="arm.stl"/></asset><worldbody>
      <body name="left_arm" pos="0.2525 0.31 0.75"><joint name="left_joint1"/></body>
      <body name="right_arm" pos="0.2525 -0.31 0.75"><joint name="right_joint1"/></body>
      <body name="gate_collision"><camera name="top"/></body>
      <body name="play_table"/><body name="overhead_camera"><camera name="extra"/></body>
      <geom name="table_plane" type="plane"/></worldbody>
      <equality><joint joint1="left_finger" joint2="right_finger"/></equality>
      <actuator><position joint="left_joint1"/><position joint="right_joint1"/></actuator>
      </mujoco>""")
    return source


def test_asset_export_is_portable_fixed_and_tamper_checked(tmp_path):
    source = minimal_source(tmp_path)
    manifest = assets.prepare(source, physical_profiles(), tmp_path / "bundle")
    facts = assets.load_manifest(manifest)
    assert facts["source_mjcf_sha256"] == assets.sha256(source)
    for side in ("left", "right"):
        tree = ET.parse(manifest.parent / (side + ".xml"))
        bodies = tree.findall("worldbody/body")
        assert len(bodies) == 1 and bodies[0].get("name") == side + "_arm"
        assert bodies[0].get("pos") == "0 0 0"
        assert not tree.findall("equality")
        assert not tree.findall(".//camera")
        assert tree.find("compiler").get("meshdir") == "meshes"
        assert tree.find("actuator/position").get("joint") == side + "_joint1"
    station = ET.parse(manifest.parent / "station.xml")
    assert station.find("worldbody/geom[@name='table_collision']").get("type") == "box"
    assert not station.findall("worldbody/geom[@type='plane']")
    with pytest.raises(FileExistsError):
        assets.prepare(source, {}, manifest.parent)
    (manifest.parent / "meshes/arm.stl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or unsafe"):
        assets.load_manifest(manifest)


def test_fiducials_are_noncoplanar_and_board_runs_preserve_pattern():
    points = np.array([p for p, _ in FIDUCIALS.values()])
    assert np.linalg.matrix_rank(points[1:] - points[0]) == 3
    pattern = np.array([[0, 0, 255, 0], [255, 0, 255, 255]], dtype=np.uint8)
    rebuilt = np.full(pattern.shape, 255, dtype=np.uint8)
    for x0, y0, x1, y1 in black_rectangles(pattern):
        rebuilt[y0:y1, x0:x1] = 0
    np.testing.assert_array_equal(rebuilt, pattern)


def test_bindings_publish_unknown_mounts_and_read_only_contacts(monkeypatch, tmp_path):
    from roboenv_runtime.bindings.backends.isaaclab.yam import bind

    monkeypatch.setattr(native, "Native", FakeNative)
    resolved = resolve_task("isaaclab/yam-calibrate-dual-arm", project=ROOT)
    adapter = IsaacLabAdapter(**resolved.task.backend_kwargs)
    adapter.run_dir = tmp_path
    context = SimpleNamespace(
        backend="isaaclab",
        embodiment="yam",
        native=adapter,
        devices={s.name: s.name for s in resolved.bindings.devices},
    )
    try:
        bound = {
            s.name: bind(s, resolved.robot, context)
            for s in resolved.bindings.devices
            if s.name != "world"
        }
        assert bound["right_arm"].definition.extrinsics["mount"]["translation_m"] == "???"
        for name in ("left_contact", "right_contact", "camera_top"):
            assert set(bound[name].actions) == {"load", "unload", "status", "read"}
        assert {"write", "stop"} <= bound["left_arm"].actions.keys()
        assert not any("submit" in device.actions for device in bound.values())
        assert adapter.environment.native.steps == 0
    finally:
        adapter.close()


@pytest.mark.parametrize("with_depth,rgb_size", [
    (False, None), (True, None), (False, (1, 1)),
    (True, (1, 1)), (False, (4, 2)), (False, (True, 1)),
])
def test_camera_serialization_keeps_rgb_only_and_rgbd_contracts(tmp_path, with_depth, rgb_size):
    from roboenv_runtime.adapters.isaaclab.common.observations import camera_state

    class Pixels:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return Pixels(self.data[index])

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    pixels = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)
    outputs = {"rgb": SimpleNamespace(torch=Pixels(pixels))}
    if with_depth:
        outputs["distance_to_image_plane"] = SimpleNamespace(
            torch=Pixels(np.ones((1, 2, 3, 1), dtype=np.float32))
        )
    calls = []
    camera = SimpleNamespace(
        cfg=SimpleNamespace(height=2, width=3),
        data=SimpleNamespace(output=outputs),
        update=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    if rgb_size is not None and (with_depth or rgb_size != (1, 1) or type(rgb_size[0]) is bool):
        with pytest.raises(ValueError):
            camera_state(camera, tmp_path, 0.034, rgb_size=rgb_size)
        assert not list(tmp_path.glob("*.png"))
        return
    state = camera_state(camera, tmp_path, 0.034, rgb_size=rgb_size)
    if rgb_size is not None:
        from PIL import Image

        image = np.asarray(Image.open(state["file"]))
        assert image.shape == (1, 1, 3)
        np.testing.assert_array_equal(image[0, 0], [10, 11, 12])
    assert Path(state["file"]).read_bytes().startswith(b"\x89PNG")
    assert state["timestamp"] == 0.034
    assert ("depth_file" in state) == with_depth
    assert calls == [((0.0,), {"force_recompute": True})]


def test_native_pose_uses_pinned_isaaclab_xyzw_convention():
    # A 90-degree Z rotation must map flange +X onto world +Y.
    transform = native.pose_matrix([1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(transform @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)


def test_supersampled_intrinsics_preserve_public_pixel_centers():
    projection = {"fx": 501.3, "fy": 499.7, "cx": 312.25, "cy": 241.75}
    intrinsic = np.array(native.render_intrinsics(projection))
    ray = np.array([0.12, -0.23, 1.0])
    rendered = (intrinsic @ ray)[:2]
    public = (rendered + 0.5) / native.RGB_SUPERSAMPLING - 0.5
    np.testing.assert_allclose(public, [501.3 * .12 + 312.25, 499.7 * -.23 + 241.75])


@pytest.mark.parametrize("compensation", [0.0, 1.0, 0.5])
def test_body_dynamics_uses_compiled_mass_inertia_and_gravity(monkeypatch, compensation):
    model = SimpleNamespace(
        nbody=2, body_mass=np.array([0, 0.001]), body_inertia=np.array([[0]*3, [0.001]*3]),
        body_ipos=np.array([[0]*3, [0.009, 0, 0]]),
        body_iquat=np.array([[1, 0, 0, 0], [0, 0, 0, 1]]),
        body_gravcomp=np.array([0, compensation]),
    )
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: model),
        mj_id2name=lambda model, kind, index: "camera_frame",
        mjtObj=SimpleNamespace(mjOBJ_BODY=1),
    ))
    if compensation == 0.5:
        with pytest.raises(ValueError, match="gravity compensation"):
            native.body_dynamics("source.xml")
    else:
        assert native.body_dynamics("source.xml") == {"camera_frame": {
            "mass": 0.001, "inertia": [0.001]*3, "center": [0.009, 0, 0],
            "axes_wxyz": [0, 0, 0, 1], "disable_gravity": bool(compensation),
        }}


@pytest.mark.parametrize("rigid", [True, False])
def test_body_dynamics_authors_inertials_and_rejects_missing_rigid_body(monkeypatch, rigid):
    authored = {}
    prim = SimpleNamespace(HasAPI=lambda api: rigid)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prim)
    monkeypatch.setattr(native, "unique_prim", lambda *args: "/World/Left/camera_frame")
    monkeypatch.setattr(native, "body_dynamics", lambda path: {"camera_frame": {
        "mass": 0.001, "inertia": [0.001]*3, "center": [0.009, 0, 0],
        "axes_wxyz": [0, 0, 0, 1], "disable_gravity": False,
    }})
    mass_api = SimpleNamespace(**{
        "Create" + key + "Attr": lambda value, key=key: authored.update({key: value})
        for key in ("Mass", "DiagonalInertia", "CenterOfMass", "PrincipalAxes")
    })
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Gf=SimpleNamespace(Vec3f=lambda *v: v, Quatf=lambda w, v: (w, *v)),
        UsdPhysics=SimpleNamespace(
            RigidBodyAPI=object(), MassAPI=SimpleNamespace(Apply=lambda p: mass_api),
        ),
        PhysxSchema=SimpleNamespace(PhysxRigidBodyAPI=SimpleNamespace(Apply=lambda p:
            SimpleNamespace(CreateDisableGravityAttr=lambda v: authored.update(gravity=v)))),
    ))
    if not rigid:
        with pytest.raises(ValueError, match="rigid-body mapping"):
            native.apply_body_dynamics(stage, "/World/Left", "source.xml")
        assert authored == {}
    else:
        native.apply_body_dynamics(stage, "/World/Left", "source.xml")
        assert authored == {
            "Mass": 0.001, "DiagonalInertia": (0.001,)*3, "CenterOfMass": (0.009, 0, 0),
            "PrincipalAxes": (0, 0, 0, 1), "gravity": False,
        }


def test_calibration_rendering_preserves_resolution_without_offline_path_tracing():
    cfg = native.calibration_render_cfg(SimpleNamespace(RenderCfg=lambda **kwargs: kwargs))
    settings = cfg["carb_settings"]
    assert settings["/rtx/rendermode"] == "RealTimePathTracing"
    assert cfg["antialiasing_mode"] == "DLAA"
    assert settings["/rtx/post/motionblur/enabled"] is False
    assert cfg["enable_dlssg"] is False and cfg["enable_dl_denoiser"] is True


def test_yam_capture_pumps_rtx_without_advancing_physics(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(
        sys.modules, "isaaclab_physx.renderers.isaac_rtx_renderer_utils",
        SimpleNamespace(ensure_isaac_rtx_render_update=lambda: calls.append("pump")),
    )

    def serialize(camera, directory, stamp, *, rgb_size):
        assert rgb_size == (640, 480)
        return {"timestamp": stamp}

    monkeypatch.setattr(
        "roboenv_runtime.adapters.isaaclab.common.observations.camera_state", serialize
    )
    world = native.Native.__new__(native.Native)
    world.steps = 12
    world.sim = SimpleNamespace(render=lambda: calls.append("render"))
    world.torch = SimpleNamespace(tensor=lambda data, **kwargs: np.array(data), float32=None)
    world.cameras = {"camera_top": SimpleNamespace(set_world_poses=lambda **kwargs: None)}
    world.physical = {"camera_top": {
        "intrinsics": {"color": {"width": 640, "height": 480}},
        "extrinsics": {"mount": {"translation_m": [0, 0, 1], "rotation_xyzw": [0, 0, 0, 1]}},
    }}
    reading = world.capture("camera_top", tmp_path)
    assert calls == ["render", "pump"] * 3
    assert world.steps == 12
    assert reading["timestamp"] == pytest.approx(12 * world.interval)


def test_yam_control_holds_clipped_targets_through_every_physics_substep(monkeypatch):
    monkeypatch.setattr(native, "array", np.array)
    world = native.Native.__new__(native.Native)
    world.steps = 0
    world.robots, world.ids, writes, staged = {}, {}, {}, {}
    calls = []
    for side in ("left", "right"):
        # Joint order is simulator-owned, not assumed to match command order.
        names = [side + "_right_finger", side + "_left_finger"] + [
            f"{side}_joint{i}" for i in range(6, 0, -1)
        ]
        world.ids[side] = {name: i for i, name in enumerate(names)}
        writes[side] = []

        def stage(*, target, side=side):
            staged[side] = target.copy()

        world.robots[side] = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(clone=lambda: np.zeros((1, 8))),
                joint_pos_limits=np.array([[[-0.04, 0.04]] * 2 + [[-1, 1]] * 6]),
            ),
            set_joint_position_target_index=stage,
            write_data_to_sim=lambda side=side: writes[side].append(staged[side].copy()),
            update=lambda dt, side=side: calls.append((side, dt)),
        )
    world.contacts = {"left": SimpleNamespace(update=lambda dt: calls.append(("contact", dt)))}
    world.sim = SimpleNamespace(step=lambda *, render: calls.append(("step", render)))
    targets = {side + "_arm": [2, -2, 0.1, 0.2, 0.3, 0.4] for side in world.robots}
    targets.update({side + "_gripper": [1.0] for side in world.robots})
    world.advance(targets)
    for side in world.robots:
        np.testing.assert_allclose(writes[side], [[[-0.04, 0.04, 0.4, 0.3, 0.2, 0.1, -1, 1]]] * 17)
        assert calls.count((side, world.dt)) == 17
    assert calls.count(("step", False)) == calls.count(("contact", world.dt)) == 17
    assert world.steps == 1


def test_isaac_yam_keeps_rgb_sensors_when_abc_adds_depth():
    rgbd = {"camera_top": {"type": "rgbd", "intrinsics": {
        "color": {"width": 640}, "depth": {"scale_m_per_unit": .001}}, "extrinsics": {}}}
    original = copy.deepcopy(rgbd)
    converted = assets.rgb_profiles(rgbd)
    assert converted["camera_top"]["type"] == "rgb"
    assert converted["camera_top"]["intrinsics"] == {"color": {"width": 640}}
    assert rgbd == original
