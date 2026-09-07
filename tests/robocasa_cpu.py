"""Real Panda–Omron physics without a kitchen or renderer; not a RoboCasa replay."""

from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.robocasa.adapter import RoboCasaAdapter
from roboenv_runtime.adapters.robocasa.backend import DEFAULT_CAMERAS, RoboCasaEnv


def make_adapter(*, realtime=False):
    from xml.etree import ElementTree as ET

    from robocasa.utils.camera_utils import get_robot_cam_configs
    from robosuite.environments.manipulation.lift import Lift

    class CameraLift(Lift):
        def _load_model(self):
            super()._load_model()
            for name, config in get_robot_cam_configs("PandaOmron").items():
                parent = self.model.worldbody.find(f".//body[@name='{config['parent_body']}']")
                camera = parent.find(f".//camera[@name='{name}']")
                if camera is None:
                    camera = ET.SubElement(parent, "camera", name=name, mode="fixed")
                camera.set("pos", " ".join(map(str, config["pos"])))
                camera.set("quat", " ".join(map(str, config["quat"])))
                camera.attrib.update(config.get("camera_attribs", {}))

    from roboenv_runtime.adapters.robocasa.backend import joint_controller_config

    native = RoboCasaEnv.__new__(RoboCasaEnv)
    native.env = CameraLift(
        robots="PandaOmron",
        controller_configs=joint_controller_config("PandaOmron"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
    )
    native.realtime = realtime
    native._render = True
    native._camera_names = DEFAULT_CAMERAS
    native._camera_size = (8, 12)
    native._frames = []
    native._max_steps = None
    native.horizon = 1000
    # Synthetic pixels test transport only. No GPU/render certification implied.
    native.render = lambda camera=None: np.full(
        (8, 12, 3), 73 + (DEFAULT_CAMERAS.index(camera) if camera else 0), dtype=np.uint8
    )
    native.render_rgbd = lambda camera: (
        native.render(camera),
        np.full((8, 12), 0.4 + DEFAULT_CAMERAS.index(camera) * 0.1),
    )
    native.env.get_ep_meta = lambda: {"lang": "Lift the cube (CPU service fixture)."}
    reset = native.reset

    def reset_scene(*, seed=None):
        result = reset(seed=seed)
        native.env.fixtures = {}
        native.env.objects = {"cube": native.env.cube}
        native.env.obj_body_id = {"cube": native.env.cube_body_id}
        return result

    native.reset = reset_scene
    native.task_fixtures = lambda: {}
    adapter = RoboCasaAdapter.__new__(RoboCasaAdapter)
    adapter._env = native
    return adapter


def check(root, temporary):
    import sys

    sys.path.insert(0, str(root.parent / "complex/python"))
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    from roboenv_runtime.adapters.robocasa.depth import write_depth

    # Test the native conversion path without creating an OpenGL context.
    renders = []
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    buffer = np.array([[0.0, 0.5], [0.9, 1.0]])

    def render(**kwargs):
        renders.append(kwargs)
        return pixels, buffer

    renderer = RoboCasaEnv.__new__(RoboCasaEnv)
    renderer._render = True
    renderer._camera_names = DEFAULT_CAMERAS
    renderer._camera_size = (2, 2)
    renderer.env = SimpleNamespace(sim=SimpleNamespace(
        render=render,
        model=SimpleNamespace(stat=SimpleNamespace(extent=2),
                              vis=SimpleNamespace(map=SimpleNamespace(znear=0.05, zfar=5))),
    ))
    rgb, depth = renderer.render_rgbd(DEFAULT_CAMERAS[0])
    np.testing.assert_array_equal(rgb, pixels[::-1])
    np.testing.assert_allclose(depth, [[0.1 / (1 - 0.9 * 0.99), 0], [0.1, 0.1 / 0.505]])
    assert renders == [{"camera_name": DEFAULT_CAMERAS[0], "height": 2, "width": 2,
                        "depth": True}]
    try:
        renderer.render_rgbd("not-enabled")
    except ValueError:
        pass
    else:
        raise AssertionError("Disabled RGB-D camera accepted")
    depth_path = temporary / "depth-test.pgm"
    write_depth(depth_path, [[1.234, 0, np.nan, np.inf, -1, 65.536, 65.535]])
    header, encoded = depth_path.read_bytes().split(b"65535\n", 1)
    assert header == b"P5\n7 1\n"
    np.testing.assert_array_equal(np.frombuffer(encoded, dtype=">u2"),
                                  [1234, 0, 0, 0, 0, 0, 65535])
    try:
        write_depth(depth_path, [1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed depth image accepted")

    from roboenv.devices.bindings.registry import BindingContext
    from roboenv.devices.bindings.spec import BindingPlan, BindingSpec
    from roboenv.devices.export import export_devices
    from roboenv.devices.model import DeviceError
    from roboenv_runtime.bindings import default_registry

    adapter = make_adapter()
    adapter.run_dir = temporary
    adapter.reset(0)
    native = adapter._env
    import shutil

    source = root / "catalog/robots/robocasa-panda-omron"
    plan = BindingPlan.load(source, backend="robocasa")
    robot = temporary / "robot"
    shutil.copytree(source, robot)
    shutil.copyfile(
        root / "tests/fixtures/robocasa-composite/profile.yaml", robot / "composite.yaml"
    )
    service = default_registry().assemble(
        plan, robot, BindingContext("robocasa", "panda_omron", adapter)
    )

    def call(name, action, args=None):
        return service.call({"device": name, "action": action, "args": args or {}})["data"]

    def local(name, action, args):
        result = drivers[name]["handle"](action, args)
        if not result["ok"]:
            raise ValueError(result["error"])
        return {key: value for key, value in result.items() if key != "ok"}

    try:
        export_devices(
            service.inventory()["devices"],
            temporary / "devices",
            {
                "endpoint": "http://127.0.0.1:8123",
                "session": "cpu",
                "generation": 0,
            },
        )
        import runpy

        drivers = {name: runpy.run_path(str(temporary / "devices" / name / "driver.py"))
                   for name in ("arm", "torso")}
        for name, driver in drivers.items():
            driver["handle"].__globals__["remote"] = (
                lambda action, args, name=name: call(name, action, args)
            )
            for action in ("fk", "ik"):
                assert action not in service.describe(name)["actions"]
                try:
                    call(name, action, {})
                except DeviceError:
                    pass
                else:
                    raise AssertionError("Simulation exposes a local kinematics operation")
        assert adapter.tools() == {}
        assert service.describe("world")["actions"] == {}
        assert "mount" not in service.describe("body")["extrinsics"]
        assert "needs_calibration" not in service.describe("body")["extrinsics"]
        # A custom rig can still explicitly opt into a localized world edge.
        base_spec = next(s for s in plan.devices if s.entity == "base")
        localized = default_registry().assemble(
            replace(plan, devices=(replace(base_spec, overrides={"extrinsics": {
                "mount": {"parent": "world", "pose": "???"}
            }}),)), robot, BindingContext("robocasa", "panda_omron", adapter),
        )
        assert localized.describe("body")["extrinsics"]["mount"] == {
            "parent": "world", "pose": "???", "dynamic": True
        }
        before = native._step_count
        image = call("camera_left", "read")
        assert Path(image["file"]).read_bytes().startswith(b"\x89PNG")
        assert (image["width"], image["height"]) == (12, 8)
        assert image["depth_scale"] == 0.001
        assert Path(image["depth_file"]).read_bytes().startswith(b"P5\n12 8\n65535\n")
        assert service.describe("camera_left")["type"] == "rgbd"
        assert native._step_count == before
        depths = [Path(call(name, "read")["depth_file"]).read_bytes()
                  for name in ("camera_left", "camera_right", "camera_wrist")]
        assert len(set(depths)) == 3
        assert native._step_count == before
        images = [
            Path(call(name, "read")["file"]).read_bytes()
            for name in ("camera_left", "camera_right", "camera_wrist")
        ]
        assert len(set(images)) == 3  # each logical camera reaches its own native sensor
        assert native._step_count == before
        camera_spec = next(s for s in plan.devices if s.name == "camera_right")
        for names in ((), (DEFAULT_CAMERAS[0],)):
            native._camera_names = names
            try:
                default_registry().assemble(
                    replace(plan, devices=(camera_spec,)), robot,
                    BindingContext("robocasa", "panda_omron", adapter),
                )
            except ValueError as exc:
                assert "Camera must be enabled" in str(exc)
            else:
                raise AssertionError("Disabled camera accepted")
        native._camera_names = DEFAULT_CAMERAS
        assert "scene" not in {p["name"] for p in service.inventory()["devices"]}
        for name in ("arm", "body", "torso", "gripper"):
            state = call(name, "read")
            assert set(state) >= {"positions", "velocities", "faults"}
            size = len(service.describe(name)["intrinsics"]["actions"])
            before = native._step_count
            for target in ([0.0] * (size + 1), [100.0] * size, [float("nan")] * size):
                try:
                    call(name, "write", {"target": target})
                except DeviceError:
                    pass
                else:
                    raise AssertionError("Invalid target accepted")
            assert native._step_count == before
            target = (service.describe(name)["intrinsics"]["home_joints"]
                      if name in ("arm", "torso") else [0.1] * size)
            call(name, "write", {"target": target})
            assert native._step_count == before + 1
            assert call(name, "stop") == {"stopped": True}
            assert native._step_count == before + 1
            if name in ("arm", "torso"):
                facts = service.describe(name)["intrinsics"]
                assert facts["source"] == "introspected"
                expected_home = (native._robot.init_qpos if name == "arm"
                                 else native._robot.robot_model.init_torso_qpos)
                np.testing.assert_allclose(facts["home_joints"], expected_home)
                assert facts["velocity_limits"]["measured_joint_velocity_cap"] is None
                assert all("max_velocity" not in j for j in facts["joints"])
                controller = native._robot.part_controllers["right" if name == "arm" else "torso"]
                indices = np.asarray(controller.joint_index)
                model = native.sim.model._model
                ranges = model.jnt_range[indices]
                assert model.jnt_limited[indices].all()
                np.testing.assert_allclose(
                    [[j["limits"]["min"], j["limits"]["max"]] for j in facts["joints"]],
                    ranges,
                )
                assert facts["joint_limit_semantics"]["units"] == (
                    "rad" if name == "arm" else "m"
                )
                np.testing.assert_array_equal(
                    facts["joint_limit_semantics"]["activation_margin"], model.jnt_margin[indices]
                )
                state = call(name, "read")
                live = native.sim.data._data
                snapshot = (live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time)
                current = local(name, "fk", {"joints": state["positions"]})["pose"]
                np.testing.assert_allclose(current, state["flange_pose"], atol=1e-8)
                alternate = list(state["positions"])
                alternate[0] += 0.015
                target = local(name, "fk", {"joints": alternate})["pose"]
                assert not np.allclose(current[:3], target[:3])
                solved = local(name, "ik", {"pose": target, "seed": state["positions"]})
                actual = local(name, "fk", solved)["pose"]
                np.testing.assert_allclose(actual[:3], target[:3], atol=1e-4)
                unreachable = [100, 100, 100, 0, 0, 0, 1]
                try:
                    local(name, "ik", {"pose": unreachable})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unreachable pose accepted")
                for measured, saved in zip(
                    (live.qpos, live.qvel, live.ctrl), snapshot[:3], strict=True
                ):
                    np.testing.assert_array_equal(measured, saved)
                assert live.time == snapshot[3]
                invalid = list(state["positions"])
                invalid[0] = facts["joints"][0]["limits"]["max"] + 1
                for action, args in (("fk", {"joints": invalid}),
                                     ("ik", {"pose": current, "seed": invalid}),
                                     ("ik", {"pose": [0, 0, 0, 0, 0, 0, 0]})):
                    try:
                        local(name, action, args)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Invalid kinematics input accepted")
                # Physical state may slightly cross either soft limit. All
                # joints must remain observable and usable as implicit IK seeds.
                saved_qpos = live.qpos.copy()
                addresses = model.jnt_qposadr[indices]
                for index in range(len(indices)):
                    for side, sign in ((0, -1), (1, 1)):
                        bounded = np.asarray(facts["home_joints"]).copy()
                        bounded[index] = ranges[index, side]
                        target = local(name, "fk", {"joints": bounded.tolist()})["pose"]
                        measured = bounded.copy()
                        measured[index] += sign * 0.001
                        live.qpos[addresses] = measured
                        native.sim.forward()
                        saved = (live.qpos.copy(), live.qvel.copy(), live.ctrl.copy(), live.time)
                        reading = call(name, "read")
                        np.testing.assert_array_equal(reading["positions"], measured)
                        assert np.isfinite(reading["flange_pose"]).all()
                        solved = local(name, "ik", {"pose": target})["joints"]
                        np.testing.assert_allclose(solved, bounded, atol=1e-10)
                        for actual, previous in zip(
                            (live.qpos, live.qvel, live.ctrl), saved[:3], strict=True
                        ):
                            np.testing.assert_array_equal(actual, previous)
                        assert live.time == saved[3]
                live.qpos[:] = saved_qpos
                native.sim.forward()
        # Repeated absolute writes advance time without restarting the controller.
        from unittest.mock import patch

        arm = native._robot.part_controllers["right"]
        goal = np.array(service.describe("arm")["intrinsics"]["home_joints"])
        goal[0] += 0.1
        before = native.sim.data.time
        with patch.object(arm, "set_goal", wraps=arm.set_goal) as set_goal:
            for _ in range(60):
                call("arm", "write", {"target": goal.tolist()})
            assert set_goal.call_count == 1
        assert abs(native.sim.data.time - before - 3) < 1e-9
        state = call("arm", "read")
        np.testing.assert_allclose(state["positions"], goal, atol=0.01)
        assert max(abs(v) for v in state["velocities"]) < 0.01
        assert state["sample_time_s"] == native.sim.data.time
        replacement = goal.copy()
        replacement[0] -= 0.2
        call("arm", "write", {"target": replacement.tolist()})
        call("arm", "write", {"target": goal.tolist()})
        np.testing.assert_array_equal(arm.goal_qpos, goal)
        # A later base command preserves the previous gripper command.
        call("gripper", "write", {"target": [1]})
        fingers = native._robot.gripper["right"].current_action.copy()
        call("body", "write", {"target": [0, 0, 0]})
        increment = np.array([-1, 1]) * native._robot.gripper["right"].speed
        np.testing.assert_allclose(native._robot.gripper["right"].current_action,
                                   np.clip(fingers + increment, -1, 1))
        for action in ("pick", "move", "invoke", "policy_run"):
            try:
                call("arm", action)
            except DeviceError:
                pass
            else:
                raise AssertionError("Task macro exposed")
        composite_spec = BindingSpec.parse(
            "controller",
            {
                "binding": "robocasa.device/1",
                "entity": "robot",
                "profile": "composite.yaml",
            },
        )
        try:
            default_registry().assemble(
                replace(plan, devices=(*plan.devices, composite_spec)),
                robot,
                BindingContext("robocasa", "panda_omron", adapter),
            )
        except ValueError as exc:
            assert "overlapping" in str(exc)
        else:
            raise AssertionError("Competing composite and part controls accepted")
        composite = default_registry().assemble(
            replace(plan, devices=(composite_spec,)),
            robot,
            BindingContext("robocasa", "panda_omron", adapter),
        )
        try:
            profile = composite.describe("controller")
            target = [0.0] * len(profile["intrinsics"]["actions"])
            for part, name in (("right", "arm"), ("torso", "torso")):
                start, end = profile["intrinsics"]["parts"][part]
                target[start:end] = service.describe(name)["intrinsics"]["home_joints"]
            before = native._step_count
            result = composite.call(
                {"device": "controller", "action": "write", "args": {"target": target}}
            )
            assert result["data"] == {"accepted": True}
            assert native._step_count == before + 1
        finally:
            composite.close()
        print(
            json.dumps(
                {
                    "passed": True,
                    "physics": "PandaOmron in robosuite Lift",
                    "camera": "synthetic pixels",
                    "gpu_replay": False,
                }
            )
        )
    finally:
        service.close()
        adapter.close()


if __name__ == "__main__":
    import sys

    if "--task" in sys.argv[1:]:
        from roboenv_runtime.core.adapter import _REGISTRY
        from roboenv_runtime.runtime.worker import main

        _REGISTRY["robocasa"] = lambda **kwargs: make_adapter(
            realtime=kwargs.get("realtime", False)
        )
        raise SystemExit(main())
    check(Path(sys.argv[1]), Path(sys.argv[2]))
