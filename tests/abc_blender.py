"""Native Blender projection, lifecycle and unchanged-physics checks; isolated ABC Python."""

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from roboenv_runtime.adapters.abc.backend import ABCEnv
from roboenv_runtime.adapters.abc.rendering import settings


def state(model, data):
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(model, signature))
    mujoco.mj_getState(model, data, result, signature)
    return result


def projection(out):
    from abc_sim.rendering.blender.live import BlenderCameraProvider

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <asset><material name="count_item_red" rgba="1 0.01 0.01 1"/></asset>
      <worldbody><geom name="target" type="sphere" pos=".08 .04 0" size=".025"
        material="count_item_red"/>
        <body name="moving"><joint type="slide" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <camera name="top" pos="0 0 1" fovy="60"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    options = {k: v for k, v in settings(0).items() if k != "identity"}
    views = [
        dict(name="top", source="top", width=300, height=200, fov_y=60),
        dict(
            name="fixed",
            width=400,
            height=300,
            fov_y=45,
            matrix_pose=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        ),
    ]
    provider = BlenderCameraProvider(
        model, data, width=300, height=200, camera_names=("top",), views=views, **options
    )
    process = provider._proc
    errors = []
    try:
        for step in range(2):
            if step:
                data.qpos[0] = 0.06
                mujoco.mj_forward(model, data)
            before = state(model, data)
            frames = provider.initial_frames() if not step else provider.frames_for_step(step, 0)
            np.testing.assert_array_equal(state(model, data), before)
            for view in views:
                frame = frames[view["name"]]
                iio.imwrite(out / f"projection-{view['name']}-{step}.png", frame)
                red, green, blue = frame.astype(float).transpose(2, 0, 1)
                yy, xx = np.where((red > 1.5 * green) & (red > 1.5 * blue))
                assert len(xx) > 10
                f = view["height"] / (2 * np.tan(np.deg2rad(view["fov_y"]) / 2))
                x = 0.08 - (0.06 if step and view["name"] == "top" else 0)
                expected = [view["width"] / 2 + f * x, view["height"] / 2 - f * 0.04]
                error = float(np.max(np.abs(np.array([xx.mean(), yy.mean()]) - expected)))
                assert error < 2, (view["name"], error)
                errors.append(error)
        # A renderer dying mid-episode must fail explicitly and release its process.
        process.kill()
        process.wait(timeout=5)
        try:
            provider.frames_for_step(3, 0)
        except RuntimeError as exc:
            assert "Blender render failed" in str(exc)
        else:
            raise AssertionError("dead Blender accepted a render")
    finally:
        provider.close()
    assert process.poll() is not None
    # Bounded startup of a binary which never opens its socket.
    sleeper = out / "sleeping-renderer"
    sleeper.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
    sleeper.chmod(0o755)
    started = time.monotonic()
    try:
        BlenderCameraProvider(
            model,
            data,
            width=300,
            height=200,
            camera_names=("top",),
            **{**options, "blender": str(sleeper)},
            startup_timeout=0.2,
            shutdown_timeout=0.2,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("unresponsive renderer did not time out")
    assert time.monotonic() - started < 10
    return {
        "max_projection_error_px": max(errors),
        "moving_camera": True,
        "failed_render_cleanup": True,
        "startup_timeout": True,
    }


def scene(task, out):
    started = time.monotonic()
    world = ABCEnv(task, camera_backend="blender")
    reference = None
    try:
        world.reset(seed=0)
        startup = time.monotonic() - started
        before = state(world.env.model, world.env.data)
        reference = mujoco.Renderer(world.env.model, height=480, width=640)
        for camera in world._camera_names:
            frame, _ = world.device_image("camera_" + camera)
            assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
            iio.imwrite(out / f"{task}-blender-{camera}.png", frame)
            reference.update_scene(world.env.data, camera=camera)
            iio.imwrite(out / f"{task}-mujoco-{camera}.png", reference.render())
        np.testing.assert_array_equal(state(world.env.model, world.env.data), before)
        assert world.control.frame == 0 and world.frame_limit == 8823
        assert world.renderer.provider._calls == 1
        target = world.device_read("left_arm")["positions"]
        target[0] += 0.08
        world.device_write("left_arm", target)
        for _ in range(8):
            world.control.tick()
        before = state(world.env.model, world.env.data)
        started = time.monotonic()
        frame, _ = world.device_image("camera_left")
        seconds = time.monotonic() - started
        iio.imwrite(out / f"{task}-blender-left-moved.png", frame)
        world.device_image("camera_top")
        np.testing.assert_array_equal(state(world.env.model, world.env.data), before)
        assert world.renderer.provider._calls == 2
        token = world.checkpoint()
        old_process = world.renderer.provider._proc
        world.restore(token)
        assert old_process.poll() is not None
        np.testing.assert_array_equal(state(world.env.model, world.env.data), before)
        world.device_image("camera_left")
        assert world.renderer.frame == 8
        return {
            "startup_s": startup,
            "three_cameras_render_s": seconds,
            "physics_unchanged": True,
            "cache": True,
            "restore": True,
        }
    finally:
        if reference is not None:
            reference.close()
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"projection": projection(args.out)}
    for task in ("put_plastic_bottles_in_bin", "multi_drawer_search"):
        result[task] = scene(task, args.out)
    (args.out / "report.json").write_text(json.dumps({"passed": True, **result}, indent=2) + "\n")
    print(json.dumps(result))
