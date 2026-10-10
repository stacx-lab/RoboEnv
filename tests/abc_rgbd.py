"""Analytic RGB-D fixtures for both ABC renderers; run with isolated ABC Python."""

import argparse
import json
from pathlib import Path

import numpy as np
from roboenv_runtime.adapters.abc.backend import ABCEnv
from roboenv_runtime.adapters.abc.rendering import mujoco_rgbd, settings
from roboenv_runtime.bindings.backends.shared.depth import write_depth


def probe(renderer, out):
    from roboenv.provisioning.gpu import configure_gpu

    configure_gpu("abc")
    import mujoco
    from abc_sim.rendering.blender.live import BlenderCameraProvider

    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <asset><material name="count_item_red" rgba="1 .01 .01 1"/></asset>
      <worldbody>
        <geom name="backplate" type="box" pos="0 0 .2" size="2 2 .01"/>
        <geom name="target" type="box" pos=".08 .04 .4" size=".03 .03 .02"
          material="count_item_red"/>
        <body name="moving"><joint type="slide" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <camera name="top" pos="0 0 1" fovy="60"/>
        </body>
        <camera name="sky" pos="0 0 3" quat="0 1 0 0"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    width, height = 300, 200
    native = None
    if renderer == "blender":
        options = {k: v for k, v in settings().items() if k != "identity"}
        native = BlenderCameraProvider(
            model, data, width=width, height=height, camera_names=("top", "sky"),
            depth=True, **options,
        )
    else:
        native = mujoco.Renderer(model, height=height, width=width)
    errors, alignment = [], []
    try:
        for step in range(2):
            data.qpos[0] = step * .06
            mujoco.mj_forward(model, data)
            signature = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(model, signature))
            mujoco.mj_getState(model, data, before, signature)
            if renderer == "blender":
                pairs = native.initial_frames() if not step else native.frames_for_step(step, 0)
            else:
                pairs = {name: mujoco_rgbd(native, model, data, name) for name in ("top", "sky")}
            after = np.empty_like(before)
            mujoco.mj_getState(model, data, after, signature)
            np.testing.assert_array_equal(after, before)
            rgb, depth = pairs["top"]
            assert rgb.shape == (height, width, 3) and rgb.dtype == np.uint8
            assert depth.shape == (height, width) and depth.dtype == np.float32
            assert np.isfinite(depth).all() and (depth >= 0).all()
            # Frontoparallel plate checks center and all corners: radial depth fails here.
            samples = depth[[10, 10, height-11, height-11, height//2],
                            [10, width-11, 10, width-11, width//2]]
            errors.extend(np.abs(samples - .79).tolist())
            assert max(errors) < .002, samples
            focal = height / (2 * np.tan(np.deg2rad(60) / 2))
            center = [(width-1)/2 + focal*(.08-step*.06)/.58,
                      (height-1)/2 - focal*.04/.58]
            x, y = np.rint(center).astype(int)
            assert abs(float(depth[y, x]) - .58) < .002
            red, green, blue = rgb.astype(float).transpose(2, 0, 1)
            yy, xx = np.where((red > 1.5*green) & (red > 1.5*blue))
            dy, dx = np.where((depth > 0) & (depth < .65))
            assert len(xx) > 20 and len(dx) > 20
            error = float(np.max(np.abs([xx.mean()-dx.mean(), yy.mean()-dy.mean()])))
            alignment.append(error)
            assert error < 2, error
            assert np.count_nonzero(pairs["sky"][1]) == 0
            write_depth(out / f"{renderer}-{step}.pgm", depth)
        return {"passed": True, "max_plane_error_m": max(errors),
                "max_alignment_error_px": max(alignment), "moving_camera": True,
                "background_invalid": True, "physics_unchanged": True}
    finally:
        native.close()


def scene(renderer, task):
    import mujoco

    world = ABCEnv(task, camera_backend=renderer)
    try:
        world.reset(seed=0)
        def snapshot():
            sig = mujoco.mjtState.mjSTATE_INTEGRATION
            result = np.empty(mujoco.mj_stateSize(world.env.model, sig))
            mujoco.mj_getState(world.env.model, world.env.data, result, sig)
            return result

        initial = snapshot()
        before = initial
        first = {}
        for name in world._camera_names:
            color, depth = world.device_image("camera_" + name)
            assert color.shape == (480, 640, 3) and depth.shape == (480, 640)
            assert color.dtype == np.uint8 and depth.dtype == np.float32
            assert np.isfinite(depth).all() and (depth >= 0).all() and (depth > 0).any()
            first[name] = (color, depth)
            pair = world.device_image("camera_" + name)
            for a, b in zip(pair, first[name], strict=True):
                np.testing.assert_array_equal(a, b)
                assert not np.shares_memory(a, b)
        np.testing.assert_array_equal(snapshot(), before)
        assert world.control.frame == 0
        target = world.device_read("left_arm")["positions"]
        target[0] += .08
        world.device_write("left_arm", target)
        for _ in range(8):
            world.control.tick()
        before = snapshot()
        color, depth = world.device_image("camera_left")
        assert not np.array_equal(depth, first["left"][1])
        np.testing.assert_array_equal(snapshot(), before)
        previous_renderer = world.renderer
        world.reset(seed=0)
        assert world.control.frame == 0
        np.testing.assert_array_equal(snapshot(), initial)
        _, restarted = world.device_image("camera_left")
        assert world.renderer is not previous_renderer
        depth = first["left"][1]
        # Recreated raster contexts can choose different surfaces at silhouettes.
        # Compare smooth interiors; fixture edges have a separate 2 px alignment gate.
        interior = np.ones(depth.shape, dtype=bool)
        for image in (depth, restarted):
            windows = np.lib.stride_tricks.sliding_window_view(image, (3, 3))
            interior[1:-1, 1:-1] &= np.ptp(windows, axis=(-2, -1)) < .002
            interior &= image > 0
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        assert interior.sum() > depth.size // 10
        np.testing.assert_allclose(restarted[interior], depth[interior], atol=.002, rtol=0)
        np.testing.assert_array_equal(snapshot(), initial)
        world.reset(seed=1)
        assert world.control.frame == 0
        assert world.device_image("camera_top")[1].shape == (480, 640)
        return {"passed": True, "three_cameras": True, "cache": True,
                "motion": True, "reset": True, "physics_unchanged": True}
    finally:
        world.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("mujoco", "blender"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"fixture": probe(args.renderer, args.out)}
    for task in ("put_plastic_bottles_in_bin", "multi_drawer_search"):
        result[task] = scene(args.renderer, task)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
