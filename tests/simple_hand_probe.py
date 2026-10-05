"""Compare portable hand FK to native transforms; no policy or task solving."""

import argparse
import json
import os
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from roboenv_runtime.adapters.simple.profiles import profiles
from roboenv_runtime.adapters.urdf_geometry import matrix, matrix_pose

ROOT = Path(__file__).resolve().parents[1]


def frame_poses(world, name):
    part = world.model["parts"][name]
    geometry = part["geometry"]
    parent = np.linalg.inv(matrix(world._pose(part["base"])))
    frames = {
        finger["tip_frame"]: matrix_pose(
            parent @ matrix(world._pose(part["frame_bodies"][finger["tip_frame"]]))
            @ matrix(finger["tip_offset"])
        )
        for finger in geometry["fingers"].values()
    }
    center = geometry.get("center")
    if center is None:
        return frames
    frames[center["frame"]] = [
        sum(weight * frames[geometry["fingers"][finger]["tip_frame"]][axis]
            for finger, weight in center["weights"].items())
        for axis in range(3)
    ] + center["rotation_xyzw"]
    return frames


@cache
def predictor(kinex):
    sys.path.insert(0, str(kinex / "packages/core/assets/python"))
    from kinex.hand import forward

    return forward


def compare(world, kinex):
    forward = predictor(kinex)
    errors = {}
    for name in ("left_hand", "right_hand"):
        ids = [world.mjmodel.joint(j["name"]).id for j in world.model["parts"][name]["joints"]]
        q = world.mjdata.qpos[world.mjmodel.jnt_qposadr[ids]].tolist()
        predicted = forward(profiles(world.model)[name], q)
        actual = frame_poses(world, name)
        assert actual.keys() == predicted.keys()
        errors[name] = max(float(np.max(np.abs(matrix(actual[key]) - matrix(predicted[key]))))
                           for key in actual)
    assert max(errors.values()) < 1e-6, errors
    return errors


def check(out, kinex):
    import mujoco

    forward = predictor(kinex)
    facts = json.loads((ROOT / "runtime/src/roboenv_runtime/adapters/simple/g1.json").read_text())
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / ".cache/simple/data/robots/g1_sonic/g1_29dof_with_hand.xml"))
    data = mujoco.MjData(model)

    def body_pose(name):
        index = model.body(name).id
        return [*data.xpos[index], *data.xquat[index][[1, 2, 3, 0]]]

    world = SimpleNamespace(model=facts, mjmodel=model, mjdata=data, _pose=body_pose)
    rng = np.random.default_rng(904)
    errors, samples = [], 0
    for name in ("left_hand", "right_hand"):
        part = facts["parts"][name]
        ids = [model.joint(j["name"]).id for j in part["joints"]]
        addresses = model.jnt_qposadr[ids]
        lower, upper = model.jnt_range[ids].T
        configurations = [np.zeros(7), lower, upper,
                          *(rng.uniform(lower, upper) for _ in range(40))]
        configurations += [np.eye(7)[i] * (upper[i] if upper[i] > 0 else lower[i]) * .3
                           for i in range(7)]
        configurations.append(upper + .002)  # Measured soft-limit overshoot is not clipped.
        reference = profiles(facts)[name]["intrinsics"]["frames"]["grasp_reference"]
        home = forward(profiles(facts)[name], [0] * 7)["grasp_center"]
        np.testing.assert_allclose(
            home, [*reference["translation_m"], *reference["rotation_xyzw"]], atol=1e-11
        )
        for q in configurations:
            data.qpos[:] = model.qpos0
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            before = data.qpos.copy()
            stamp = data.time
            errors.append(max(compare(world, kinex).values()))
            np.testing.assert_array_equal(data.qpos, before)
            assert data.time == stamp
            samples += 1
        for finger in part["geometry"]["fingers"].values():
            assert np.linalg.norm(finger["tip_offset"][:3]) > .05
        for frame, source in part["geometry_sources"].items():
            finger = next(f for f in part["geometry"]["fingers"].values()
                          if f["tip_frame"] == frame)
            axis = source["distal_axis"]
            bound = source["bounds_m"][1 if source["distal_sign"] > 0 else 0][axis]
            assert abs(finger["tip_offset"][axis] - bound) < 1e-11
    report = {"passed": True, "configurations": samples, "max_transform_error": max(errors)}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kinex", type=Path,
        default=Path(os.environ.get("KINEX_ROOT", ROOT.parent / "kinex")),
    )
    args = parser.parse_args()
    check(args.out, args.kinex.resolve())
