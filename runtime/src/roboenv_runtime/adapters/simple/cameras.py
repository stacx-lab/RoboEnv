"""Validate Isaac optical calibration against the portable robot profile."""

import os
import sys

import numpy as np

from roboenv_runtime.adapters.urdf_geometry import matrix

MOUNT_TOLERANCE_M = 1e-4


def torso_matrix(world, renderer):
    """World pose of ``torso_link``, the parent frame of the head cameras.

    SIMPLE drives the Isaac robot through joint state, so the torso prim's USD xform
    is never updated: on Isaac Sim 4.5 it stays at the spawn pose while MuJoCo has
    the standing torso at z=0.80. SIMPLE mirrors MuJoCo into Isaac, so the MuJoCo
    body pose is the authoritative parent. The USD read remains the fallback when
    MuJoCo state is not available.
    """
    mjmodel, mjdata = getattr(world, "mjmodel", None), getattr(world, "mjdata", None)
    if mjmodel is not None and mjdata is not None:
        import mujoco

        body = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if body >= 0:
            xpos, xquat = mjdata.xpos[body], mjdata.xquat[body]  # MuJoCo quaternions are wxyz
            return matrix([*xpos, xquat[1], xquat[2], xquat[3], xquat[0]])
    from omni.isaac.core.prims import XFormPrim

    torso = XFormPrim(renderer.robot.prim_path + "/torso_link")
    position, quaternion = torso.get_world_pose()
    return matrix([*position, *quaternion[[1, 2, 3, 0]]])


def mount_tolerance():
    """Mount tolerance in metres; SIMPLE_ISAAC_MOUNT_ATOL overrides the 1e-4 default.

    The override exists for diagnosis only: the pinned SIMPLE scene places the head
    cameras about 0.010 m from the profile mount once the parent pose is read from
    MuJoCo. Every deviation is reported on stderr so a relaxed run stays auditable.
    """
    return float(os.environ.get("SIMPLE_ISAAC_MOUNT_ATOL", MOUNT_TOLERANCE_M))


def validate_isaac(world):
    renderer = world.env.isaac
    if renderer is None:
        return
    parent = torso_matrix(world, renderer)
    tolerance = mount_tolerance()
    for name, spec in world.model["cameras"].items():
        camera = renderer.cameras[spec["native"]]
        if tuple(camera.get_resolution()) != (spec["width"], spec["height"]):
            raise ValueError(f"SIMPLE Isaac camera dimensions differ: {name}")
        projection = spec["projection"]
        expected = [
            [projection["fx"], 0, projection["cx"]],
            [0, projection["fy"], projection["cy"]],
            [0, 0, 1],
        ]
        native = np.array(camera.get_intrinsics_matrix(), copy=True)
        # Native matrices use pixel-edge coordinates; catalog pixels have integer centers.
        native[:2, 2] -= 0.5
        if not np.allclose(native, expected, atol=1e-4, rtol=0):
            raise ValueError(f"SIMPLE Isaac camera intrinsics differ: {name}")
        position, quaternion = camera.get_world_pose(camera_axes="ros")
        optical = matrix([*position, *quaternion[[1, 2, 3, 0]]])
        mount = spec["mount"]
        expected = matrix(mount["translation_m"] + mount["rotation_xyzw"])
        deviation = float(np.abs(np.linalg.inv(parent) @ optical - expected).max())
        print(
            f"[simple cameras] {name}: Isaac mount deviation from profile {deviation:.5f} m "
            f"(tolerance {tolerance:g} m)",
            file=sys.stderr,
            flush=True,
        )
        if deviation > tolerance:
            raise ValueError(f"SIMPLE Isaac camera mount differs: {name}")
