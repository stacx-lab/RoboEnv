"""Chassis facts and measured spatial velocity for the native Omron controller."""

import numpy as np

from roboenv_runtime.bindings.motion import control_binding


def intrinsics(world, pose_frame="world"):
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "control": {
            "mode": "motion",
            "limit_handling": "native",
            "body_frame": "base",
            "modes": {
                "drive": {
                    "velocity": {
                        "frame": "body",
                        "components": [
                            {"axis": axis, "units": "normalized", "bounds": None}
                            for axis in ("vx", "vy", "wz")
                        ],
                    },
                    "posture": None,
                }
            },
        },
        "stop": {"behavior": "halt", "affected_devices": ["base"]},
        "feedback": {
            "pose_frame": pose_frame,
            "velocity_frame": "body",
            "velocity_components": ["vx", "vy", "wz"],
            "clock": "simulation",
        },
    }


def reading(model, data, body):
    import mujoco

    # Object velocity is at the body origin, including its offset from the yaw
    # joint. Raw generalized qvel is neither body-frame velocity nor this point.
    spatial = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, spatial, 1)
    quat = data.xquat[body]
    return {
        "pose": {
            "translation_m": data.xpos[body].tolist(),
            "rotation_xyzw": [*quat[1:].tolist(), float(quat[0])],
        },
        "velocities": spatial[[3, 4, 2]].tolist(),
        "faults": [],
        "sample_time_s": float(data.time),
    }


@control_binding
def bind(spec, root, context):
    from roboenv.devices.velocity import command, command_schema

    from .common import action, finish

    definition = spec.definition(root)
    if definition.type != "velocity" or spec.config != {"part": "base"}:
        raise ValueError("velocity binding requires type velocity and native base part")
    world = context.native._env
    robot, sim = world._robot, world.sim
    model, data = sim.model._model, sim.data._data
    controller = robot.part_controllers["base"]
    if getattr(controller, "name", None) != "JOINT_VELOCITY":
        raise ValueError("velocity binding requires a native JOINT_VELOCITY controller")
    body = sim.model.body_name2id(robot.robot_model.base.root_body)
    facts = intrinsics(world, (context.devices or {}).get("world", "world"))
    facts["stop"]["affected_devices"] = [definition.name]

    def write(**args):

        value = command(facts["control"], args)
        world.step({"base": [value["velocity"][axis] for axis in ("vx", "vy", "wz")]})
        return {}

    def stop():
        world.stop_part("base")
        return {"stopped": True, **facts["stop"]}

    return finish(
        spec,
        definition,
        context,
        facts,
        {
            "read": action("read", lambda: reading(model, data, body)),
            "write": action("write", write, command_schema(facts["control"]), mutating=True),
            "stop": action("stop", stop, mutating=True),
        },
    )
