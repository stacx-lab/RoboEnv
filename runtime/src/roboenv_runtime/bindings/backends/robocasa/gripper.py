"""Typed gripper facts; native command dimensions differ from measured joints."""

from roboenv_runtime.bindings.motion import control_binding


def intrinsics():
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "aperture_m": None,
        "control": {
            "mode": "direction",
            "open_target": -1,
            "close_target": 1,
        },
        "feedback": {"aperture_m": True},
        "stop": {"behavior": "hold", "affected_devices": []},
    }


@control_binding
def bind(spec, root, context):
    import numpy as np

    from roboenv_runtime.adapters.robocasa.geometry import vector

    from .common import action, finish, target_schema

    definition = spec.definition(root)
    if definition.type != "gripper" or spec.config != {"part": "right_gripper"}:
        raise ValueError("gripper binding requires type gripper and native right_gripper part")
    world = context.native._env
    robot, sim = world._robot, world.sim
    model, data = sim.model._model, sim.data._data
    joints = np.asarray(robot.part_controllers["right_gripper"].joint_index, dtype=int)
    from .gripper_geometry import geometry

    facts = intrinsics()
    shape, aperture = geometry(robot.gripper["right"], sim, joints)
    facts.update(shape)
    facts["stop"]["affected_devices"] = [definition.name]

    def read():
        return {
            "faults": [],
            "timestamp": float(data.time),
            "aperture_m": max(
                0.0,
                aperture["offset_m"]
                + float(np.dot(aperture["coefficients"], data.qpos[model.jnt_qposadr[joints]])),
            ),
        }

    def write(target):
        command = np.sign(vector(target, 1)).tolist()
        world.step({"right_gripper": command})
        return {}

    def stop():
        world.stop_part("right_gripper")
        return {"stopped": True, **facts["stop"]}

    return finish(
        spec,
        definition,
        context,
        facts,
        {
            "read": action("read", read),
            "write": action(
                "write", write, target_schema([{"limits": {"min": -1, "max": 1}}]), mutating=True
            ),
            "stop": action("stop", stop, mutating=True),
        },
    )
