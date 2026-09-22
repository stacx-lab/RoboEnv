"""Typed gripper facts; native command dimensions differ from measured joints."""


def intrinsics(robot, sim, joints):
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "aperture_m": None,
        "control": {
            "mode": "direction",
            "open_target": -1,
            "close_target": 1,
            "increment_per_interval": float(robot.gripper["right"].speed),
        },
        "feedback": {
            "positions": [{"name": sim.model.joint_id2name(int(j)), "units": "m"} for j in joints],
            "velocities": True,
            "sensors": [],
            "clock": "simulation",
            "aperture_model": None,
        },
        "stop": {"behavior": "hold", "affected_devices": []},
    }


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

    facts = intrinsics(robot, sim, joints)
    shape, aperture, frames = geometry(robot.gripper["right"], sim, joints)
    facts.update(shape)
    facts["feedback"]["aperture_model"] = aperture
    facts["stop"]["affected_devices"] = [definition.name]

    def read():
        return {
            "positions": data.qpos[model.jnt_qposadr[joints]].tolist(),
            "velocities": data.qvel[model.jnt_dofadr[joints]].tolist(),
            "faults": [],
            "sample_time_s": float(data.time),
            "aperture_m": max(
                0.0,
                aperture["offset_m"]
                + float(np.dot(aperture["coefficients"], data.qpos[model.jnt_qposadr[joints]])),
            ),
            "frame_poses": frames(),
            "force_n": None,
            "object_detected": None,
            "pressure_pa": None,
        }

    def write(target, for_s=None, arrive=None):
        """Drive the jaws for for_s of the world's own clock and report the time.

        A direction-controlled gripper has no position goal to arrive at, so it
        never reports reached; the caller's budget is what ends the segment.
        """
        command = np.sign(vector(target, 1, lower=[-1], upper=[1])).tolist()
        elapsed = world.step({"right_gripper": command}, for_s=for_s)
        return {"accepted": True, "elapsed_s": elapsed, "reached": False, "state": read()}

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
