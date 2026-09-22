"""Joint-position devices extracted from the compiled model, in physical units."""


def intrinsics(sim, indexes, chain, home, name):
    model = sim.model._model
    joints = [
        {
            "name": sim.model.joint_id2name(int(index)),
            "type": "prismatic"
            if model.jnt_type[index] == 2
            else "revolute"
            if model.jnt_limited[index]
            else "continuous",
            "limits": {
                "min": float(model.jnt_range[index, 0]),
                "max": float(model.jnt_range[index, 1]),
            }
            if model.jnt_limited[index]
            else None,
            "max_velocity": None,
            "max_effort": None,
        }
        for index in indexes
    ]
    return {
        "source": "introspected",
        "method": "introspection",
        "calibrated_at": None,
        "model": {"format": "serial_chain", "joints": joints, "chain": chain},
        "kinematics": {
            "frame": "flange",
            "fk": "numerical",
            "feedback": "driver",
            "frame_to_reported": [0, 0, 0, 0, 0, 0, 1],
            "ik": {
                "backend": "numerical",
                "method": "damped_least_squares",
                "position_tolerance_m": 0.0001,
                "rotation_tolerance_rad": 0.001,
                "max_iterations": 200,
                "damping": 0.01,
            },
        },
        "home": {joint["name"]: float(value) for joint, value in zip(joints, home, strict=True)},
        "control": {
            "mode": "position",
            "speed": None,
        },
        "feedback": {"velocities": True, "efforts": False, "clock": "simulation"},
        "stop": {"behavior": "hold", "affected_devices": [name]},
    }


def bind(spec, root, context):
    import numpy as np

    from roboenv_runtime.adapters.robocasa.geometry import chain, pose, vector

    from .common import action, finish, target_schema

    definition = spec.definition(root)
    if definition.type != "joint" or set(spec.config) != {"part"}:
        raise ValueError("joint binding requires type joint and native part")
    world = context.native._env
    robot, sim = world._robot, world.sim
    part = spec.config["part"]
    if part not in {"right", "torso"}:
        raise ValueError("unsupported native joint controller part")
    controller = robot.part_controllers[part]
    if getattr(controller, "name", None) != "JOINT_POSITION":
        raise ValueError("joint binding requires a native JOINT_POSITION controller")
    model, data = sim.model._model, sim.data._data
    indexes = np.asarray(controller.joint_index, dtype=int)
    arm_base = int(
        model.body_parentid[model.jnt_bodyid[robot.part_controllers["right"].joint_index[0]]]
    )
    base = arm_base if part == "right" else sim.model.body_name2id(robot.robot_model.base.root_body)
    tip = (
        sim.model.body_name2id(robot.robot_model.eef_name["right"]) if part == "right" else arm_base
    )
    home = robot.init_qpos if part == "right" else robot.init_torso_qpos
    if home is None:
        home = robot.robot_model.init_torso_qpos
    facts = intrinsics(
        sim, indexes, chain(model, indexes, tip, base)["chain"], home, definition.name
    )
    frame = "flange" if part == "right" else "top"
    facts["kinematics"]["frame"] = frame
    facts["frames"] = {frame: None}
    limits = [joint["limits"] for joint in facts["model"]["joints"]]

    def read():
        return {
            "positions": data.qpos[model.jnt_qposadr[indexes]].tolist(),
            "velocities": data.qvel[model.jnt_dofadr[indexes]].tolist(),
            "efforts": [],
            "faults": [],
            "sample_time_s": float(data.time),
            "frame_poses": {frame: pose(model, data, tip, base)},
        }

    def write(target, for_s=None, arrive=None):
        """Set this device's goal and execute it; report the time spent.

        This layer judges arrival because it knows which joints it owns; the
        world owns the clock. `reached` is a measurement, never a promise.
        """
        command = vector(
            target,
            len(indexes),
            lower=[v["min"] if v else -np.inf for v in limits],
            upper=[v["max"] if v else np.inf for v in limits],
        )

        def arrived():
            tolerance = (arrive or {}).get("joint_tolerance")
            if not tolerance:
                return False
            measured = np.asarray(data.qpos[model.jnt_qposadr[indexes]], dtype=float)
            if not np.all(np.abs(measured - np.asarray(command, dtype=float))
                          <= np.asarray(tolerance, dtype=float)):
                return False
            quiet = (arrive or {}).get("quiet_velocity")
            if quiet is None:
                return True
            speeds = np.asarray(data.qvel[model.jnt_dofadr[indexes]], dtype=float)
            return bool(np.all(np.abs(speeds) <= np.asarray(quiet, dtype=float)))

        elapsed = world.step({part: command}, for_s=for_s, arrived=arrived)
        return {"accepted": True, "elapsed_s": elapsed, "reached": arrived(), "state": read()}

    def stop():
        world.stop_part(part)
        return {"stopped": True, **facts["stop"]}

    return finish(
        spec,
        definition,
        context,
        facts,
        {
            "read": action("read", read),
            "write": action("write", write, target_schema(facts["model"]["joints"]), mutating=True),
            "stop": action("stop", stop, mutating=True),
        },
    )
