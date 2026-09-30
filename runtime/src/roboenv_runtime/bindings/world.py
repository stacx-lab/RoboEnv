"""Public world clock, independent of backend observation policies."""

from roboenv.devices.model import Action

from roboenv_runtime.bindings.motion import object_schema


def clock_actions(context):
    empty = object_schema({}, [])

    def world():
        if context.world is None:
            raise RuntimeError("world clock is not bound")
        return context.world

    def read(args, execution):
        session = world()
        data = {"time_s": session.time_s}
        budget = session.native.budget()
        # Only action frames have a known duration of one native control interval.
        # Budget consumption may include initialization before the world clock starts.
        if budget.unit == "action_frames":
            interval = session.native.control_interval_s
            data["budget"] = {
                "elapsed_s": budget.used * interval,
                "max_s": None if budget.max is None else budget.max * interval,
            }
        return data

    return {
        "read": Action(read, empty),
        "wait": Action(
            lambda args, execution: world().wait(args["duration_s"]),
            object_schema({"duration_s": {"type": "number", "minimum": 0}}, ["duration_s"]),
        ),
        "load": Action(lambda args, execution: {"loaded": True}, empty),
        "unload": Action(lambda args, execution: {"unloaded": True}, empty),
        "status": Action(lambda args, execution: {"loaded": True}, empty),
    }
