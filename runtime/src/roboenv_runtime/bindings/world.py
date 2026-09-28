"""Public world clock, independent of backend observation policies."""

from roboenv.devices.model import Action
from roboenv_runtime.bindings.motion import object_schema


def clock_actions(context):
    empty = object_schema({}, [])

    def world():
        if context.world is None:
            raise RuntimeError("world clock is not bound")
        return context.world

    return {
        "read": Action(lambda args, execution: {"time_s": world().time_s}, empty),
        "wait": Action(
            lambda args, execution: world().wait(args["duration_s"]),
            object_schema({"duration_s": {"type": "number", "minimum": 0}}, ["duration_s"]),
        ),
        "load": Action(lambda args, execution: {"loaded": True}, empty),
        "unload": Action(lambda args, execution: {"unloaded": True}, empty),
        "status": Action(lambda args, execution: {"loaded": True}, empty),
    }
