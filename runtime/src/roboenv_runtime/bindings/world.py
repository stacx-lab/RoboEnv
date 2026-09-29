"""Public world clock, independent of backend observation policies."""

from roboenv.devices.model import Action

from roboenv_runtime.bindings.motion import object_schema

# Native counts that one control interval converts to episode seconds. An
# `api_calls` or unknown unit has no time value, so no budget is reported.
TIMED_BUDGET_UNITS = frozenset({"action_frames", "sim_steps"})


def budget_seconds(native):
    """The episode budget as `{elapsed_s, max_s}` in seconds, or None when untimed.

    This is the informational shape Kinex reads on world results; `max_s` is
    None for an uncapped episode. The owner-facing frame budget is unchanged.
    """
    budget = native.budget()
    if budget.unit not in TIMED_BUDGET_UNITS:
        return None
    interval = native.control_interval_s
    return {
        "elapsed_s": budget.used * interval,
        "max_s": None if budget.max is None else budget.max * interval,
    }


def clock_actions(context):
    empty = object_schema({}, [])

    def world():
        if context.world is None:
            raise RuntimeError("world clock is not bound")
        return context.world

    def with_budget(data):
        budget = budget_seconds(world().native)
        return data if budget is None else {**data, "budget": budget}

    return {
        "read": Action(lambda args, execution: with_budget({"time_s": world().time_s}), empty),
        "wait": Action(
            lambda args, execution: world().wait(args["duration_s"]),
            object_schema({"duration_s": {"type": "number", "minimum": 0}}, ["duration_s"]),
        ),
        "load": Action(lambda args, execution: {"loaded": True}, empty),
        "unload": Action(lambda args, execution: {"unloaded": True}, empty),
        "status": Action(lambda args, execution: with_budget({"loaded": True}), empty),
    }
