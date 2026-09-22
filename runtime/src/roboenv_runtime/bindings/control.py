"""Bounded writes for component controllers with named command arguments."""

from roboenv.devices.execution import map_result


def write_segment(world, name, *, for_s=None, arrive=None, **command):
    def arrived():
        tolerance = (arrive or {}).get("joint_tolerance")
        target = command.get("target")
        if not tolerance or target is None:
            return False
        state = world.read_device(name)
        positions = state.get("positions") or []
        if len(positions) != len(target) or len(tolerance) != len(target):
            return False
        if any(abs(p - t) > limit for p, t, limit in zip(positions, target, tolerance)):
            return False
        quiet = (arrive or {}).get("quiet_velocity")
        if quiet is None:
            return True
        velocities = state.get("velocities") or []
        return len(velocities) == len(quiet) and all(
            abs(v) <= limit for v, limit in zip(velocities, quiet)
        )

    outcome = world.control.write(name, command, for_s=for_s, arrived=arrived)
    return map_result(outcome, lambda value: {**value, "reached": arrived(), "state": world.read_device(name)})
