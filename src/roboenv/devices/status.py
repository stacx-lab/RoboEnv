"""Operational simulation feedback without the final task judgment."""


def status_feedback(status):
    if status is None:
        return None
    return {key: value for key, value in status.items() if key in {
        "schema", "session", "generation", "state", "terminal", "budget",
        "simulation_time_s", "capabilities", "service_state",
    }}
