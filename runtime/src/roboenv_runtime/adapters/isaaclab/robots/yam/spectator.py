"""Look through the open front of ABC's three-walled tabletop enclosure."""

from ...common.spectator import SpectatorSource, look_at


def sources(environment):
    pose = look_at([-1.10, -0.50, 1.85], [0.50, 0, 1.05])
    return {"third_person": SpectatorSource(lambda: pose.copy())}
