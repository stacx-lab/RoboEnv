"""Agent-independent environment access. No agent or simulator imports at startup."""

__version__ = "0.10.3"


def connect(*args, **kwargs):
    """Connect to an environment without owning its lifetime."""
    from roboenv.environment import connect as attach

    return attach(*args, **kwargs)


def launch(task, **kwargs):
    """Launch an environment; use the returned owner as a context manager."""
    from roboenv.environment import launch as start

    return start(task, **kwargs)
