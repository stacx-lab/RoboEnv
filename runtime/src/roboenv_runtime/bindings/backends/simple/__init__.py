"""Control-contract bindings over the existing SIMPLE world."""

from functools import partial


def register(registry):
    from .devices import bind

    for kind in ("velocity", "joint", "camera", "world"):
        registry.register(f"simple.{kind}/1", partial(bind, kind=kind))
