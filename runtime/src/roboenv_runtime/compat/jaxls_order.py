"""Make jaxls order its cost groups by creation order, not by memory address.

``jaxls._problem._get_function_signature`` keys a cost group by
``(bytecode, closure_vars, id(instance))`` and ``analyze()`` then sorts the
groups with ``key=str`` — so the *order* in which residual groups are stacked,
and therefore the floating-point summation order of every IK solve, depends on
where Python happened to allocate the cost objects.

Measured 2026-08-28 on LIBERO: with every tool the legacy one and nothing else
changed, allocating 1, 5 or 300 inert closures before a certificate program
reproduced it (reward 1.0, 1713 steps) and allocating 40 did not (reward 0.0,
3867 steps) — a marginal grasp flipped by last-bit differences in the IK
solution.  It is why certificates replay 69 of 69 on the tree that earned
them and lose a few percent in a clean rebuild, and it is not a property of
any simulator.

This replaces the address with a per-instance ordinal assigned in creation
order, which is deterministic for a deterministic program.  Grouping identity
is unchanged (distinct instances stay distinct); only the ordering stops
depending on the allocator.
"""

from __future__ import annotations

import itertools
import re
import weakref
from typing import Any

__all__ = ["make_ik_deterministic", "make_jaxls_deterministic", "pin_xla_gpu"]

#: XLA's GPU autotuner picks GEMM algorithms per buffer alignment, so a change of memory
#: layout changes the last bits of an IK solve.  Measured 2026-08-29 on seven LIBERO
#: certificates that still flipped between two layouts after the jaxls order was pinned:
#: with these flags (or with IK on CPU) all seven are layout-independent and equal to
#: their certificates.
XLA_DETERMINISTIC_FLAGS = ("--xla_gpu_autotune_level=0", "--xla_gpu_deterministic_ops=true")


def pin_xla_gpu() -> None:
    """Add the deterministic GPU flags to ``XLA_FLAGS`` (idempotent; before jax imports)."""
    import os

    current = os.environ.get("XLA_FLAGS", "")
    missing = [f for f in XLA_DETERMINISTIC_FLAGS if f not in current]
    if missing:
        os.environ["XLA_FLAGS"] = (current + " " + " ".join(missing)).strip()


def make_ik_deterministic() -> bool:
    """Both halves: the jaxls group order and the XLA GPU flags."""
    pin_xla_gpu()
    return make_jaxls_deterministic()

_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


class _StableKey(tuple):
    """A group key that compares by content but *sorts* without memory addresses.

    ``closure_vars`` are ``str(cell_contents)`` and an object without a custom
    ``__repr__`` prints its address, so the key's string — which is what jaxls
    sorts by — carried addresses even after the instance id was replaced.
    Grouping (``__eq__``/``__hash__``) is the tuple's own; only ``__str__`` changes.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return _ADDRESS.sub("0x0", tuple.__repr__(self))

    __repr__ = __str__


def make_jaxls_deterministic() -> bool:
    """Install the patch once; ``False`` when jaxls is not importable here."""
    try:
        from jaxls import _problem as problem
    except Exception:  # not this interpreter's business
        return False
    if getattr(problem, "_roboenv_stable_order", False):
        return True
    original = problem._get_function_signature
    counter = itertools.count(1)
    weak: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
    strong: dict[int, tuple[Any, int]] = {}  # instances that cannot be weakly referenced

    def ordinal_of(instance: Any) -> int:
        try:
            found = weak.get(instance)
            if found is None:
                found = next(counter)
                weak[instance] = found
            return found
        except TypeError:
            entry = strong.get(id(instance))
            if entry is None or entry[0] is not instance:
                entry = (instance, next(counter))
                strong[id(instance)] = entry
            return entry[1]

    def stable_signature(func: Any) -> Any:
        bytecode, closure_vars, instance_id = original(func)
        if instance_id is None:
            return _StableKey((bytecode, closure_vars, None))
        instance = getattr(func, "__self__", None)
        return _StableKey((bytecode, closure_vars, ordinal_of(instance)))

    problem._get_function_signature = stable_signature
    problem._roboenv_stable_order = True
    return True
