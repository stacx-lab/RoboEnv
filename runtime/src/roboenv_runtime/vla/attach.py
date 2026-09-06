"""Attaching ``vla_run`` to an adapter instance.

The tool is the same for every simulator — ``vla_run(n_actions, instruction=None,
policy=<default>)`` running :func:`roboenv_runtime.vla.runner.run_policy` — so it is built
here, from the policy spec, and attached by :func:`roboenv_runtime.core.adapter.make_adapter`
to any adapter whose simulator has a registered hooks factory.  Adapters stay
untouched: they only provide the hooks (observation, one control step, budget)
in a module of their own.  The tool's docstring, the part the model reads,
comes from the spec's ``tool_doc`` so it describes *that* policy's strengths.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

from roboenv_runtime.core.types import Capability
from roboenv_runtime.vla.spec import PolicySpec, list_policies, load_policy_spec

__all__ = ["HOOKS", "attach_vla_tool", "build_vla_tool", "register_hooks"]

log = logging.getLogger(__name__)

#: simulator name -> "package.module:callable(adapter, spec, instruction) -> VlaHooks"
HOOKS: dict[str, str] = {
    "libero": "roboenv_runtime.adapters.libero.vla:LiberoPolicyHooks",
    "robocasa": "roboenv_runtime.adapters.robocasa.vla:RoboCasaPolicyHooks",
    "robotwin": "roboenv_runtime.adapters.robotwin.vla:RoboTwinPolicyHooks",
}


def register_hooks(simulator: str, target: str) -> None:
    """Register (or override) the hooks factory for ``simulator``."""
    HOOKS[simulator] = target


def _resolve(target: str) -> Callable[..., Any]:
    import importlib

    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def default_policy(simulator: str) -> PolicySpec | None:
    """The spec marked ``default: true`` for ``simulator`` (else the first one)."""
    specs = list_policies(simulator)
    if not specs:
        return None
    for spec in specs:
        if spec.default:
            return spec
    return specs[0]


def build_vla_tool(adapter: Any, simulator: str) -> Callable[..., dict[str, Any]] | None:
    """The ``vla_run`` callable for ``adapter``, or ``None`` when nothing is registered."""
    target = HOOKS.get(simulator)
    spec = default_policy(simulator)
    if target is None or spec is None:
        return None

    def prepare(instruction=None, policy=None):
        from roboenv_runtime.vla.client import WebsocketPolicyClient
        from roboenv_runtime.vla.server import ensure_server

        chosen = load_policy_spec(policy) if policy else spec
        if chosen.simulator != simulator:
            raise ValueError(f"policy {chosen.name!r} is for {chosen.simulator}, not {simulator}")
        hooks = _resolve(HOOKS[simulator])(adapter, chosen, instruction)
        if hooks.steps_left() <= 0:
            raise RuntimeError("episode step budget exhausted; vla_run refused")
        server = ensure_server(chosen)
        clients = adapter.__dict__.setdefault("_vla_clients", {})
        client = clients.get(server.url)
        if client is None:
            if chosen.protocol == "openpi-ws":
                client = WebsocketPolicyClient(server.url)
            elif chosen.protocol == "xpolicylab-ws":
                from roboenv_runtime.vla.client_xpl import XplPolicyClient

                client = XplPolicyClient(server.url, spec=chosen, adapter=adapter)
            else:
                raise ValueError(f"unknown policy protocol {chosen.protocol!r}")
            clients[server.url] = client
        return hooks, client, chosen

    def factory(n_actions=50, instruction=None, policy=None):
        from roboenv_runtime.vla.runner import iterate_policy

        hooks, client, chosen = prepare(instruction, policy)
        return iterate_policy(
            hooks, client, chosen, n_actions=n_actions,
            reward_fn=lambda: float(adapter.score().reward), poses_fn=hooks.poses,
            budget_fn=lambda: adapter.budget().as_dict(),
        )

    def vla_run(n_actions=50, instruction=None, policy=None, until=None):
        from roboenv_runtime.vla.runner import drive_policy

        return drive_policy(factory(n_actions, instruction, policy), until)

    vla_run._interactive_factory = factory

    doc = spec.tool_doc.strip() or (
        f"Hand closed-loop control to the pretrained policy {spec.name} for up to "
        "``n_actions`` control steps, then get control back with a before/after report."
    )
    doc += (
        "\n\nuntil=<callable>: optional check evaluated after EVERY policy action; the run "
        "stops as soon as it returns True (report note 'until condition met'). Use it to hand "
        "the policy the motion it was trained for and keep the decision in code, e.g. "
        "``vla_run(400, until=lambda: get_object_pose('bowl')[0][2] > 0.05)`` for a lift, "
        "instead of guessing a step count."
    )
    others = [s.name for s in list_policies(simulator) if s.name != spec.name]
    if others:
        doc += f"\n\nOther policies available through ``policy=``: {', '.join(others)}."
    vla_run.__doc__ = doc
    vla_run.__name__ = "vla_run"
    return vla_run


def attach_vla_tool(adapter: Any, simulator: str) -> Any:
    """Give ``adapter`` a ``vla_run`` tool (and ``Capability.VLA``).

    Only when its simulator has registered hooks and at least one policy spec; an
    adapter that already exports its own ``vla_run`` (RoboDojo) keeps it.  Returns the
    adapter.

    Nothing here touches the adapter's state.  It used to ask ``adapter.tools()``
    whether a ``vla_run`` was already there, which built the tool cache *before the
    first* ``reset`` — and cap-x rebuilds LIBERO's env on reset, so every cached LIBERO
    tool then closed over a stale environment: eight certified solutions replayed to
    0.0 (round-4 integration, 2026-08-28).  The check now happens inside the wrapper,
    where ``setdefault`` costs nothing and reads the tools the adapter really has.
    """
    if simulator not in HOOKS:
        return adapter
    tool = build_vla_tool(adapter, simulator)
    if tool is None:
        return adapter
    original_tools = adapter.tools

    def tools() -> dict[str, Callable[..., Any]]:
        out = dict(original_tools())
        out.setdefault("vla_run", tool)
        return out

    adapter.tools = tools
    with contextlib.suppress(Exception):  # a read-only capability set is fine, the tool exists
        adapter.capabilities = frozenset(set(adapter.capabilities) | {Capability.VLA})
    return adapter
