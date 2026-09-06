"""Pretrained policies (VLAs) as tools — one serving/observation layer for every simulator.

* :mod:`roboenv_runtime.vla.spec` — one YAML per (policy, simulator): how to serve it, what it expects
* :mod:`roboenv_runtime.vla.server` — start/reuse/stop policy-server subprocesses per GPU
* :mod:`roboenv_runtime.vla.client` — the openpi websocket protocol + the exact client-side preprocessing
* :mod:`roboenv_runtime.vla.runner` — the shared ``vla_run`` loop and its before/after report

Adapters add a ``vla_run`` tool by implementing :class:`roboenv_runtime.vla.runner.VlaHooks`
(see ``roboenv/adapters/libero_vla.py``); the worker's ``vla`` operation and
``finalize``'s action accounting work unchanged.  The client (``roboenv_runtime.vla.client``)
needs ``msgpack`` and ``websockets`` and is imported where it is used, so this
package imports cleanly in every simulator interpreter.
"""

from roboenv_runtime.vla.runner import VlaHooks, run_policy
from roboenv_runtime.vla.server import PolicyServer, ensure_server, stop_all
from roboenv_runtime.vla.spec import PolicySpec, list_policies, load_policy_spec

__all__ = [
    "PolicyServer",
    "PolicySpec",
    "VlaHooks",
    "ensure_server",
    "list_policies",
    "load_policy_spec",
    "run_policy",
    "stop_all",
]
