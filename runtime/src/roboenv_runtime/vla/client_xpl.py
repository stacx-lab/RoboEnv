"""The XPolicyLab websocket protocol behind the same ``infer(obs)`` call.

RoboDojo's policy zoo does not speak openpi's one-shot ``infer`` protocol: its
servers expose the *model* methods (``reset`` / ``update_obs`` /
``get_action``) over a websocket, and one ``get_action`` answers with a whole
action chunk.  :class:`XplPolicyClient` wraps the vendored
``client_server/ws/model_client.py:WsModelClient`` so that
:func:`roboenv_runtime.vla.runner.run_policy` can drive an XPolicyLab policy exactly the
way it drives an openpi one — ``infer(obs) -> {"actions": (chunk, dim)}``.

Two shape conventions have to be reconciled here.  XPolicyLab answers with the
*unpacked* action dictionary of its data standard (``left_arm_joint_state``,
``left_ee_joint_state``, ``right_arm_joint_state``, ``right_ee_joint_state``
for a dual-arm joint policy), one dict per predicted control step; the runner
wants a plain ``(chunk, dim)`` array.  The concatenation order is the spec's
``action.keys`` — for RoboTwin it is exactly the ``[left arm 6, left gripper,
right arm 6, right gripper]`` vector that ``take_action(..., 'qpos')`` expects.

``WsModelClient`` lives in the RoboDojo checkout, not on ``sys.path``; it is
imported inside the constructor so importing this module never depends on a
checkout being present.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from roboenv_runtime.paths import robodojo_root
from roboenv_runtime.vla.spec import PolicySpec

__all__ = ["XplPolicyClient", "xpolicylab_root"]

log = logging.getLogger(__name__)

#: Default concatenation order for a dual-arm joint policy (XPolicyLab's
#: ``unpack_robot_state`` layout for ``arm_dim=[6, 6]``, ``ee_dim=[1, 1]``).
DEFAULT_ACTION_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)


def xpolicylab_root() -> str:
    """The vendored XPolicyLab checkout that ships ``client_server``."""
    return str(robodojo_root() / "XPolicyLab")


def _import_ws_model_client() -> Any:
    """``WsModelClient``, with the XPolicyLab checkout appended to ``sys.path``.

    Appended (never prepended) and only inside this function: XPolicyLab
    shadows a couple of very common module names (``utils``, ``scripts``), so
    putting it in front of the simulator's own path breaks the simulator.
    """
    root = xpolicylab_root()
    if root not in sys.path:
        sys.path.append(root)
    from client_server.ws.model_client import WsModelClient

    return WsModelClient


class XplPolicyClient:
    """``infer(obs) -> {"actions": (chunk, dim)}`` over the XPolicyLab protocol.

    One connection per adapter; the policy's episode state is cleared on
    connect, so a new episode should build a new client (which is what
    :func:`roboenv_runtime.vla.attach.build_vla_tool` does — it caches clients on the
    adapter instance).  ``reset_on_connect=False`` plus a short
    ``max_connect_seconds`` makes it a liveness probe for a server another
    process already runs (:func:`roboenv_runtime.vla.server._attached`).
    """

    def __init__(
        self,
        url: str,
        *,
        spec: PolicySpec,
        adapter: Any = None,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        reset_on_connect: bool = True,
        connect_timeout_s: float | None = None,
        max_connect_seconds: float | None = None,
    ):
        self.url = url if url.startswith("ws") else f"ws://{url}"
        self.spec = spec
        self.adapter = adapter
        task = str(getattr(adapter, "task_name", "") or spec.simulator)
        self.action_keys = tuple(spec.action.get("keys") or DEFAULT_ACTION_KEYS)
        self.action_dim = spec.action_dim
        ws_model_client = _import_ws_model_client()
        self._client = ws_model_client(
            url=self.url,
            evaluation_id=f"roboenv-{spec.simulator}-{spec.name}",
            trial_id=trial_id or f"{spec.simulator}-{task}",
            action_case_id=evaluation_id or f"{spec.simulator}_{task}",
            # keepalive off, like the zoo's own deploy.yml for local evaluation:
            # a simulator that blocks for seconds between calls is normal here.
            ws_ping_interval_s=None,
            ws_ping_timeout_s=None,
            connect_timeout_s=connect_timeout_s,
            max_connect_seconds=max_connect_seconds,
        )
        self.metadata = {"policy": spec.name, "protocol": spec.protocol, "url": self.url}
        if reset_on_connect:
            self.reset()
        log.info("connected to XPolicyLab policy server %s (%s)", self.url, spec.name)

    @property
    def supports_rng_reset(self) -> bool:
        """False: the XPolicyLab protocol has no call that reseeds the sampler.

        ``reset`` clears the model's episode state, not its RNG, so a flow-matching
        policy like InternVLA-A1.5 draws fresh noise on every chunk.  Specs for this
        protocol therefore set ``deterministic: false``.
        """
        return False

    def reset(self) -> None:
        """Clear the policy's episode state (``model.reset()``, no payload)."""
        self._client.call(func_name="reset")

    def infer(
        self, obs: Mapping[str, Any], *, reset_rng: int | bool | None = None
    ) -> dict[str, Any]:
        """One ``update_obs`` + ``get_action`` round trip, as an action chunk.

        ``reset_rng`` is accepted for the shared runner's signature and ignored: see
        :attr:`supports_rng_reset`.
        """
        self._client.call(func_name="update_obs", obs=dict(obs))
        chunk = self._client.call(func_name="get_action")
        return {"actions": self.to_array(chunk)}

    def to_array(self, chunk: Any) -> np.ndarray:
        """Flatten one XPolicyLab action chunk into ``(steps, dim)`` floats.

        Accepts what the zoo actually returns: a list of action dictionaries,
        a ``(steps, dim)`` array, a single action (dict or vector), or any of
        those wrapped in ``{"actions": ...}``.
        """
        if isinstance(chunk, Mapping) and "actions" in chunk:
            return self.to_array(chunk["actions"])
        if isinstance(chunk, Mapping):
            return self._flatten(chunk)[None, :]
        if chunk is None:
            return np.zeros((0, self.action_dim), dtype=float)
        if isinstance(chunk, np.ndarray):
            array = np.asarray(chunk, dtype=float)
            return array[None, :] if array.ndim == 1 else array.reshape(len(array), -1)
        if isinstance(chunk, Sequence) and not isinstance(chunk, (str, bytes, bytearray)):
            if not chunk:
                return np.zeros((0, self.action_dim), dtype=float)
            rows = [
                self._flatten(step)
                if isinstance(step, Mapping)
                else np.asarray(step, dtype=float).reshape(-1)
                for step in chunk
            ]
            return np.stack(rows)
        raise TypeError(f"unsupported action chunk of type {type(chunk).__name__}")

    def _flatten(self, action: Mapping[str, Any]) -> np.ndarray:
        parts = []
        for key in self.action_keys:
            if key not in action:
                raise KeyError(
                    f"{self.spec.name}: action dict is missing {key!r}; got {sorted(action)}"
                )
            parts.append(np.asarray(action[key], dtype=float).reshape(-1))
        flat = np.concatenate(parts) if parts else np.zeros(0)
        if flat.shape[0] != self.action_dim:
            raise ValueError(
                f"{self.spec.name}: action is {flat.shape[0]}-dim, spec says {self.action_dim}"
            )
        return flat

    def close(self) -> None:
        """Close the websocket (best effort); the server keeps running."""
        with contextlib.suppress(Exception):
            self._client.close()
