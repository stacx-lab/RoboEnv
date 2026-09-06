"""Talking to a policy server, and the image/state helpers every client shares.

Two wire protocols exist in the zoo we use:

* **openpi** (``protocol: openpi-ws``) — one websocket, msgpack+numpy; the
  server sends its metadata on connect, then every ``infer(obs)`` request is
  answered with ``{"actions": (chunk, dim)}``.  π0 / π0.5 checkpoints served by
  ``scripts/serve_policy.py`` speak it.
* **XPolicyLab** (``protocol: xpolicylab-ws``) — RoboDojo's benchmark server
  (``update_obs`` / ``get_action`` calls); the RoboDojo backend uses the
  vendored ``WsModelClient`` for it.

The helpers below reproduce the *client-side* preprocessing of the openpi
evaluation scripts exactly (``resize_with_pad`` is their PIL port of
``tf.image.resize_with_pad``), because a policy is only as good as the match
between what it sees now and what it saw in training.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import Any

import numpy as np

from roboenv_runtime.vla import msgpack_numpy

__all__ = [
    "WebsocketPolicyClient",
    "quat_xyzw_to_axisangle",
    "resize_with_pad",
    "rotate180",
]

log = logging.getLogger(__name__)


class WebsocketPolicyClient:
    """The openpi websocket protocol: ``infer(obs) -> {"actions": ...}``."""

    def __init__(self, url: str, *, connect_timeout_s: float = 30.0, api_key: str | None = None):
        import websockets.sync.client

        self.url = url if url.startswith("ws") else f"ws://{url}"
        self._packer = msgpack_numpy.Packer()
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "compression": None,
                    "max_size": None,
                    "additional_headers": headers,
                }
                try:  # websockets >= 16 wants an explicit opt-out of the context manager
                    self._ws = websockets.sync.client.connect(self.url, legacy=True, **kwargs)
                except TypeError:
                    self._ws = websockets.sync.client.connect(self.url, **kwargs)
                self.metadata = msgpack_numpy.unpackb(self._ws.recv())
                break
            except (ConnectionRefusedError, OSError) as exc:  # server still booting
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"policy server at {self.url} not reachable: {exc}"
                    ) from exc
                time.sleep(1.0)
        log.info("connected to policy server %s: %s", self.url, self.metadata)

    @property
    def supports_rng_reset(self) -> bool:
        """Whether this server understands :meth:`infer`'s ``reset_rng`` (see serve_openpi)."""
        return bool((self.metadata or {}).get("roboenv_rng_reset"))

    def infer(self, obs: dict[str, Any], *, reset_rng: int | bool | None = None) -> dict[str, Any]:
        """One inference: the policy's observation in, ``{"actions": (chunk, dim)}`` out.

        ``reset_rng`` restores the policy's sampling key first (servers started by
        ``roboenv_runtime.vla.serve_openpi``), which is what makes a replay reproduce.
        """
        if reset_rng is not None and self.supports_rng_reset:
            obs = {**obs, "__roboenv_reset_rng__": reset_rng}
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server error: {response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        """Close the socket (best effort)."""
        with contextlib.suppress(Exception):
            self._ws.close()


def rotate180(image: np.ndarray) -> np.ndarray:
    """``image[::-1, ::-1]`` as a contiguous array (LIBERO's train-time convention)."""
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize without distortion and zero-pad to ``height x width``.

    openpi's PIL port of ``tf.image.resize_with_pad`` (bilinear); ``image`` is HWC uint8.
    """
    from PIL import Image

    image = np.asarray(image)
    if image.shape[:2] == (height, width):
        return image
    pil = Image.fromarray(image)
    cur_width, cur_height = pil.size
    ratio = max(cur_width / width, cur_height / height)
    resized = pil.resize((int(cur_width / ratio), int(cur_height / ratio)), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_h = max(0, int((height - resized.height) / 2))
    pad_w = max(0, int((width - resized.width) / 2))
    canvas.paste(resized, (pad_w, pad_h))
    return np.asarray(canvas)


def quat_xyzw_to_axisangle(quat: Any) -> np.ndarray:
    """The ``quat2axisangle`` of robosuite (xyzw in, exponential coordinates out)."""
    q = np.array(quat, dtype=np.float64).reshape(4).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den
