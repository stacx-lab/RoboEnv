"""msgpack with NumPy arrays — the wire format of openpi policy servers.

Same encoding as ``openpi_client.msgpack_numpy`` (Apache-2.0): an array travels
as ``{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": [...]}``.
Unlike the original, arrays are converted *before* packing and *after*
unpacking instead of through msgpack's ``default``/``object_hook`` — the LIBERO
interpreter carries a globally patched msgpack (``msgpack_numpy.patch()`` from
viser) whose own array hooks would otherwise win and put a format on the wire
the server cannot read.  Object/void/complex arrays are refused, never pickled.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

__all__ = ["Packer", "packb", "unpackb"]


def _encode(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if b"__ndarray__" in obj:
            return np.ndarray(
                buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"])
            )
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def packb(obj: Any) -> bytes:
    """Serialise ``obj`` (dicts/lists/scalars/NumPy arrays) to msgpack bytes."""
    return msgpack.packb(_encode(obj), use_bin_type=True)


def unpackb(data: bytes) -> Any:
    """Deserialise msgpack bytes, turning encoded arrays back into NumPy arrays."""
    return _decode(msgpack.unpackb(data, raw=False, strict_map_key=False))


class Packer:
    """Object with ``pack``: the interface openpi's client exposes."""

    def pack(self, obj: Any) -> bytes:
        """Same as :func:`packb`."""
        return packb(obj)
