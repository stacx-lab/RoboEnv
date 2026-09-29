"""Versioned, agent-neutral device service vocabulary (no runtime imports)."""

DEVICE_CONTRACT = "roboenv/devices/1"
DEVICE_OPS = frozenset(
    {"devices.list", "devices.describe", "devices.call", "devices.cancel"}
)
SERVICE_OPS = DEVICE_OPS | {"environment.describe", "operations.get"}
READ_ACTIONS = frozenset(
    {"read", "state", "status", "load", "unload", "ik", "fk"}
)
