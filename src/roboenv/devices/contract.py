"""Versioned, agent-neutral device service vocabulary (no runtime imports)."""

DEVICE_CONTRACT = "roboenv/devices/1"
DEVICE_OPS = frozenset({"devices.list", "devices.describe", "devices.call"})
SERVICE_OPS = DEVICE_OPS | {"environment.describe", "operations.get"}
READ_ACTIONS = frozenset(
    {"read", "snapshot", "state", "status", "load", "unload", "ik", "fk", "catalog", "api", "ping"}
)
# Existing catalog bridges cannot implement these faithfully. No fake success.
UNSUPPORTED_ACTIONS = {
    "stop": "This backend has no interruptible motion controller or emergency stop.",
    "ik": "General joint-space inverse kinematics is not implemented by this bridge.",
    "fk": "General joint-space forward kinematics is not implemented by this bridge.",
    "unload": "Device lifecycle belongs to the environment owner, not agent clients.",
    "policy_run": "Use the separately journaled vla operation for policy execution.",
}
