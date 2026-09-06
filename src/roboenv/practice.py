"""Explicit authoring-only lifecycle operations; never enabled by the ordinary SDK."""

from .client import PUBLIC_OPS, Client

CONTROL = frozenset({"reset", "reset_same_task", "checkpoint", "restore"})


class PracticeClient(Client):
    allowed_ops = PUBLIC_OPS | CONTROL

    def request(self, op, args=None, **kwargs):
        if op in CONTROL:
            health = super().request("health")
            if health.get("budget_mode") != "practice":
                raise ValueError(
                    "rehearsal controls require an explicitly configured practice task"
                )
        return super().request(op, args, **kwargs)
