"""Explicit practice reset; never enabled by the ordinary SDK."""

from roboenv.client import PUBLIC_OPS, Client


class PracticeClient(Client):
    allowed_ops = PUBLIC_OPS | {"reset"}

    def request(self, op, args=None, **kwargs):
        if op == "reset":
            health = super().request("health")
            if health.get("budget_mode") != "practice":
                raise ValueError(
                    "rehearsal controls require an explicitly configured practice task"
                )
        return super().request(op, args, **kwargs)
