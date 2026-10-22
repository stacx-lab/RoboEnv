"""Exercise actual public instrument commands through Kinex's Python tool runner."""

from kinex import RPCError, device, finish

before = device("instrument", "read")
for action, arguments in [("light_set", {"on": True}), ("sash_set", {"height_m": 0.457})]:
    result = device("instrument", action, arguments)
    assert result["status"] == "succeeded", result
try:
    device("world", "wait", {"duration_s": 1.2})
except RPCError as error:
    if error.code != 'episode_finished':
        raise
finish({"before": before, "after": device("instrument", "read")})
