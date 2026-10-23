"""Exercise actual public instrument commands through Kinex's Python tool runner."""

from kinex import RPCError, args, device, finish

params = args()
before = device("instrument", "read")
device("world", "wait", {"duration_s": 2.0})
for action, arguments in params["commands"]:
    result = device("instrument", action, arguments)
    assert result["status"] == "succeeded", result
try:
    device("world", "wait", {"duration_s": params["wait_s"]})
except RPCError as error:
    if error.code != 'episode_finished':
        raise
after = device("instrument", "read")
# Large instruments have hundreds of public slots. Keep the controller result
# below Kinex's output cap; the probe and owner retain the complete snapshots.
finish({label: {"timestamp": reading["timestamp"],
                "observation_count": len(reading["observations"])}
        for label, reading in [("before", before), ("after", after)]})
