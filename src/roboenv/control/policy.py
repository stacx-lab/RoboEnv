"""Agent-side callback handling for a continuous environment-owned policy rollout."""


def run_with_callback(client, arguments, until):
    if not callable(until):
        raise TypeError("until must be callable")
    state = client.request("vla", {"_interactive": "start", "arguments": arguments})
    try:
        while state["status"] != "done":
            if state.get("protocol") != "roboenv/vla-callback/1":
                raise RuntimeError("unexpected policy callback protocol")
            stop = bool(until())
            state = client.request(
                "vla",
                {
                    "_interactive": "advance",
                    "token": state["token"],
                    "sequence": state["sequence"],
                    "stop": stop,
                },
            )
        return state["report"]
    except BaseException:
        # Cancels this rollout only; Harbor still owns the episode and verifier submit.
        client.request("vla", {"_interactive": "stop", "token": state["token"]})
        raise
