"""Print the current public task from a copied RoboEnv/Kinex workspace."""

import json
import secrets
import sys
from pathlib import Path

root = Path(__file__).resolve().parent / ".kinex/devices"
sys.path.insert(0, str(root))
from _roboenv import Transport  # noqa: E402


def read_task():
    driver = next(iter(sorted(root.glob("*/driver.py"))), None)
    if driver is None:
        raise ValueError("Workspace has no RoboEnv device drivers")
    transport = Transport(driver)
    try:
        if not transport.endpoint:
            raise ValueError("Public task reads require an HTTP endpoint")
        request = {
            **transport.identity,
            "request_id": "task-" + secrets.token_hex(8),
            "deadline_s": 30,
            "op": "environment.describe",
            "args": {},
        }
        response = json.loads(
            transport.exchange(
                transport.endpoint, "POST", body=json.dumps(request).encode() + b"\n"
            )
        )
        if any(response.get(k) != request[k] for k in (*transport.identity, "request_id")):
            raise ValueError("Mismatched task response")
        if response.get("ok") is not True:
            raise ValueError(response.get("error", "Task read failed"))
        return response["result"]["task"]
    finally:
        transport.close_all()


if __name__ == "__main__":
    print(json.dumps(read_task()))
