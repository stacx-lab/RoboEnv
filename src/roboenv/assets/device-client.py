"""Thin device-driver/1 client. Environment implementation stays on the server."""

import json
import sys
from pathlib import Path

from roboenv.client import Client, RemoteError, UncertainResult


def handle(action, args):
    try:
        connection = Path(__file__).with_name("connection.json")
        if connection.is_file():
            config = json.loads(connection.read_text())
            client = Client(
                config.get("endpoint") or config["socket"],
                session=config["session"],
                generation=config["generation"],
            )
        else:
            client = Client()
        reply = client.device(Path(__file__).parent.name).call(action, args)
        data = reply["data"]
        return {**(data if isinstance(data, dict) else {"result": data}), "ok": True}
    except RemoteError as exc:
        return {"ok": False, "error": str(exc), "code": exc.code, "request_id": exc.request_id}
    except UncertainResult as exc:
        return {
            "ok": False,
            "error": str(exc),
            "code": "outcome_unknown",
            "request_id": exc.request_id,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


if __name__ == "__main__":
    # Profiles use one-shot mode: the environment already owns the expensive
    # connection. Do not wrap motion commands in automatic process retries.
    action = sys.argv[1] if len(sys.argv) > 1 else "read"
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(handle(action, args), allow_nan=False))
