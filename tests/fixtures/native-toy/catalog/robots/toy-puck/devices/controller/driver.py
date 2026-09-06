import json
import sys

from roboenv.client import call, rpc


def handle(action, args):
    result = (
        call(args["tool_name"], **args.get("kwargs", {})) if action == "invoke" else rpc("observe")
    )
    return {"result": result}


if __name__ == "__main__":
    print(json.dumps(handle(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})))
