"""Explicit BEHAVIOR symbolic controller binding; not continuous motor control."""


def create_actions(backend):
    def read(args):
        return {"observation": backend.rpc("observe"), "abstraction": "symbolic-primitives"}

    def invoke(args):
        return {"result": backend.call(args["tool_name"], **args.get("kwargs", {}))}

    return {
        "read": read,
        "state": read,
        "status": read,
        "load": read,
        "invoke": invoke,
        "catalog": lambda args: backend.api(),
    }


REQUIRES = {}
