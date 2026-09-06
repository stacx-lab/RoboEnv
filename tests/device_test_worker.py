"""CPU worker with a deliberately fallible binding for protocol conformance tests.

Fault injection belongs in tests, not in the production binding registry.
"""

import time

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.bindings import assembly, default_registry
from roboenv_runtime.core.access import is_read_only
from roboenv_runtime.runtime.worker import main


def bind_test_device(spec, root, context):
    def call(execution, name, kwargs):
        args = {"name": name, "kwargs": kwargs}
        return execution.perform(
            "call",
            args,
            lambda: context.public_dispatch("call", args),
            mutating=not is_read_only(name),
        )

    def read(args, execution):
        if args.get("bad_read"):
            return call(execution, "push", {"dx": 0.1})
        return execution.perform("observe", {}, lambda: context.public_dispatch("observe", {}))

    def invoke(args, execution):
        result = call(execution, args["tool_name"], args.get("kwargs", {}))
        time.sleep(args.get("sleep_s", 0))
        if args.get("fail_after"):
            raise RuntimeError("partial motion")
        return {"result": result}

    return BoundDevice(
        spec.definition(root),
        {
            "read": Action(read),
            "state": Action(read),
            "invoke": Action(invoke),
            "stop": Action(reason="Test device has no stop controller"),
        },
        implementation="test-binding",
    )


def test_registry():
    registry = default_registry()
    registry.register("test.device/1", bind_test_device)
    return registry


if __name__ == "__main__":
    assembly.default_registry = test_registry
    raise SystemExit(main())
