"""CPU worker with a deliberately fallible binding for protocol conformance tests.

Fault injection belongs in tests, not in the production binding registry.
"""

import time

from roboenv.devices.model import Action, BoundDevice
from roboenv_runtime.bindings import assembly, default_registry
from roboenv_runtime.runtime.worker import main


def bind_test_device(spec, root, context):
    def write(args, execution):
        target = args["target"]
        result = execution.perform(
            "binding.write", {"target": target},
            lambda: context.native.push(dx=target[0] - context.native.pos[0]),
            mutating=True,
        )
        time.sleep(args.get("sleep_s", 0))
        if args.get("fail_after"):
            raise RuntimeError("partial motion")
        return {"result": result}

    def read(args, execution):
        if args.get("bad_read"):
            return write({"target": [0.1]}, execution)
        return execution.perform("observe", {}, lambda: context.native.observe())

    return BoundDevice(
        spec.definition(root),
        {
            "read": Action(read),
            "state": Action(read),
            "write": Action(write),
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
