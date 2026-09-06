"""In-process access to counted backend controllers, not a public RPC client.

Device handlers use the actual allowlisted controller functions captured by the
worker. Scoring/capture remain explicit world services. Nothing loads executable
code from the task's catalog, and no function here connects to a worker socket.
"""

from contextvars import ContextVar

from roboenv.devices.model import DeviceError


class _Functions:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.backend.call(name, *args, **kwargs)


class BackendControllers:
    def __init__(self, context):
        if context.public_tools is None:
            raise ValueError("backend bindings require the worker's allowlisted controller table")
        self.context = context
        self.tools = dict(context.public_tools)
        self.functions = _Functions(self)
        self._execution = ContextVar("roboenv_backend_execution", default=None)

    def execute(self, handler, args, execution):
        token = self._execution.set(execution)
        try:
            return handler(args)
        finally:
            self._execution.reset(token)

    def _run(self, op, args, fn, *, mutating=False):
        execution = self._execution.get()
        if execution is None:
            raise DeviceError("binding_setup_failed", "Controllers cannot run during binding setup")
        return execution.perform(op, args, fn, mutating=mutating)

    def call(self, name, *args, **kwargs):
        from roboenv_runtime.core.access import is_read_only

        if not isinstance(name, str) or name not in self.tools:
            raise DeviceError("unknown_tool", f"Unknown backend controller: {name!r}")
        return self._run(
            "binding.controller",
            {"name": name, "args": list(args), "kwargs": kwargs},
            lambda: self.tools[name](*args, **kwargs),
            mutating=not is_read_only(name),
        )

    def rpc(self, op, args=None, timeout=None):
        if op not in {"api", "health", "observe", "score.peek", "capture"}:
            raise DeviceError("permission_denied", "Only read-only world services are available")
        args = {} if args is None else args
        result = self._run("world." + op, args, lambda: self.context.public_dispatch(op, args))
        if op == "health" and isinstance(result.get("worker"), str):
            result = {**result, "worker": {"state": result["worker"]}}
        return result

    def api(self):
        return self.rpc("api")

    def capture(self, name):
        return self.rpc("capture", {"path": name})["path"]
