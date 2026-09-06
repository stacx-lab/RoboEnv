"""An adapter that lives on the other side of a worker socket.

:class:`RemoteAdapter` implements the full :class:`roboenv_runtime.core.EnvAdapter`
contract by talking to a worker (or supervisor) through :class:`RemoteAPI`.
Every interaction style therefore runs unchanged against a simulator in
another process — another interpreter, another GPU, or another machine —
and every action leaves a journal entry on the supervisor side.

Tools are rebuilt from the worker's ``api`` operation: each becomes a closure
calling ``call`` with keyword arguments only (positional arguments are mapped
onto the documented signature), so agent code written for the in-process
adapter keeps working.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from roboenv_runtime.core.adapter import AdapterBase, register_adapter
from roboenv_runtime.core.types import Budget, ScoreReport
from roboenv_runtime.runtime.client import ClientError, RemoteAPI

__all__ = ["RemoteAdapter"]


def _documentation_signature(
    signature_text: str, parameters: list[str]
) -> inspect.Signature | None:
    """Rebuild a signature for prompt rendering: full text when it evaluates, names otherwise."""
    body = signature_text.split(" -> ")[0]
    if body.startswith("(") and body.endswith(")"):
        namespace: dict[str, Any] = {"__builtins__": __builtins__, "Any": Any}
        try:
            exec(f"def _f{body}: pass", namespace)  # trusted worker text
            return inspect.signature(namespace["_f"])
        except Exception:  # unresolvable annotation names
            pass
    if parameters:
        return inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in parameters
            ]
        )
    return None


def _make_tool(api: RemoteAPI, name: str, spec: Mapping[str, Any]) -> Callable[..., Any]:
    parameters = [str(p) for p in (spec.get("parameters") or [])]
    documentation = _documentation_signature(str(spec.get("signature") or ""), parameters)

    def tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            if len(args) > len(parameters):
                raise TypeError(
                    f"{name}() takes {len(parameters)} positional arguments "
                    f"but {len(args)} were given"
                )
            kwargs = {**dict(zip(parameters, args, strict=False)), **kwargs}
        try:
            return api.call(name, **kwargs)
        except ClientError as exc:
            raise RuntimeError(str(exc)) from exc

    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = str(spec.get("doc") or "")
    if documentation is not None:
        tool.__signature__ = documentation  # type: ignore[attr-defined]
    return tool


def _make_vla_tool(api: RemoteAPI, spec: Mapping[str, Any]) -> Callable[..., Any]:
    """``vla_run`` over the wire: the worker runs the policy loop, the client waits."""
    parameters = [str(p) for p in (spec.get("parameters") or [])]
    documentation = _documentation_signature(str(spec.get("signature") or ""), parameters)

    def vla_run(*args: Any, **kwargs: Any) -> Any:
        if args:
            kwargs = {**dict(zip(parameters, args, strict=False)), **kwargs}
        try:
            return api.vla(**kwargs)
        except ClientError as exc:
            raise RuntimeError(str(exc)) from exc

    vla_run.__name__ = vla_run.__qualname__ = "vla_run"
    vla_run.__doc__ = str(spec.get("doc") or "")
    if documentation is not None:
        vla_run.__signature__ = documentation  # type: ignore[attr-defined]
    return vla_run


@register_adapter
class RemoteAdapter(AdapterBase):
    """The adapter contract over a worker socket."""

    name: ClassVar[str] = "remote"
    embodiment: ClassVar[str] = "general"

    def __init__(
        self,
        socket_path: str | None = None,
        *,
        api: RemoteAPI | None = None,
        session: str | None = None,
        run_dir: str | None = None,
        generation: int | None = None,
        deadline_s: float = 600.0,
    ) -> None:
        self.api = api or RemoteAPI(
            socket_path,
            run_dir=run_dir,
            session=session,
            generation=generation,
            default_deadline_s=deadline_s,
        )
        self._describe: dict[str, Any] | None = None
        self._tools: dict[str, Callable[..., Any]] | None = None

    # ------------------------------------------------------------ discovery

    def describe(self) -> dict[str, Any]:
        """The worker's ``api`` description (cached until reset)."""
        if self._describe is None:
            self._describe = dict(self.api.api())
            context = self._describe.get("context") or {}
            self.embodiment = str(context.get("embodiment") or self.embodiment)  # type: ignore[misc]
            caps = context.get("capabilities") or []
            self.capabilities = frozenset(str(c) for c in caps)  # type: ignore[misc]
        return self._describe

    def _context(self) -> dict[str, Any]:
        return dict(self.describe().get("context") or {})

    def situation(self) -> str:
        """From the worker's adapter."""
        return str(self._context().get("situation") or "")

    def facts(self) -> str:
        """From the worker's adapter."""
        return str(self._context().get("facts") or "")

    def goal(self) -> str:
        """From the worker's adapter."""
        return str(self._context().get("goal") or "")

    # ------------------------------------------------------------ tools

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        """Closures over the worker's allow-listed tools."""
        if self._tools is None:
            specs = self.describe().get("tools") or {}
            self._tools = {name: _make_tool(self.api, name, spec) for name, spec in specs.items()}
            vla_spec = self._context().get("vla_tool")
            if isinstance(vla_spec, Mapping):  # served through the worker's ``vla`` op
                self._tools.setdefault("vla_run", _make_vla_tool(self.api, vla_spec))
        return dict(self._tools)

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int) -> None:
        """Ask the worker to reset the same task with ``seed``."""
        self.api.reset(seed=int(seed))
        self._describe = None
        self._tools = None

    def close(self) -> None:
        """The worker outlives clients; nothing to release here."""
        return None

    # ------------------------------------------------------------ evaluation

    def observe(self) -> dict[str, Any]:
        """The worker's normalised observation."""
        return dict(self.api.observe())

    def score(self) -> ScoreReport:
        """``score.peek`` (read-only) mapped to a :class:`ScoreReport`."""
        peek = self.api.score_peek() or {}
        return ScoreReport(
            reward=float(peek.get("reward", 0.0) or 0.0),
            raw_score=float(peek.get("raw_score", peek.get("reward", 0.0)) or 0.0),
            tier=int(peek.get("tier", 0) or 0),
            completed=bool(peek.get("completed", False)),
            detail={k: v for k, v in peek.items() if k not in ("reward", "raw_score", "tier")},
        )

    def budget(self) -> Budget:
        """From the observation's ``budget`` block."""
        return Budget.coerce(self.observe().get("budget"))

    def submit(self) -> dict[str, Any]:
        """Freeze the episode on the worker and return the final evaluation."""
        return dict(self.api.submit())

    # ------------------------------------------------------------ media

    def frames(self) -> list[Any]:
        """Frames stay on the worker; use ``capture``."""
        return []

    def capture(self) -> Any | None:
        """Ask the worker for a PNG (path inside the run directory) and return it."""
        try:
            return self.api.capture("captures/latest.png")
        except ClientError:
            return None

    # ------------------------------------------------------------ optional

    def checkpoint(self) -> Any:
        """Worker-side checkpoint (``Capability.CHECKPOINT``)."""
        return self.api.checkpoint()

    def restore(self, token: Any) -> None:
        """Worker-side restore."""
        self.api.restore(token=token)

    def __repr__(self) -> str:
        return f"RemoteAdapter({self.api!r})"
