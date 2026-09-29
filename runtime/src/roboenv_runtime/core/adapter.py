"""Native lifecycle, control, private observations and verdicts for one world."""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

from roboenv_runtime.core.types import Budget, ScoreReport

__all__ = [
    "BUILTIN_ADAPTERS",
    "AdapterBase",
    "EnvAdapter",
    "list_adapters",
    "make_adapter",
    "register_adapter",
]

log = logging.getLogger(__name__)


@runtime_checkable
class EnvAdapter(Protocol):
    """What the harness requires from a simulator.

    Attributes:
        name: Registry key (``"robodojo"``, ``"robotwin"``, ...).
        embodiment: Native robot embodiment (``"arx_x5"``,
            ``"piper"``, ``"panda"``, ...).
    """

    name: str
    embodiment: str
    control_interval_s: float

    def advance_control(self) -> None:
        """Advance all active devices through one native control interval."""
        ...

    def goal(self) -> str:
        """The benchmark-provided language goal, or ``""`` when tasks carry none."""
        ...

    def reset(self, seed: int) -> None:
        """Start a fresh episode."""
        ...

    def configure_execution_mode(self, mode: str) -> None:
        """Select episode-limit policy before resetting the world."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...

    def observe(self) -> dict[str, Any]:
        """Normalised ground-truth snapshot (``objects``, ``grippers``, ``budget``)."""
        ...

    def score(self) -> ScoreReport:
        """Evaluate the current world state."""
        ...

    def budget(self) -> Budget:
        """How much of the episode has been consumed."""
        ...

    def frames(self) -> list[Any]:
        """Camera frames captured since the last reset (may be empty)."""
        ...

    def capture(self) -> Any | None:
        """One current camera image (``None`` when unsupported)."""
        ...


class AdapterBase(ABC):
    """Optional defaults for native adapters; reset and score are required."""

    name: ClassVar[str] = "base"
    embodiment: ClassVar[str] = "general"
    execution_mode = "realtime"

    def goal(self) -> str:
        """Default: no benchmark-provided goal text."""
        return ""

    def configure_execution_mode(self, mode: str) -> None:
        """Realtime counts steps but does not enforce an episode step limit."""
        if mode not in {"stepped", "realtime"}:
            raise ValueError("execution mode must be stepped or realtime")
        self.execution_mode = mode

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Start a fresh episode."""

    def close(self) -> None:
        """Default: nothing to release."""
        return None

    @abstractmethod
    def score(self) -> ScoreReport:
        """Evaluate the current world state."""

    def budget(self) -> Budget:
        """Default: an unbounded placeholder; real adapters override."""
        return Budget()

    def observe(self) -> dict[str, Any]:
        """Private owner snapshot; native adapters provide their recording state."""
        return {"budget": self.budget().as_dict()}

    def frames(self) -> list[Any]:
        """Default: no video capture."""
        return []

    def capture(self) -> Any | None:
        """Default: the most recent captured frame, if any."""
        frames = self.frames()
        return frames[-1] if frames else None


# ------------------------------------------------------------------ registry

_REGISTRY: dict[str, type] = {}

BUILTIN_ADAPTERS: dict[str, str] = {
    "fakeshell": "roboenv_runtime.adapters.fakeshell",
    "robodojo": "roboenv_runtime.adapters.robodojo",
    "robotwin": "roboenv_runtime.adapters.robotwin",
    "libero": "roboenv_runtime.adapters.libero",
    "robocasa": "roboenv_runtime.adapters.robocasa",
    "simple": "roboenv_runtime.adapters.simple",
    "behavior": "roboenv_runtime.adapters.behavior",
}
"""Adapter name -> module that registers it (imported on first use)."""


def register_adapter(cls: type) -> type:
    """Class decorator: make ``cls`` constructible through :func:`make_adapter`.

    The class must define a non-empty ``name``.  Re-registering the same name
    replaces the previous class (useful for tests and hot reloads) and is
    logged at debug level.
    """
    name = getattr(cls, "name", "")
    if not isinstance(name, str) or not name:
        raise TypeError(f"{cls.__name__} must define a non-empty class attribute 'name'")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        log.debug("adapter %r re-registered by %s", name, cls.__qualname__)
    _REGISTRY[name] = cls
    return cls


def list_adapters() -> list[str]:
    """Names that :func:`make_adapter` accepts (registered and built-in)."""
    return sorted(set(_REGISTRY) | set(BUILTIN_ADAPTERS))


def make_adapter(name: str, **kwargs: Any) -> EnvAdapter:
    """Instantiate a registered adapter, importing its module on demand.

    Raises:
        KeyError: when ``name`` is neither registered nor built-in.
        ImportError: when the simulator's interpreter lacks its dependencies;
            the message says which module failed so the caller can activate
            the right environment.
    """
    if name not in _REGISTRY:
        module = BUILTIN_ADAPTERS.get(name)
        if module is None:
            raise KeyError(f"No adapter {name!r}. Known: {list_adapters()}")
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ImportError(
                f"adapter {name!r} could not be imported from {module}: {exc}. "
                "Prepare the simulator using roboenv prepare BACKEND."
            ) from exc
        if name not in _REGISTRY:
            raise KeyError(f"{module} imported but did not register adapter {name!r}")
    return _REGISTRY[name](**kwargs)
