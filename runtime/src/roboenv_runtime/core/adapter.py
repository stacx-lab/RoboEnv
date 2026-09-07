"""The simulator contract: every simulator joins the harness as an *adapter*.

An adapter is the only thing a simulator must provide.  Heavy integrations
(Isaac boot, motion stacks, asset handling) stay in their own repositories;
the adapter imports them lazily and exposes:

* identity — ``name`` (registry key) and ``embodiment`` (which skill domain
  applies, e.g. ``"arx_x5"``), plus declared ``capabilities``;
* context — :meth:`EnvAdapter.situation`, :meth:`EnvAdapter.facts`,
  :meth:`EnvAdapter.goal` prompt contributions;
* tools — :meth:`EnvAdapter.tools`, optional compatibility primitives, alongside device bindings;
* observation — :meth:`EnvAdapter.observe`, a normalised ground-truth snapshot;
* lifecycle — :meth:`EnvAdapter.reset` / :meth:`EnvAdapter.close`;
* evaluation — :meth:`EnvAdapter.score` and :meth:`EnvAdapter.budget`;
* media — :meth:`EnvAdapter.frames` / :meth:`EnvAdapter.capture`.

The protocol below is the *complete* surface the harness consumes.  (Its
predecessor listed only half of it and every consumer reached for the rest
with ``getattr``; that is the ambiguity this module removes.)

Adapters register themselves with :func:`register_adapter`; built-in ones are
imported lazily by :func:`make_adapter`, so a new simulator is one new module
under :mod:`roboenv_runtime.adapters` and one line in :data:`BUILTIN_ADAPTERS`.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from roboenv_runtime.core.tools import ToolBox
from roboenv_runtime.core.types import Budget, ScoreReport

__all__ = [
    "BUILTIN_ADAPTERS",
    "AdapterBase",
    "EnvAdapter",
    "SupportsCheckpoint",
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
        embodiment: Skill domain of the robot being driven (``"arx_x5"``,
            ``"piper"``, ``"panda"``, ``"yam"``, ...).
        capabilities: Optional features, see :class:`roboenv_runtime.core.Capability`.
    """

    name: str
    embodiment: str
    capabilities: frozenset[str]

    def situation(self) -> str:
        """Prompt section: what world and robot the agent is in."""
        ...

    def facts(self) -> str:
        """Prompt section: environment mechanics worth knowing up front."""
        ...

    def goal(self) -> str:
        """The benchmark-provided language goal, or ``""`` when tasks carry none."""
        ...

    def reset(self, seed: int) -> None:
        """Start a fresh episode."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        """Agent-callable functions keyed by public name (docstrings are the docs)."""
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


@runtime_checkable
class SupportsCheckpoint(Protocol):
    """Adapters whose world can be rewound in place (``Capability.CHECKPOINT``)."""

    def checkpoint(self) -> Any:
        """Snapshot the world; returns an opaque token."""
        ...

    def restore(self, token: Any) -> None:
        """Rewind the world to a previous snapshot."""
        ...


class AdapterBase(ABC):
    """Convenience base class implementing the optional parts of the contract.

    Subclasses must implement :meth:`reset` and :meth:`score`; everything else
    has a sensible default.  ``tools()`` collects ``@tool``-marked methods;
    ``observe()`` derives a snapshot from the conventional
    ``list_objects`` / ``get_object_pose`` / ``get_gripper_pose*`` tools.
    """

    name: ClassVar[str] = "base"
    embodiment: ClassVar[str] = "general"
    capabilities: ClassVar[frozenset[str]] = frozenset()
    situation_text: ClassVar[str] = ""
    facts_text: ClassVar[str] = ""

    # ------------------------------------------------------------ context

    def situation(self) -> str:
        """Default: the class-level ``situation_text``."""
        return self.situation_text

    def facts(self) -> str:
        """Default: the class-level ``facts_text``."""
        return self.facts_text

    def goal(self) -> str:
        """Default: no benchmark-provided goal text."""
        return ""

    # ------------------------------------------------------------ lifecycle

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Start a fresh episode."""

    def close(self) -> None:
        """Default: nothing to release."""
        return None

    # ------------------------------------------------------------ evaluation

    @abstractmethod
    def score(self) -> ScoreReport:
        """Evaluate the current world state."""

    def budget(self) -> Budget:
        """Default: an unbounded placeholder; real adapters override."""
        return Budget()

    # ------------------------------------------------------------ tools

    def tools(self) -> Mapping[str, Callable[..., Any]]:
        """Every ``@tool``-marked method, keyed by tool name."""
        return ToolBox.collect(self).namespace()

    def toolbox(self) -> ToolBox:
        """The tools as a :class:`ToolBox` (documentation, subsetting)."""
        return ToolBox.from_mapping(self.tools())

    # ------------------------------------------------------------ observation

    def observe(self) -> dict[str, Any]:
        """A normalised snapshot built from the conventional observation tools.

        Returns a dict with ``objects`` (``name -> {position, quaternion_wxyz}``),
        ``grippers`` (``arm -> {tip_position}``) and ``budget``.  Every read is
        guarded so a partially-implemented adapter still yields a usable
        snapshot; missing pieces are simply absent.
        """
        tools = self.tools()
        out: dict[str, Any] = {"objects": {}, "grippers": {}, "budget": self.budget().as_dict()}
        list_fn = tools.get("list_objects")
        pose_fn = tools.get("get_object_pose")
        if callable(list_fn) and callable(pose_fn):
            try:
                names = list(list_fn())
            except Exception:
                names = []
            for name in names:
                try:
                    pos, quat = pose_fn(name)
                    out["objects"][name] = {
                        "position": [float(v) for v in list(pos)[:3]],
                        "quaternion_wxyz": [float(v) for v in list(quat)[:4]],
                    }
                except Exception:
                    out["objects"][name] = {"position": None, "quaternion_wxyz": None}
        for arm in ("left", "right"):
            fn = tools.get(f"get_gripper_pose_{arm}")
            try:
                if callable(fn):
                    pose = list(fn())
                elif callable(tools.get("get_gripper_pose")):
                    pose = list(tools["get_gripper_pose"](arm))
                else:
                    continue
                out["grippers"][arm] = {"tip_position": [float(v) for v in pose[:3]]}
            except Exception:
                continue
        return out

    # ------------------------------------------------------------ media

    def frames(self) -> list[Any]:
        """Default: no video capture."""
        return []

    def capture(self) -> Any | None:
        """Default: the most recent captured frame, if any."""
        frames = self.frames()
        return frames[-1] if frames else None

    # ------------------------------------------------------------ compat / info

    @property
    def domain(self) -> str:
        """Alias of ``embodiment`` kept for pre-refactor callers."""
        return self.embodiment

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description (used by ``roboenv adapters`` and workers)."""
        return {
            "name": self.name,
            "embodiment": self.embodiment,
            "capabilities": sorted(self.capabilities),
            "tools": [spec.__dict__ for spec in self.toolbox().specs()],
        }


# ------------------------------------------------------------------ registry

_REGISTRY: dict[str, type] = {}

BUILTIN_ADAPTERS: dict[str, str] = {
    "fakeshell": "roboenv_runtime.adapters.fakeshell",
    "robodojo": "roboenv_runtime.adapters.robodojo",
    "robotwin": "roboenv_runtime.adapters.robotwin",
    "libero": "roboenv_runtime.adapters.libero",
    "robocasa": "roboenv_runtime.adapters.robocasa",
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
                "Activate the simulator's interpreter (see docs/REFACTOR_PLAN.md §5)."
            ) from exc
        if name not in _REGISTRY:
            raise KeyError(f"{module} imported but did not register adapter {name!r}")
    adapter = _REGISTRY[name](**kwargs)
    try:  # pretrained policies as tools (roboenv_runtime.vla) — optional, never fatal
        from roboenv_runtime.vla.attach import attach_vla_tool

        if name != "robocasa":
            attach_vla_tool(adapter, name)
    except Exception as exc:
        logging.getLogger(__name__).debug("vla tool not attached to %s: %s", name, exc)
    return adapter
