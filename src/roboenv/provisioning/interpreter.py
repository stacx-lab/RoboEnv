"""Resolve prepared simulator interpreters without installing dependencies."""

import os
from pathlib import Path

from roboenv.validation import json_loads


def resolve_interpreter(project, backend, interpreter=None):
    """An explicit executable takes precedence over the project's registration."""
    if interpreter is None:
        if not isinstance(backend, str) or not backend or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in backend
        ):
            raise ValueError("Invalid simulator backend")
        registration = Path(project) / ".cache/roboenv/installations" / f"{backend}.json"
        guidance = f"run 'roboenv prepare {backend}' or pass --interpreter"
        if not registration.is_file():
            raise ValueError(f"Simulator is not prepared; {guidance}")
        try:
            record = json_loads(registration.read_text())
            interpreter = record["python"]
            if not isinstance(interpreter, str) or not interpreter:
                raise ValueError("missing interpreter")
            if record["backend"] != backend:
                raise ValueError("backend mismatch")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid simulator registration; {guidance}") from exc
    interpreter = Path(interpreter).absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(
            "Simulator interpreter must be an existing executable; "
            f"run 'roboenv prepare {backend}' or pass --interpreter"
        )
    return interpreter
