"""Persist sealed owner evidence without changing the authoritative verdict."""

import hashlib

from .atomic import atomic_write_json


def save_terminal_evidence(directory, observation, verdict, *, session, diagnostics=None):
    native = observation.get("schema") == "roboenv/native-device-episode/1"
    report = {"schema": "roboenv/terminal-evidence/1", "complete": True, "files": {}}

    def save(name, value):
        entry = {"path": name, "complete": False}
        report["files"][name] = entry
        try:
            atomic_write_json(directory / name, value, indent=None)
            entry.update(complete=True, sha256=hashlib.sha256((directory / name).read_bytes()).hexdigest())
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["complete"] = False
        return entry

    snapshot = save("native-episode.json" if native else "final-observation.json", observation)
    if diagnostics is not None:
        try:
            detail = diagnostics()
        except Exception as exc:
            detail = {"complete": False, "error": f"{type(exc).__name__}: {exc}"}
        detail.update(
            schema="roboenv/native-diagnostics/1", session=session,
            task=observation["task"], seed=observation["seed"],
            budget=observation["budget"], verdict=verdict,
            observation=snapshot,
            state_sha256=observation.get("state", {}).get("sha256"),
            implementation=observation.get("implementation"),
        )
        save("native-diagnostics.json", detail)
        report["diagnostics_complete"] = detail.get("complete", False)
    return report
