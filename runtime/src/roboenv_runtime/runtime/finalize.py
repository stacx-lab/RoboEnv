"""Offline, fail-closed finalization of one immutable run.

The finalizer never contacts a worker.  It derives ``result.json`` exclusively
from the run's ``manifest.json`` and the durable supervisor action journal
(``journal/actions.jsonl``), then writes it atomically — and refuses to write
anything when the evidence is inconsistent (no submit, a mutating call after
submit, a budget that regressed, an official run over its cap ...).
"""

# Adapted from codex_dojo_alltasks/harness/finalize_run.py on 2026-08-26 (refactor P3).

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboenv_runtime.runtime.atomic import atomic_write_json
from roboenv_runtime.runtime.protocol import ProtocolError, Request, Response

RESULT_SCHEMA_VERSION = 1
SUCCESS_THRESHOLD = 1.0 - 1e-3
JOURNAL_VERSION = 1

_PHASES = frozenset({"intent", "commit", "error"})
_POST_SUBMIT_READ_ONLY_OPS = frozenset(
    {"health", "api", "observe", "score.peek", "capture", "shutdown"}
)
_READ_ONLY_CALLS = frozenset(
    {
        "list_objects",
        "get_object_pose",
        "get_object_category",
        "get_object_bbox_extent",
        "get_gripper_pose",
        "get_gripper_state",
        "check_reachable",
        "get_object_semantics",
        "get_step_budget",
    }
)


class FinalizationError(ValueError):
    """The run lacks sufficient, internally consistent terminal evidence."""


def _reject_json_constant(token: str) -> None:
    raise FinalizationError(f"non-finite JSON constant is forbidden: {token}")


def _strict_json(data: str, *, source: Path) -> Any:
    try:
        return json.loads(data, parse_constant=_reject_json_constant)
    except FinalizationError:
        raise
    except json.JSONDecodeError as exc:
        raise FinalizationError(f"invalid JSON in {source}: {exc}") from exc


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FinalizationError(f"cannot read {label} {path}: {exc}") from exc
    value = _strict_json(raw, source=path)
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object: {path}")
    return value


def _below(path: Path, root: Path, *, require_exists: bool = True) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=require_exists)
    if resolved == root or root not in resolved.parents:
        raise FinalizationError(f"path must be below {root}: {resolved}")
    return resolved


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizationError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    result = _nonnegative_int(value, field=field)
    if result == 0:
        raise FinalizationError(f"{field} must be positive")
    return result


def _normalize_budget(raw: Any, *, source: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise FinalizationError(f"{source} budget must be an object")
    used = _nonnegative_int(raw.get("used"), field=f"{source}.budget.used")
    maximum = _positive_int(raw.get("max"), field=f"{source}.budget.max")
    if used > maximum:
        raise FinalizationError(f"{source} budget exceeds its clamp: used={used}, max={maximum}")
    expected_remaining = maximum - used
    for name in ("remaining", "remaining_steps"):
        if name not in raw:
            continue
        observed = _nonnegative_int(raw[name], field=f"{source}.budget.{name}")
        if observed != expected_remaining:
            raise FinalizationError(
                f"{source}.budget.{name}={observed} conflicts with max-used={expected_remaining}"
            )
    return {
        "used": used,
        "max": maximum,
        "remaining": expected_remaining,
        "remaining_steps": expected_remaining,
    }


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FinalizationError(f"cannot read action journal {path}: {exc}") from exc
    if not data:
        raise FinalizationError("action journal is empty")
    if not data.endswith(b"\n"):
        raise FinalizationError("action journal has an incomplete final line")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizationError(f"action journal is not UTF-8: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        value = _strict_json(line, source=path)
        if not isinstance(value, dict):
            raise FinalizationError(f"journal line {line_number} is not a JSON object")
        events.append(value)
    return events, _sha256(data)


def _request_key(request: Request) -> tuple[str, int, str]:
    return (request.session, request.generation, request.request_id)


def _parse_journal(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    intents: dict[tuple[str, int, str], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    phase_counts = {"intent": 0, "commit": 0, "error": 0}
    for event_index, event in enumerate(events, 1):
        if event.get("journal_version") != JOURNAL_VERSION:
            raise FinalizationError(f"journal event {event_index} has unsupported journal_version")
        phase = event.get("phase")
        if phase not in _PHASES:
            raise FinalizationError(f"journal event {event_index} has invalid phase {phase!r}")
        phase_counts[phase] += 1
        raw_request = event.get("request")
        if not isinstance(raw_request, Mapping):
            raise FinalizationError(f"journal event {event_index} lacks a request object")
        try:
            request = Request.from_mapping(raw_request)
        except ProtocolError as exc:
            raise FinalizationError(
                f"journal event {event_index} has invalid request: {exc}"
            ) from exc
        key = _request_key(request)
        if phase == "intent":
            if key in intents:
                raise FinalizationError(f"duplicate intent for request key {key!r}")
            intents[key] = {
                "request_mapping": request.as_dict(),
                "intent_event": event_index,
                "terminal": None,
            }
            continue
        state = intents.get(key)
        if state is None:
            raise FinalizationError(f"terminal event {event_index} has no preceding intent")
        if state["terminal"] is not None:
            raise FinalizationError(f"request key {key!r} has more than one terminal event")
        if request.as_dict() != state["request_mapping"]:
            raise FinalizationError(f"terminal event {event_index} changed its intent request")
        raw_response = event.get("response")
        if not isinstance(raw_response, Mapping):
            raise FinalizationError(f"terminal event {event_index} lacks a response object")
        try:
            response = Response.from_mapping(raw_response)
        except ProtocolError as exc:
            raise FinalizationError(
                f"terminal event {event_index} has invalid response: {exc}"
            ) from exc
        if (
            response.session != request.session
            or response.generation != request.generation
            or response.request_id != request.request_id
        ):
            raise FinalizationError(f"terminal event {event_index} response identity mismatch")
        if (phase == "commit") != response.ok:
            raise FinalizationError(f"terminal event {event_index} phase/ok mismatch")
        action = {
            "request": request,
            "response": response,
            "phase": phase,
            "intent_event": state["intent_event"],
            "terminal_event": event_index,
            "timestamp": event.get("ts"),
        }
        state["terminal"] = action
        completed.append(action)
    dangling = [key for key, state in intents.items() if state["terminal"] is None]
    if dangling:
        raise FinalizationError(
            f"journal contains {len(dangling)} intent(s) without terminal events"
        )
    return completed, phase_counts


def _successful(actions: Sequence[Mapping[str, Any]], op: str) -> list[Mapping[str, Any]]:
    return [
        action for action in actions if action["phase"] == "commit" and action["request"].op == op
    ]


def _result_mapping(action: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = action["response"].result
    if not isinstance(result, dict):
        raise FinalizationError(f"{label} response result must be an object")
    return result


def _action_evidence(action: Mapping[str, Any], result: Any) -> dict[str, Any]:
    request: Request = action["request"]
    return {
        "op": request.op,
        "request_id": request.request_id,
        "session": request.session,
        "generation": request.generation,
        "intent_event": action["intent_event"],
        "terminal_event": action["terminal_event"],
        "timestamp": action["timestamp"],
        "result": result,
    }


def _is_read_only_call(request: Request) -> bool:
    if request.op == "devices.call":
        return request.args.get("action") in {
            "read", "state", "status", "load", "catalog", "api", "ping"
        }
    name = request.args.get("name", request.args.get("fn"))
    if not isinstance(name, str):
        return False
    from roboenv_runtime.runtime.worker import SimWorker

    return name in _READ_ONLY_CALLS or SimWorker.is_read_only(name)


def _is_physical_mutation(request: Request) -> bool:
    if request.op in {"call", "devices.call"}:
        return not _is_read_only_call(request)
    return request.op in {"vla", "reset", "reset_same_task", "restore", "exec"}


def _latest_budget(
    successful_actions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], Mapping[str, Any]]:
    candidates: list[tuple[dict[str, int], Mapping[str, Any]]] = []
    for action in successful_actions:
        request: Request = action["request"]
        result = action["response"].result
        raw_budget: Any = None
        if isinstance(result, Mapping) and isinstance(result.get("budget"), Mapping):
            raw_budget = result["budget"]
        elif (
            request.op == "call"
            and request.args.get("name", request.args.get("fn")) == "get_step_budget"
            and isinstance(result, Mapping)
        ):
            raw_budget = result
        if raw_budget is not None:
            candidates.append(
                (
                    _normalize_budget(
                        raw_budget,
                        source=f"journal event {action['terminal_event']}",
                    ),
                    action,
                )
            )
    if not candidates:
        raise FinalizationError("journal contains no successful budget snapshot")
    return candidates[-1]


def _validate_manifest(run_dir: Path) -> tuple[dict[str, Any], bytes]:
    manifest_path = _below(run_dir / "manifest.json", run_dir)
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_object(manifest_path, label="manifest")
    if manifest.get("schema_version") != 1:
        raise FinalizationError("manifest schema_version must be 1")
    task = manifest.get("task")
    if not isinstance(task, str) or not task:
        raise FinalizationError("manifest task must be a non-empty string")
    if manifest.get("budget_mode") not in {"official", "practice"}:
        raise FinalizationError("manifest budget_mode must be official or practice")
    _nonnegative_int(manifest.get("seed"), field="manifest.seed")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise FinalizationError("manifest run_id must be non-empty")
    frames = manifest.get("official_action_frames")
    cap = manifest.get("official_sim_step_cap")
    if frames is not None:
        frames = _positive_int(frames, field="manifest.official_action_frames")
        if cap is None:
            cap = frames * 10
    if cap is not None:
        cap = _positive_int(cap, field="manifest.official_sim_step_cap")
        if frames is not None and cap != frames * 10:
            raise FinalizationError("manifest simulator-step cap conflicts with action frames")
    if manifest["budget_mode"] == "official" and cap is None:
        raise FinalizationError(
            "an official run needs official_action_frames/official_sim_step_cap"
        )
    manifest["official_sim_step_cap"] = cap
    manifest["official_action_frames"] = frames
    return manifest, manifest_bytes


def _resolve_run(run: str | os.PathLike[str]) -> Path:
    """An absolute run directory, or ``<task_id>/<run_id>`` / ``<run_id>`` below ``runs/``."""
    from roboenv_runtime.paths import runs_root

    raw_run = Path(run)
    if not raw_run.is_absolute() and (raw_run / "manifest.json").is_file():
        raw_run = raw_run.resolve()  # a run directory given relative to the cwd
    if not raw_run.is_absolute():
        root = runs_root().resolve()
        candidate = root / raw_run
        if not candidate.is_dir():
            matches = [p for p in root.glob(f"*/{raw_run}") if p.is_dir()]
            if len(matches) == 1:
                candidate = matches[0]
        raw_run = candidate
    run_dir = raw_run.resolve()
    if not run_dir.is_dir():
        raise FinalizationError(f"run directory does not exist: {run_dir}")
    return run_dir


def build_result(run: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a run and return its deterministic result without writing it."""

    run_dir = _resolve_run(run)
    manifest, manifest_bytes = _validate_manifest(run_dir)
    if not (run_dir / "journal" / "actions.jsonl").is_file():
        raise FinalizationError(
            f"run has no action journal: {run_dir / 'journal' / 'actions.jsonl'}"
        )
    journal_path = _below(run_dir / "journal" / "actions.jsonl", run_dir)
    events, journal_sha = _read_journal(journal_path)
    actions, phase_counts = _parse_journal(events)
    successful_actions = [action for action in actions if action["phase"] == "commit"]

    submit_attempts = [action for action in actions if action["request"].op == "submit"]
    if len(submit_attempts) != 1:
        raise FinalizationError(
            f"run must contain exactly one submit attempt, found {len(submit_attempts)}"
        )
    submit = submit_attempts[0]
    if submit["phase"] != "commit":
        raise FinalizationError("the unique submit attempt did not commit successfully")
    if submit["request"].args:
        raise FinalizationError("submit request must have empty args")
    submit_result = _result_mapping(submit, label="submit")
    for name in ("submitted", "completed", "latched"):
        if submit_result.get(name) is not True:
            raise FinalizationError(f"successful submit requires {name}=true")
    reward = submit_result.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) < SUCCESS_THRESHOLD
    ):
        raise FinalizationError("successful submit requires reward >= 0.999")

    for action in successful_actions:
        if action["terminal_event"] <= submit["terminal_event"]:
            continue
        request: Request = action["request"]
        allowed = request.op in (_POST_SUBMIT_READ_ONLY_OPS | {
            "devices.list", "devices.describe", "environment.describe", "operations.get"
        }) or (
            request.op in {"call", "devices.call"} and _is_read_only_call(request)
        )
        if not allowed:
            raise FinalizationError(f"successful mutating op {request.op!r} occurred after submit")

    observes = _successful(actions, "observe")
    if not observes:
        raise FinalizationError("run has no successful terminal observe")
    last_observe = observes[-1]
    observe_result = _result_mapping(last_observe, label="last observe")
    if observe_result.get("task") not in (None, manifest["task"]):
        raise FinalizationError("last observe task conflicts with manifest")
    if observe_result.get("seed") not in (None, manifest["seed"]):
        raise FinalizationError("last observe seed conflicts with manifest")
    if observe_result.get("budget_mode") not in (
        None,
        manifest["budget_mode"],
    ):
        raise FinalizationError("last observe budget_mode conflicts with manifest")
    observe_budget = _normalize_budget(observe_result.get("budget"), source="last observe")
    # A tool may move the world before returning an error. Require the
    # terminal observation to follow every mutating attempt, not only commits.
    last_mutation = max(
        (
            action["terminal_event"]
            for action in actions
            if _is_physical_mutation(action["request"])
        ),
        default=0,
    )
    if last_observe["terminal_event"] < last_mutation:
        raise FinalizationError("last observe predates the final physical mutation")

    for reset in _successful(actions, "reset_same_task") + _successful(actions, "reset"):
        reset_result = _result_mapping(reset, label="reset")
        if (
            reset_result.get("task") != manifest["task"]
            or reset_result.get("seed") != manifest["seed"]
        ):
            raise FinalizationError("reset_same_task changed manifest task or seed")

    score_actions = [
        action for action in successful_actions if action["request"].op in {"score.peek", "submit"}
    ]
    if not score_actions:
        raise FinalizationError("journal contains no final score snapshot")
    last_score = score_actions[-1]
    score_result = _result_mapping(last_score, label="last score")
    if score_result.get("completed") is not True or score_result.get("latched") is not True:
        raise FinalizationError("last score snapshot contradicts successful submit")

    budget, budget_action = _latest_budget(successful_actions)
    if (
        budget["used"] < observe_budget["used"]
        and budget_action["terminal_event"] >= last_observe["terminal_event"]
    ):
        raise FinalizationError("latest budget regressed after terminal observe")
    official_cap = manifest.get("official_sim_step_cap")
    if manifest["budget_mode"] == "official" and budget["max"] != official_cap:
        raise FinalizationError(
            f"official budget max {budget['max']} does not equal cap {official_cap}"
        )
    within_official_cap = official_cap is not None and budget["used"] <= official_cap
    if manifest["budget_mode"] == "official" and not within_official_cap:
        raise FinalizationError("official run exceeded its simulator-step cap")

    failed_vla = [
        action
        for action in actions
        if action["request"].op == "vla" and action["phase"] != "commit"
    ]
    if failed_vla:
        # A raised VLA request may execute a prefix but its error envelope
        # carries no trustworthy actions_taken count.
        raise FinalizationError("failed VLA request makes the total action count indeterminate")

    vla_total = 0
    vla_entries: list[dict[str, Any]] = []
    for action in _successful(actions, "vla"):
        result = _result_mapping(action, label="vla")
        taken = _nonnegative_int(result.get("actions_taken"), field="vla.result.actions_taken")
        vla_total += taken
        vla_entries.append(
            {
                "request_id": action["request"].request_id,
                "terminal_event": action["terminal_event"],
                "actions_taken": taken,
            }
        )

    seed = int(manifest["seed"])
    if manifest["budget_mode"] == "practice":
        ledger_candidate: str | None = "practice_full"
    elif seed == 0:
        ledger_candidate = "official_seed0_full"
    else:
        ledger_candidate = None

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run": {
            "run_id": manifest["run_id"],
            "task": manifest["task"],
            "seed": seed,
            "budget_mode": manifest["budget_mode"],
            "mechanism": manifest.get("mechanism"),
            "adapter": manifest.get("adapter"),
            "physical_gpu": manifest.get("physical_gpu"),
        },
        "outcome": {
            "success": True,
            "evaluator_completed": True,
            "ledger_status_candidate": ledger_candidate,
        },
        "official_budget": {
            "action_frames": manifest.get("official_action_frames"),
            "sim_step_cap": official_cap,
            "within_cap": within_official_cap,
            "eligible": (manifest["budget_mode"] == "official" and within_official_cap),
        },
        "budget": {
            **budget,
            "source_op": budget_action["request"].op,
            "source_request_id": budget_action["request"].request_id,
            "source_terminal_event": budget_action["terminal_event"],
        },
        "vla": {
            "calls": len(vla_entries),
            "actions_taken_total": vla_total,
            "entries": vla_entries,
        },
        "submit": _action_evidence(submit, submit_result),
        "last_score": _action_evidence(last_score, score_result),
        "last_observe": _action_evidence(last_observe, observe_result),
        "journal": {
            "path": "journal/actions.jsonl",
            "sha256": journal_sha,
            "events": len(events),
            "requests": phase_counts["intent"],
            "commits": phase_counts["commit"],
            "errors": phase_counts["error"],
        },
        "provenance": {
            "manifest_sha256": _sha256(manifest_bytes),
            "submit_terminal_event": submit["terminal_event"],
        },
    }


def finalize_run(run: str | os.PathLike[str]) -> dict[str, Any]:
    """Create result.json exactly once and return the validated result."""

    result = build_result(run)
    run_dir = _resolve_run(run)
    output = _below(run_dir / "result.json", run_dir, require_exists=False)
    if output.exists():
        existing = _load_object(output, label="existing result")
        if existing != result:
            raise FinalizationError(
                "result.json already exists and does not match current journal evidence"
            )
        return existing
    atomic_write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="absolute path, or <task_id>/<run_id> / <run_id> below runs/",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_run(args.run_dir)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":") if args.compact else None,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
