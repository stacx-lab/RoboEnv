#!/usr/bin/env python3
"""Independent Harbor verifier. Standard library only; never imports /app code.

The simulator's live verdict plus its read-only supervisor journal are evidence.
This verifies one trial, not a fresh-process replay certificate.
"""

import hashlib
import json
import math
import os
import socket
import uuid
from pathlib import Path


def strict_json(raw):
    """Reject ambiguous keys and non-finite numeric values at every JSON boundary."""

    def reject(value):
        raise ValueError(f"non-finite JSON: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    return json.loads(
        raw, parse_constant=reject, parse_float=finite_float, object_pairs_hook=unique
    )


def rpc(op, session, args=None):
    """Send an identity-checked request without importing any client code."""
    request = {
        "protocol": "roboenv/1",
        "session": session,
        "generation": 0,
        "request_id": f"verify-{uuid.uuid4().hex}",
        "deadline_s": 240.0,
        "op": op,
        "args": args or {},
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(245)
        sock.connect(os.environ.get("ROBOENV_SOCKET", "/run/roboenv/sim.sock"))
        sock.sendall(json.dumps(request).encode() + b"\n")
        with sock.makefile("rb") as stream:
            raw = stream.readline(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("incomplete or oversized response")
    response = strict_json(raw)
    if not isinstance(response, dict) or type(response.get("generation")) is not int:
        raise ValueError("response must contain a typed episode identity")
    for field in ("protocol", "session", "generation", "request_id"):
        if response.get(field) != request[field]:
            raise ValueError(f"response {field} mismatch")
    if response.get("ok") is not True:
        raise ValueError(f"{op} failed: {response.get('error')}")
    return response["result"], request["request_id"]


def attach_budget(verdict, observation):
    """Use the frozen world's independently observed budget, including ScoreClock."""
    budget = observation.get("budget", {})
    if "budget" in verdict:
        for field in ("used", "max", "unit"):
            if verdict["budget"].get(field) != budget.get(field):
                raise ValueError("submit and observe disagree about final budget")
    return {**verdict, "budget": budget}


def validate_score(result):
    """Validate typed score/budget data; return full-success eligibility."""
    reward = result.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("reward must be numeric")
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("reward must be finite and normalized")
    if not isinstance(result.get("completed"), bool) or result.get("submitted") is not True:
        raise ValueError("missing typed completion/submission verdict")
    budget = result.get("budget", {})
    if type(budget.get("used")) is not int or budget["used"] < 0:
        raise ValueError("invalid budget.used")
    if "max" not in budget:
        raise ValueError("missing budget.max")
    maximum = budget["max"]
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("invalid budget.max")
    if not isinstance(budget.get("unit"), str) or not budget["unit"]:
        raise ValueError("budget needs a named unit")
    if set(budget) != {"used", "max", "unit"}:
        raise ValueError("invalid budget fields")
    return bool(
        result["completed"]
        and reward >= 1 - 1e-9
        and (maximum is None or budget["used"] < maximum)
    )


def validate_journal(raw, *, session, submit_id, verdict, practice=False):
    """Reject incomplete history, resets, repeated submits and post-submit motion."""
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("missing or truncated supervisor journal")
    pending, finished = {}, set()
    submitted = False
    count = 0
    forbidden = {"reset", "restore", "checkpoint", "shutdown"}
    if practice:
        forbidden -= {"reset", "restore", "checkpoint"}
    for line in raw.splitlines():
        event = strict_json(line)
        request = event["request"]
        ident = request["request_id"]
        if not isinstance(ident, str) or not ident or not isinstance(request.get("args"), dict):
            raise ValueError("invalid journal request identity or arguments")
        if (
            request.get("session") != session
            or type(request.get("generation")) is not int
            or request["generation"] != 0
        ):
            raise ValueError("journal episode identity changed")
        if (
            request.get("protocol") != "roboenv/1"
            or type(event.get("journal_version")) is not int
            or event["journal_version"] != 1
        ):
            raise ValueError("unexpected journal protocol")
        phase = event["phase"]
        if phase == "intent":
            if ident in pending or ident in finished:
                raise ValueError("duplicate journal intent")
            pending[ident] = request
            continue
        if phase not in ("commit", "error") or pending.pop(ident, None) != request:
            raise ValueError("unmatched journal terminal event")
        finished.add(ident)
        count += 1
        response = event["response"]
        if type(response.get("generation")) is not int:
            raise ValueError("journal response episode identity must be typed")
        for field in ("protocol", "session", "generation", "request_id"):
            if response.get(field) != request[field]:
                raise ValueError("journal response identity mismatch")
        op = request["op"]
        if op in forbidden:
            raise ValueError(f"operation forbidden in a single-episode trial: {op}")
        read_only = op in {
            "observe",
            "score.peek",
            "capture",
            "health",
            "environment.describe",
            "devices.list",
            "devices.describe",
            "operations.get",
        }
        if op == "devices.call":
            # devices/1 enforces that these actions cannot dispatch mutating tools.
            read_only = request["args"].get("action") in {
                "read",
                "state",
                "status",
                "load",
                "unload",
                "ik",
                "fk",
            }
        if (
            phase == "error"
            and response.get("error", {}).get("details", {}).get("effects") == "none"
        ):
            read_only = True  # explicit server-side rejection before dispatch, any operation
        if submitted and not read_only:
            raise ValueError("mutation or repeated submit after final verdict")
        if phase == "error":
            if response.get("ok") is not False:
                raise ValueError("journal error has a successful response")
            if not read_only:
                raise ValueError("failed motion leaves its effects indeterminate")
            continue
        if response.get("ok") is not True:
            raise ValueError("journal commit has no successful response")
        if op == "submit":
            if ident != submit_id or response.get("result") != verdict:
                raise ValueError("submit was not the verifier's sole evaluation")
            submitted = True
    if pending or not submitted:
        raise ValueError("unfinished actions or missing verifier submit")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "requests": count}


def retain_journal(out):
    """Preserve the original bytes even when verification rejects the trial."""
    journal = Path(os.environ.get("ROBOENV_RUN_DIR", "/run/roboenv/run")) / "journal/actions.jsonl"
    raw = journal.read_bytes()
    (out / "actions.jsonl").write_bytes(raw)
    return raw


def main():
    """Write a fail-closed live-trial verdict and its independent evidence."""
    out = Path(os.environ.get("ROBOENV_VERIFIER_OUT", "/logs/verifier"))
    out.mkdir(parents=True, exist_ok=True)
    # A malformed/missing task definition must not leave an old success behind.
    (out / "reward.json").write_text('{"reward": 0.0}\n')
    (out / "reward.txt").write_text("0.0\n")
    report = {"certified": False, "benchmark_eligible": False}
    reward, valid = 0.0, False
    raw = None
    try:
        task = strict_json(Path(__file__).with_name("task.json").read_bytes())
        identity_path = Path(os.environ.get("ROBOENV_SOCKET", "/run/roboenv/sim.sock"))
        identity = strict_json(identity_path.with_name("identity.json").read_bytes())
        task["session"] = identity["session"]
        mode = task.get("budget_mode", "official")
        if (
            type(task.get("seed")) is not int
            or task["seed"] < 0
            or mode not in {"official", "practice"}
            or any(
                not isinstance(task.get(key), str) or not task[key]
                for key in ("task_id", "session")
            )
        ):
            raise ValueError("invalid verifier task identity or budget mode")
        report.update(
            task=task["task_id"],
            seed=task["seed"],
            budget_mode=mode,
            benchmark_eligible=mode == "official",
        )
        health, _ = rpc("health", task["session"])
        report["health"] = health
        if type(health.get("seed")) is not int or (
            health.get("task"),
            health.get("seed"),
            health.get("budget_mode"),
        ) != (
            task["task_id"],
            task["seed"],
            mode,
        ):
            raise ValueError("server task/seed/budget identity differs from task.json")
        verdict, submit_id = rpc("submit", task["session"])
        report["submit"] = verdict
        observation, _ = rpc("observe", task["session"])
        report["observation"] = observation
        raw = retain_journal(out)
        success = validate_score(attach_budget(verdict, observation))
        report["journal"] = validate_journal(
            raw,
            session=task["session"],
            submit_id=submit_id,
            verdict=verdict,
            practice=mode == "practice",
        )
        valid, reward = True, float(success)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    if raw is None:
        try:
            raw = retain_journal(out)
        except OSError as exc:
            report["journal_capture_error"] = f"{type(exc).__name__}: {exc}"
    if raw is not None:
        report["journal_artifact"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "validated": valid,
        }
    report.update(valid=valid, trial_success=bool(reward))
    (out / "submit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "reward.json").write_text(json.dumps({"reward": reward}) + "\n")
    (out / "reward.txt").write_text(f"{reward}\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
