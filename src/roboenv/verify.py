"""Reference certification of one run: its final score and its supervisor journal.

RoboEnv owns the verdict, so it also publishes the rules that make a run's evidence
certifiable. This module states those rules and applies them to one run end to end.
It uses only the standard library and imports nothing else from this package: a
harness or benchmark imports it, or copies this one file into a verifier container,
instead of re-implementing journal validation.

The simulator's live verdict plus its read-only supervisor journal are the evidence.
This certifies one run; it is not a fresh-process replay certificate.
"""

import hashlib
import json
import math
import socket
import uuid
from pathlib import Path

PROTOCOL = "roboenv/1"
JOURNAL_VERSION = 1
BUDGET_MODES = frozenset({"official", "practice"})
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
# Operations that never dispatch to the world, and devices/1 actions that cannot mutate.
READ_ONLY_OPS = frozenset(
    {
        "observe",
        "score.peek",
        "capture",
        "health",
        "environment.describe",
        "devices.list",
        "devices.describe",
        "operations.get",
    }
)
READ_ONLY_ACTIONS = frozenset({"read", "state", "status", "load", "unload", "ik", "fk"})
# Never certifiable in an official episode; practice runs may reset, restore and checkpoint.
FORBIDDEN_OPS = frozenset({"reset", "restore", "checkpoint", "shutdown"})
PRACTICE_OPS = frozenset({"reset", "restore", "checkpoint"})


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


def rpc(socket_path, op, *, session, generation=0, args=None, deadline_s=240.0):
    """Send one identity-checked owner request over the Unix socket; no client code."""
    request = {
        "protocol": PROTOCOL,
        "session": session,
        "generation": generation,
        "request_id": f"verify-{uuid.uuid4().hex}",
        "deadline_s": deadline_s,
        "op": op,
        "args": args or {},
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(deadline_s + 5)
        sock.connect(str(socket_path))
        sock.sendall(json.dumps(request).encode() + b"\n")
        with sock.makefile("rb") as stream:
            raw = stream.readline(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES or not raw.endswith(b"\n"):
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


def validate_journal(raw, *, session, submit_id, verdict, practice=False, generation=0):
    """Reject incomplete history, resets, repeated submits and post-submit motion.

    Every request needs an intent and exactly one commit or error with a matching
    identity. The verifier's own submit must be the only evaluation, and nothing may
    mutate the world after it; a rejection before dispatch (``effects: none``) is not
    a mutation. Returns the journal digest and the number of finished requests.
    """
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("missing or truncated supervisor journal")
    pending, finished = {}, set()
    submitted = False
    count = 0
    forbidden = FORBIDDEN_OPS - PRACTICE_OPS if practice else FORBIDDEN_OPS
    for line in raw.splitlines():
        event = strict_json(line)
        request = event["request"]
        ident = request["request_id"]
        if not isinstance(ident, str) or not ident or not isinstance(request.get("args"), dict):
            raise ValueError("invalid journal request identity or arguments")
        if (
            request.get("session") != session
            or type(request.get("generation")) is not int
            or request["generation"] != generation
        ):
            raise ValueError("journal episode identity changed")
        if (
            request.get("protocol") != PROTOCOL
            or type(event.get("journal_version")) is not int
            or event["journal_version"] != JOURNAL_VERSION
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
        read_only = op in READ_ONLY_OPS
        if op == "devices.call":
            # devices/1 enforces that these actions cannot dispatch mutating tools.
            read_only = request["args"].get("action") in READ_ONLY_ACTIONS
        if (
            phase == "error"
            and isinstance(response.get("error", {}).get("details"), dict)
            and response["error"]["details"].get("effects") == "none"
        ):
            read_only = True  # explicit server-side rejection before dispatch, any operation
        if submitted and not read_only:
            raise ValueError("mutation or repeated submit after final verdict")
        if phase == "error":
            if response.get("ok") is not False:
                raise ValueError("journal error has a successful response")
            error = response.get("error", {})
            details = error.get("details")
            ended = (
                op == "devices.call"
                and error.get("code") == "episode_finished"
                and isinstance(details, dict)
                and details.get("effects") in ("none", "partial")
            )
            if not read_only and not ended:
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


def retain_journal(journal, out):
    """Copy the journal's original bytes into out, even when certification rejects them."""
    raw = Path(journal).read_bytes()
    (Path(out) / "actions.jsonl").write_bytes(raw)
    return raw


def _write_verdict(out, report, reward):
    (out / "submit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "reward.json").write_text(json.dumps({"reward": reward}) + "\n")


def _reset_outputs(out):
    """Overwrite stale outputs first: an interrupted certification leaves no old success."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "actions.jsonl").unlink(missing_ok=True)
    _write_verdict(out, {"valid": False, "trial_success": False, "error": "not certified"}, 0.0)


def refuse(out, error):
    """Record a fail-closed verdict for a run that could not even be identified."""
    out = Path(out)
    _reset_outputs(out)
    report = {"valid": False, "trial_success": False, "error": error}
    _write_verdict(out, report, 0.0)
    return report


def certify(*, socket_path, session, task_id, seed, budget_mode, journal, out, generation=0):
    """Certify one run end to end and write its verdict into out; fail closed.

    Checks that the service at socket_path still runs the expected task, seed and
    budget mode, submits, observes the frozen world, retains the journal as
    ``actions.jsonl``, then validates the score and the journal. ``submit.json`` holds
    the returned report and ``reward.json`` the reward: 1.0 only for a valid, completed,
    in-budget success. Any failure leaves ``valid: false`` and reward 0.0.
    """
    out = Path(out)
    _reset_outputs(out)
    report = {}
    reward, valid = 0.0, False
    raw = None
    try:
        if (
            type(seed) is not int
            or seed < 0
            or type(generation) is not int
            or generation < 0
            or budget_mode not in BUDGET_MODES
            or any(not isinstance(value, str) or not value for value in (task_id, session))
        ):
            raise ValueError("invalid task identity or budget mode")
        report.update(
            task=task_id, seed=seed, budget_mode=budget_mode, session=session, generation=generation
        )
        health, _ = rpc(socket_path, "health", session=session, generation=generation)
        report["health"] = health
        if type(health.get("seed")) is not int or (
            health.get("task"),
            health.get("seed"),
            health.get("budget_mode"),
        ) != (task_id, seed, budget_mode):
            raise ValueError("service task/seed/budget identity differs from the expected run")
        verdict, submit_id = rpc(socket_path, "submit", session=session, generation=generation)
        report["submit"] = verdict
        report["submit_request_id"] = submit_id
        observation, _ = rpc(socket_path, "observe", session=session, generation=generation)
        report["observation"] = observation
        raw = retain_journal(journal, out)
        success = validate_score(attach_budget(verdict, observation))
        report["journal"] = validate_journal(
            raw,
            session=session,
            submit_id=submit_id,
            verdict=verdict,
            practice=budget_mode == "practice",
            generation=generation,
        )
        valid, reward = True, float(success)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    if raw is None:
        try:
            raw = retain_journal(journal, out)
        except OSError as exc:
            report["journal_capture_error"] = f"{type(exc).__name__}: {exc}"
    if raw is not None:
        report["journal_artifact"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "validated": valid,
        }
    report.update(valid=valid, trial_success=bool(reward))
    _write_verdict(out, report, reward)
    return report
