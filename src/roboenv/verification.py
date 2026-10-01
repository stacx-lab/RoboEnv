"""Owner-side score and journal checks, independent of agent orchestration.

These checks validate single-episode evidence, not fresh-process replay.
"""

import hashlib
import json
import math


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
