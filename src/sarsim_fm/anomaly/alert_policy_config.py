"""Closed validation for reviewed schedule policies, separate from detector configs."""
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sarsim_fm.data.annotations import utc, window

VERSION = "derconnect-alert-policy/v1"
TOP = {"kind", "schema_version", "policy_name", "timezone", "rules", "qa_report"}
RULE = {"rule_id", "enabled", "meter_id", "signal", "weekdays", "local_start", "local_end",
        "effective_start", "effective_end", "expected_direction", "preceding_context_minutes",
        "max_duration_seconds", "minimum_absolute_reference", "relative_change_lower",
        "relative_change_upper", "reference", "reference_unit", "review"}
REVIEW = {"status", "reviewer", "reviewed_at", "schedule_evidence"}
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError(f"{name} requires exactly {sorted(allowed)}")


def _ref(value):
    if value is not None:
        _keys(value, {"path", "sha256"}, "evidence")
        if not isinstance(value["path"], str) or not value["path"] or (not isinstance(value["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])):
            raise ValueError("evidence requires path and SHA-256")


def _number(value, name, *, positive=False):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or
                              (positive and value <= 0)):
        raise ValueError(f"invalid {name}")


def rule_ready(rule):
    return (rule["review"]["status"] == "approved" and rule["reference"] is not None and
            all(rule[k] is not None for k in ("minimum_absolute_reference", "relative_change_lower",
                                              "relative_change_upper")))


def validate_policy(config, mode="preview"):
    _keys(config, TOP, "alert policy")
    if config["kind"] != "derconnect_alert_policy" or config["schema_version"] != VERSION:
        raise ValueError("unsupported alert policy kind/version")
    if mode not in {"preview", "active"}:
        raise ValueError("policy mode must be preview or active")
    if not isinstance(config["policy_name"], str) or not config["policy_name"]:
        raise ValueError("policy_name required")
    try:
        ZoneInfo(config["timezone"])
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError("invalid policy timezone") from exc
    _ref(config["qa_report"])
    if not isinstance(config["rules"], list) or not config["rules"]:
        raise ValueError("nonempty policy rules required")
    ids = set()
    for rule in config["rules"]:
        _keys(rule, RULE, "schedule rule")
        if not all(isinstance(rule[k], str) and rule[k] for k in ("rule_id", "meter_id", "signal", "reference_unit")):
            raise ValueError("rule identity and reference_unit required")
        if rule["rule_id"] in ids:
            raise ValueError("duplicate rule_id")
        ids.add(rule["rule_id"])
        if type(rule["enabled"]) is not bool:
            raise ValueError("enabled must be Boolean")
        if (not isinstance(rule["weekdays"], list) or not rule["weekdays"] or
                any(d not in DAYS for d in rule["weekdays"]) or len(set(rule["weekdays"])) != len(rule["weekdays"])):
            raise ValueError("invalid weekdays")
        for k in ("local_start", "local_end"):
            if not isinstance(rule[k], str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", rule[k]):
                raise ValueError("local boundaries must be HH:MM")
        if rule["local_start"] == rule["local_end"]:
            raise ValueError("schedule window cannot cover an entire day")
        window({"start": rule["effective_start"], "end": rule["effective_end"]})
        if rule["expected_direction"] not in {"increase", "decrease"}:
            raise ValueError("invalid expected_direction")
        if type(rule["preceding_context_minutes"]) is not int or not 5 <= rule["preceding_context_minutes"] <= 1440:
            raise ValueError("preceding context must be 5..1440 minutes")
        if type(rule["max_duration_seconds"]) is not int or not 60 <= rule["max_duration_seconds"] <= 86400 or rule["max_duration_seconds"] % 60:
            raise ValueError("max duration must be whole minutes, at most one day")
        _number(rule["minimum_absolute_reference"], "minimum reference", positive=True)
        for k in ("relative_change_lower", "relative_change_upper"):
            _number(rule[k], k)
        lo, hi = rule["relative_change_lower"], rule["relative_change_upper"]
        if lo is not None and hi is not None and (lo >= hi or
                (rule["expected_direction"] == "increase" and lo < 0) or
                (rule["expected_direction"] == "decrease" and hi > 0)):
            raise ValueError("change envelope conflicts with direction or ordering")
        _ref(rule["reference"])
        _keys(rule["review"], REVIEW, "review record")
        review = rule["review"]
        if review["status"] not in {"pending", "approved"}:
            raise ValueError("invalid review status")
        if review["status"] == "approved":
            if not all(isinstance(review[k], str) and review[k] for k in ("reviewer", "schedule_evidence")):
                raise ValueError("approved review requires reviewer and schedule evidence")
            utc(review["reviewed_at"])
            if not rule_ready(rule):
                raise ValueError("approved rules require reference evidence and complete bounds")
        if mode == "active" and rule["enabled"] and not rule_ready(rule):
            raise ValueError("active mode requires approved complete rules")
    if mode == "active" and config["qa_report"] is None:
        raise ValueError("active mode requires forecast QA evidence")
