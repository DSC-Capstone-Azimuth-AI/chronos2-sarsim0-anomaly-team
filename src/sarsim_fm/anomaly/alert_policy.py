"""Conservative, model-free overlays for finalized DERCONNECT events."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.alert_policy_config import DAYS, rule_ready, validate_policy
from sarsim_fm.anomaly.derconnect_artifacts import (
    GROUP,
    BundleReader,
    ensure_new_output,
    evidence_id,
    portable_config,
    read_evidence,
)
from sarsim_fm.anomaly.export import write_json
from sarsim_fm.anomaly.policy_contract import VERSION, write_table
from sarsim_fm.anomaly.provenance import environment, file_hash, object_hash, save_source_snapshot
from sarsim_fm.data.annotations import utc


def occurrences(rule, start, end, timezone):
    """Timezone-aware half-open occurrences anchored to their local start weekday."""
    first = utc(start).tz_convert(timezone).date() - timedelta(days=1)
    last = utc(end).tz_convert(timezone).date()
    result, warnings = [], []
    for day in pd.date_range(first, last):
        if DAYS[day.weekday()] not in rule["weekdays"]:
            continue
        stop_day = day + pd.Timedelta(days=int(rule["local_end"] < rule["local_start"]))
        try:
            lo = pd.Timestamp(f"{day:%Y-%m-%d} {rule['local_start']}").tz_localize(
                timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
            hi = pd.Timestamp(f"{stop_day:%Y-%m-%d} {rule['local_end']}").tz_localize(
                timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
        except Exception as exc:
            if not isinstance(exc, ValueError) and type(exc).__name__ not in {"NonExistentTimeError", "AmbiguousTimeError"}:
                raise
            warnings.append("invalid_dst_occurrence")
            continue
        lo, hi = max(lo, utc(rule["effective_start"])), min(hi, utc(rule["effective_end"]))
        if lo < hi and start < hi and end > lo:
            result.append((lo, hi))
    return result, warnings


def load_policy_evidence(config, reader, base):
    qa = read_evidence(config["qa_report"], base)
    if qa is not None and (qa.get("schema_version") != "derconnect-forecast-qa/v1" or
                qa.get("kind") != "derconnect_forecast_qa" or
                qa.get("source_run_id") != reader.manifest["run_id"] or
                qa.get("source_manifest_sha256") != file_hash(reader.root / "run-manifest.json") or
                qa.get("model_sha256") != reader.manifest["model"]["sha256"]):
        raise ValueError("forecast QA does not match source/model identity")
    references = {}
    for rule in config["rules"]:
        ref = read_evidence(rule["reference"], base)
        if ref is not None:
            fields = {"rule_id": rule["rule_id"], "meter_id": rule["meter_id"], "signal": rule["signal"],
                      "unit": rule["reference_unit"], "expected_direction": rule["expected_direction"],
                      **{k: rule[k] for k in ("minimum_absolute_reference", "relative_change_lower",
                                             "relative_change_upper", "preceding_context_minutes",
                                             "max_duration_seconds")}}
            if (ref.get("kind") != "derconnect_schedule_reference" or
                    ref.get("review_status") != "approved" or not ref.get("source_hashes") or
                    any(ref.get(k) != v for k, v in fields.items()) or
                    any(ref.get(k) != rule["review"][k] for k in
                        ("reviewer", "reviewed_at", "schedule_evidence"))):
                raise ValueError(f"reference evidence does not match approved rule: {rule['rule_id']}")
        references[rule["rule_id"]] = ref
    return qa, references


def _valid_grid(frame, start, end):
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    return (not frame.empty and len(frame) == len(expected) and
            frame.timestamp_utc.reset_index(drop=True).equals(pd.Series(expected)) and
            frame.actual.notna().all() and np.isfinite(frame.actual.to_numpy(float)).all() and
            not frame.prediction_status.isin(["invalid_input", "insufficient_coverage"]).any())


def decide_event(event, reader, config, qa, references, mode):
    start, end = utc(event.start_utc), utc(event.end_utc)
    row = {"event_id": event.event_id, "rule_id": None, "candidate_rule_ids": "[]",
           "start_utc": start, "end_utc": end, "decision_data_cutoff_utc": reader.end,
           "reference_actual": None, "relative_change_min": None, "relative_change_max": None,
           "suppressed": False, "would_suppress": False, "time_overlap": False,
           "fully_contained": False, "direction_matches": None, "magnitude_matches": None,
           "qa_passed": False, "approved": False, "evidence_hashes": "{}"}
    reasons, candidates, contained = [], [], []
    for rule in config["rules"]:
        if not rule["enabled"] or (rule["meter_id"], rule["signal"]) != (event.meter_id, event.signal):
            continue
        windows, warnings = occurrences(rule, start, end, config["timezone"])
        reasons.extend(warnings)
        if windows:
            candidates.append(rule)
            if any(lo <= start and end <= hi for lo, hi in windows):
                contained.append(rule)
    row["time_overlap"] = bool(candidates)
    row["fully_contained"] = bool(contained)
    row["candidate_rule_ids"] = json.dumps(sorted(r["rule_id"] for r in candidates))
    hard_fail, insufficient = False, False
    if not candidates:
        reasons.append("outside_schedule")
        hard_fail = True
    elif len(candidates) != 1:
        reasons.append("ambiguous_matching_rules")
        hard_fail = True
    elif not contained:
        reasons.append("event_not_fully_contained")
        hard_fail = True
    else:
        rule = candidates[0]
        row["rule_id"] = rule["rule_id"]
        row["approved"] = rule_ready(rule)
        row["evidence_hashes"] = json.dumps({"reference": rule["reference"]["sha256"]
                                             if rule["reference"] else None,
                                             "qa": config["qa_report"]["sha256"] if config["qa_report"] else None})
        if not row["approved"]:
            reasons.append("pending_or_incomplete_rule")
            insufficient = True
        if (end - start).total_seconds() > rule["max_duration_seconds"]:
            reasons.append("event_too_long")
            hard_fail = True
        else:
            pre_start = start - pd.Timedelta(minutes=rule["preceding_context_minutes"])
            pre = reader.window(event.meter_id, event.signal, event.detector_id, pre_start, start)
            points = reader.window(event.meter_id, event.signal, event.detector_id, start, end)
            if not _valid_grid(pre, pre_start, start) or not _valid_grid(points, start, end):
                reasons.append("insufficient_valid_policy_context")
                insufficient = True
            else:
                if not (pre.unit.eq(rule["reference_unit"]).all() and points.unit.eq(rule["reference_unit"]).all()):
                    raise ValueError("policy reference units differ from source values")
                reference = float(pre.actual.median())
                row["reference_actual"] = reference
                differences = points.actual.to_numpy(float) - reference
                positive = (differences >= 0).all() and (differences > 0).any()
                negative = (differences <= 0).all() and (differences < 0).any()
                row["direction_matches"] = bool(positive if rule["expected_direction"] == "increase" else negative)
                if not row["direction_matches"]:
                    reasons.append("unexpected_or_mixed_direction")
                    hard_fail = True
                floor = rule["minimum_absolute_reference"]
                if floor is None:
                    reasons.append("missing_minimum_reference")
                    insufficient = True
                elif abs(reference) < floor:
                    reasons.append("reference_too_close_to_zero")
                    insufficient = True
                else:
                    delta = (points.actual.to_numpy(float) - reference) / abs(reference)
                    row["relative_change_min"], row["relative_change_max"] = float(delta.min()), float(delta.max())
                    lo, hi = rule["relative_change_lower"], rule["relative_change_upper"]
                    if lo is None or hi is None:
                        reasons.append("missing_change_envelope")
                        insufficient = True
                    else:
                        row["magnitude_matches"] = bool((delta >= lo).all() and (delta <= hi).all())
                        if not row["magnitude_matches"]:
                            reasons.append("change_outside_reviewed_envelope")
                            hard_fail = True
                finite_q = np.isfinite(points[["q01", "q10", "q50", "q90", "q99"]].to_numpy(float)).all()
                qa_state = qa.get("detectors", {}).get(event.detector_id, {}) if qa else {}
                row["qa_passed"] = bool(qa_state.get("status") == "pass" and finite_q and
                                         points.prediction_status.eq("scored").all())
        if not row["qa_passed"]:
            reasons.append("forecast_qa_unavailable_or_unresolved")
            insufficient = True
        if references.get(rule["rule_id"]) is None:
            reasons.append("missing_reference_evidence")
            insufficient = True
        settings = reader.manifest["resolved_config"]["detection"]
        guard = settings["merge_gap_seconds"] + settings["min_run_seconds"]
        follow_end = end + pd.Timedelta(seconds=guard)
        following = reader.window(event.meter_id, event.signal, event.detector_id, end,
                                  min(follow_end, reader.end))
        has_grid = (not following.empty and len(following) == guard // 60 and
                    following.timestamp_utc.reset_index(drop=True).equals(
                        pd.Series(pd.date_range(end, follow_end, freq="min", inclusive="left"))))
        invalid = following.prediction_status.isin(["invalid_input", "insufficient_coverage"]) if not following.empty else []
        explicit_break = False
        if len(invalid) and any(invalid):
            position = int(np.flatnonzero(invalid)[0])
            prefix = following.iloc[:position + 1]
            explicit_break = prefix.timestamp_utc.reset_index(drop=True).equals(pd.Series(
                pd.date_range(end, prefix.timestamp_utc.iloc[-1], freq="min")))
        if not has_grid and not explicit_break:
            reasons.append("right_censored_or_missing_finalization_context")
            insufficient = True
    row["would_suppress"] = False if hard_fail else None if insufficient else True
    row["suppressed"] = bool(mode == "active" and row["would_suppress"] is True)
    row["decision"] = "preview_only" if mode == "preview" else "suppress" if row["suppressed"] else "keep"
    row["eligibility_status"] = "not_matched" if hard_fail else "insufficient_evidence" if insufficient else "matched"
    row["reason_codes"] = json.dumps(sorted(set(reasons)) if reasons else ["approved_expected_transition"])
    return row


def _point_overlay(frame, identity, suppressed):
    membership = frame.event_member & ~frame.event_id.isin(suppressed)
    scored = frame.anomaly_flag.notna()
    flag = pd.Series(0, index=frame.index, dtype="Int64")
    flag.loc[scored & membership & frame.anomaly_flag.eq(1)] = 1
    flag.loc[~scored] = pd.NA
    return pd.DataFrame({**{k: v for k, v in identity.items()}, "detector_id": frame.detector_id,
                         "meter_id": frame.meter_id, "signal": frame.signal,
                         "timestamp_utc": frame.timestamp_utc, "event_id": frame.event_id,
                         "raw_anomaly_flag": frame.anomaly_flag.astype("Int64"), "actionable_flag": flag,
                         "actionable_event_member": membership})


def apply_policy(source, config, out, *, mode="preview", evidence_base=".", annotations=None):
    validate_policy(config, mode)
    reader = BundleReader(source)
    if Path(out).exists():
        raise FileExistsError(out)
    qa, references = load_policy_evidence(config, reader, evidence_base)
    env = environment()
    policy_id = evidence_id(config)
    application = object_hash({**reader.source_identity(), "policy_id": policy_id,
                               "mode": mode, "code_sha256": env["source_sha256"]})
    ident = {"schema_version": VERSION, "source_run_id": reader.manifest["run_id"],
             "policy_id": policy_id, "policy_application_id": application}
    events = pd.read_parquet(reader.root / "events.parquet")
    event_keys = ["run_id", "detector_id", "meter_id", "signal", "event_id"]
    if events.duplicated(event_keys).any() or events.duplicated(["run_id", "event_id"]).any():
        raise ValueError("duplicate source event keys")
    rows = []
    for event in events.itertuples():
        decision = decide_event(event, reader, config, qa, references, mode)
        rows.append({**ident, "detector_id": event.detector_id, "meter_id": event.meter_id,
                     "signal": event.signal, **decision})
    suppressed = {r["event_id"] for r in rows if r["suppressed"]}
    out = ensure_new_output(out)
    write_table(out / "event-decisions.parquet", rows, "event-decisions")
    hours, parts = {}, []
    for i, (_, _, frame) in enumerate(reader.frames()):
        overlay = _point_overlay(frame, ident, suppressed)
        name = f"point-decisions.parquet/{i:06d}.parquet"
        write_table(out / name, overlay, "point-decisions")
        parts.append(name)
        group = frame.assign(hour=frame.timestamp_utc.dt.floor("h"))
        for key, g in group.groupby([*GROUP, "hour"], sort=True):
            _, detector, meter, signal, hour = key
            h = hours.setdefault(key, {**ident, "detector_id": detector, "meter_id": meter, "signal": signal,
                                       "timestamp_utc": hour, "scored_minutes": 0, "raw_flagged_minutes": 0,
                                       "retained_flagged_minutes": 0, "actionable_flagged_minutes": 0})
            h["scored_minutes"] += int(g.anomaly_flag.notna().sum())
            h["raw_flagged_minutes"] += int(g.anomaly_flag.fillna(0).sum())
            h["retained_flagged_minutes"] += int((g.anomaly_flag.eq(1) & g.event_member).sum())
            h["actionable_flagged_minutes"] += int(overlay.loc[g.index, "actionable_flag"].fillna(0).sum())
    for h in hours.values():
        relevant = events[(events.detector_id == h["detector_id"]) & (events.meter_id == h["meter_id"]) &
                          (events.signal == h["signal"])]
        hour, end = h["timestamp_utc"], h["timestamp_utc"] + pd.Timedelta(hours=1)
        starts = relevant[(relevant.start_utc >= hour) & (relevant.start_utc < end)]
        overlap = relevant[(relevant.start_utc < end) & (relevant.end_utc > hour)]
        h.update(raw_event_starts=len(starts), suppressed_event_starts=int(starts.event_id.isin(suppressed).sum()),
                 actionable_event_starts=int((~starts.event_id.isin(suppressed)).sum()),
                 has_raw_event_overlap=bool(len(overlap)) if h["scored_minutes"] else None,
                 has_actionable_event_overlap=bool((~overlap.event_id.isin(suppressed)).any())
                 if h["scored_minutes"] else None)
    write_table(out / "hourly-policy-summary.parquet", list(hours.values()), "hourly-policy-summary")
    reasons = Counter(reason for r in rows for reason in json.loads(r["reason_codes"]))
    report = {**ident, "mode": mode, "raw_events": len(events), "effective_events": len(events) - len(suppressed),
              "suppressed_events": len(suppressed), "temporal_candidates": sum(r["time_overlap"] for r in rows),
              "fully_contained_candidates": sum(r["fully_contained"] for r in rows),
              "direction_matches": sum(r["direction_matches"] is True for r in rows),
              "magnitude_matches": sum(r["magnitude_matches"] is True for r in rows),
              "approval_complete": sum(r["approved"] for r in rows),
              "qa_passed": sum(r["qa_passed"] for r in rows),
              "would_suppress_known_true": sum(r["would_suppress"] is True for r in rows),
              "would_suppress_unknown": sum(r["would_suppress"] is None for r in rows),
              "reason_counts": dict(reasons), "accuracy_claim": None,
              "scope": "retrospective finalized-event review policy; no ground truth inferred"}
    write_json(out / "policy-report.json", report)
    if annotations is not None:
        from sarsim_fm.anomaly.alert_policy_evaluation import evaluate_policy
        result = evaluate_policy(reader, suppressed, annotations)
        write_json(out / "policy-evaluation.json", result)
    save_source_snapshot(out / "source-snapshot.zip")
    outputs = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {**ident, "complete": True, "source": reader.source_identity(), "mode": mode,
                "config": portable_config(config), "resolved_config": config, "code_sha256": env["source_sha256"],
                "executed_at_utc": datetime.now(UTC).isoformat(), "point_parts": parts, "outputs": outputs}
    write_json(out / "policy-manifest.json", manifest)
    verify_policy_bundle(out, source)
    return report


def verify_policy_bundle(policy, source):
    policy = Path(policy)
    manifest = json.loads((policy / "policy-manifest.json").read_text())
    if manifest.get("schema_version") != VERSION or not manifest.get("complete"):
        raise ValueError("policy bundle incomplete or unsupported")
    reader = BundleReader(source)
    if manifest["source"] != reader.source_identity():
        raise ValueError("policy source identity mismatch")
    for name, expected in manifest["outputs"].items():
        path = (policy / name).resolve()
        if not path.is_relative_to(policy.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"policy checksum mismatch: {name}")
    return manifest
