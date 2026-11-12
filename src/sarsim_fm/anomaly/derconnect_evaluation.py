"""Coverage-scoped metrics: unknown time never supplies negatives."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import pandas as pd

from sarsim_fm.data.annotations import load_annotations


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _blocks(annotations):
    """Only explicit fully-reviewed blocks support event recall/precision."""
    groups = defaultdict(list)
    for a in annotations:
        if a.get("review_block") and a["label"] != "unknown":
            groups[a["review_block"]].append(a)
    blocks = []
    for name, rows in groups.items():
        rows.sort(key=lambda a: a["start"])
        if any(a["end"] != b["start"] for a, b in pairwise(rows)):
            raise ValueError(f"fully reviewed block {name} has unknown gaps")
        blocks.append((rows[0]["start"], rows[-1]["end"], name))
    return blocks


def _match(events, truths):
    """Maximum-cardinality one-to-one overlap matching, deterministic augmenting paths."""
    adjacency = [[j for j, a in enumerate(truths)
                  if e.start_utc < a["end"] and e.end_utc > a["start"]] for e in events]
    owners = {}

    def visit(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(events)):
        visit(i, set())
    return [(i, j) for j, i in sorted(owners.items())]


def evaluate_files(run, parts, annotations):
    run = Path(run)
    events = pd.read_parquet(run / "events.parquet")
    stats = {}
    sources = sorted({a["label_source"] for a in annotations}) or [None]
    for name in parts:
        frame = pd.read_parquet(run / name)
        for source in sources:
            meter, signal, detector = frame.iloc[0][["meter_id", "signal", "detector_id"]]
            key = (meter, signal, detector, source)
            arows = [a for a in annotations if a["meter_id"] == meter and a["signal"] == signal
                     and a["label_source"] == source and a["review_status"] == "approved"]
            s = stats.setdefault(key, {"grid": 0, "scored": 0, "flagged": 0, "normal": 0,
                                       "fp": 0, "positive": 0, "tp": 0, "fn": 0, "tn": 0,
                                       "start": frame.timestamp_utc.min(), "end": frame.bin_end_utc.max(),
                                       "annotations": arows, "eligible_ranges": [], "block_counts": {}})
            s["start"] = min(s["start"], frame.timestamp_utc.min())
            s["end"] = max(s["end"], frame.bin_end_utc.max())
            scored = frame.anomaly_flag.notna()
            flags = frame.anomaly_flag.fillna(0).eq(1)
            s["grid"] += len(frame)
            s["scored"] += int(scored.sum())
            s["flagged"] += int(flags.sum())
            # Compress scored coverage to intervals for event eligibility checks.
            opened = None
            for t, end, ok in zip(frame.timestamp_utc, frame.bin_end_utc, scored, strict=True):
                if ok:
                    if opened is None:
                        opened = [t, end]
                    else:
                        opened[1] = end
                elif opened:
                    s["eligible_ranges"].append(opened)
                    opened = None
            if opened:
                s["eligible_ranges"].append(opened)
            for a in arows:
                mask = scored & (frame.timestamp_utc >= a["start"]) & (frame.bin_end_utc <= a["end"])
                n, f = int(mask.sum()), int((mask & flags).sum())
                if a.get("review_block"):
                    block_counts = s["block_counts"].setdefault(a["review_block"],
                                                               {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                    if a["label"] == "anomaly":
                        block_counts["tp"] += f
                        block_counts["fn"] += n - f
                    elif a["label"] == "reviewed_normal":
                        block_counts["fp"] += f
                        block_counts["tn"] += n - f
                if a["label"] == "reviewed_normal":
                    s["normal"] += n
                    s["fp"] += f
                    s["tn"] += n - f
                elif a["label"] == "anomaly":
                    s["positive"] += n
                    s["tp"] += f
                    s["fn"] += n - f
    results = []
    for (meter, signal, detector, source), s in sorted(stats.items(), key=lambda kv: str(kv[0])):
        arows = s.pop("annotations")
        ranges = s.pop("eligible_ranges")
        merged = []
        for start, end in sorted(ranges):
            if merged and merged[-1][1] == start:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        blocks = _blocks(arows)
        selected_events = events[(events.meter_id == meter) & (events.signal == signal) &
                                 (events.detector_id == detector)]
        candidates, excluded = [], 0
        for e in selected_events.itertuples():
            if any(start <= e.start_utc and e.end_utc <= end for start, end, _ in blocks):
                candidates.append(e)
            else:
                excluded += 1
        truths = [a for a in arows if a["label"] == "anomaly" and a.get("review_block") and
                  s["start"] <= a["start"] and a["end"] <= s["end"] and
                  any(start <= a["start"] and a["end"] <= end for start, end in merged)]
        # No partly unscored review block can supply event denominators.
        eligible_blocks = [(start, end, name) for start, end, name in blocks
                           if any(lo <= start and end <= hi for lo, hi in merged)]
        candidates = [e for e in candidates if any(start <= e.start_utc and e.end_utc <= end
                                                  for start, end, _ in eligible_blocks)]
        truths = [a for a in truths if any(start <= a["start"] and a["end"] <= end
                                          for start, end, _ in eligible_blocks)]
        excluded = len(selected_events) - len(candidates)
        eligible_names = {name for _, _, name in eligible_blocks}
        reviewed_block_counts = {k: sum(v[k] for name, v in s["block_counts"].items()
                                        if name in eligible_names) for k in ("tp", "fp", "tn", "fn")}
        matched = _match(candidates, truths)
        precision = _ratio(len(matched), len(candidates)) if eligible_blocks else None
        recall = _ratio(len(matched), len(truths)) if eligible_blocks else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              0. if precision is not None and recall is not None else None)
        delays = [max(0., (candidates[i].detected_at_utc - truths[j]["start"]).total_seconds())
                  for i, j in matched]
        normal_intervals = []
        for a in arows:
            if a["label"] != "reviewed_normal":
                continue
            if normal_intervals and normal_intervals[-1][1] == a["start"]:
                normal_intervals[-1][1] = a["end"]
            else:
                normal_intervals.append([a["start"], a["end"]])
        false_normal_events = sum(any(start <= e.start_utc and e.end_utc <= end
                                     for start, end in normal_intervals)
                                  for e in selected_events.itertuples())
        event_count = len(selected_events)
        result = {"meter_id": meter, "signal": signal, "detector_id": detector,
                  "annotation_source": source, "start_utc": s["start"].isoformat(),
                  "end_utc": s["end"].isoformat(), "eligible_seconds": s["scored"] * 60,
                  "scored_fraction": _ratio(s["scored"], s["grid"]),
                  "flagged_points": s["flagged"], "events": event_count,
                  "events_per_device_day": _ratio(event_count, s["scored"] * 60 / 86400),
                  "reviewed_normal_seconds": s["normal"] * 60,
                  "reviewed_anomaly_seconds": s["positive"] * 60,
                  "point_fpr": _ratio(s["fp"], s["normal"]),
                  "reviewed_point_precision": _ratio(s["tp"], s["tp"] + s["fp"]) if s["normal"] else None,
                  "point_precision": _ratio(reviewed_block_counts["tp"], reviewed_block_counts["tp"] +
                                            reviewed_block_counts["fp"]) if eligible_blocks else None,
                  "point_recall": _ratio(reviewed_block_counts["tp"], reviewed_block_counts["tp"] +
                                         reviewed_block_counts["fn"]) if eligible_blocks else None,
                  "point_f1": None,
                  "false_events_per_reviewed_normal_day": _ratio(false_normal_events, s["normal"] / 1440),
                  "event_precision": precision, "event_recall": recall, "event_f1": f1,
                  "event_true_positives": len(matched) if eligible_blocks else None,
                  "evaluated_events": len(candidates), "evaluated_true_events": len(truths),
                  "fully_reviewed_seconds": sum((end - start).total_seconds()
                                                 for start, end, _ in eligible_blocks),
                  "boundary_or_unknown_events_excluded": excluded,
                  "mean_delay_seconds": sum(delays) / len(delays) if delays else None,
                  "unavailable_reason": ("no_approved_labels" if not arows else
                                         "no_fully_reviewed_eligible_blocks" if not eligible_blocks else
                                         "undefined_metrics_have_zero_denominator"),
                  "point_counts_reviewed_only": {k: s[k] for k in ("tp", "fp", "tn", "fn")},
                  "point_counts_fully_reviewed_blocks": reviewed_block_counts}
        p, r = result["point_precision"], result["point_recall"]
        if p is not None and r is not None:
            result["point_f1"] = 2 * p * r / (p + r) if p + r else 0.
        results.append(result)
    data_kind = json.loads((run / "identity.json").read_text())["config"]["data_kind"]
    return {"schema_version": "derconnect/v1", "evaluation_kind": data_kind,
            "matching": "maximum_one_to_one_half_open_overlap",
            "calibration_diagnostics": "calibration.json (in-sample; not held-out performance)",
            "results": results}


def evaluate_run(run, annotation_path):
    from sarsim_fm.anomaly.export import verify_bundle
    manifest = verify_bundle(run)
    result = evaluate_files(run, manifest["prediction_parts"], load_annotations(annotation_path))
    # Separate sidecar; never mutate the completed run's signed/checksummed metrics.
    print(json.dumps(result, indent=2, allow_nan=False))
    return result
