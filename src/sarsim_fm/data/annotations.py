"""Explicit UTC review coverage; absence of annotations remains unknown."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError(f"timezone-aware timestamp required: {value}")
    return stamp.tz_convert("UTC").as_unit("us")


def window(value):
    start, end = utc(value["start"]), utc(value["end"])
    if end <= start:
        raise ValueError("window end must follow start")
    return start, end


def load_annotations(path=None):
    if path is None:
        return []
    path = Path(path)
    records = json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path).to_dict("records")
    result = []
    fields = {"meter_id", "signal", "start", "end", "label", "label_source", "review_status",
              "ground_truth_type", "review_block"}
    for record in records:
        if set(record) - fields:
            raise ValueError("unknown annotation fields")
        start, end = window(record)
        if record["label"] not in {"anomaly", "reviewed_normal", "unknown"}:
            raise ValueError("invalid annotation label")
        if not record["label_source"] or record["review_status"] not in {"approved", "pending"}:
            raise ValueError("annotation needs source and approved/pending review_status")
        result.append({**record, "start": start, "end": end})
    result.sort(key=lambda a: (a["meter_id"], a["signal"], a["start"], a["end"]))
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if (a["meter_id"], a["signal"]) != (b["meter_id"], b["signal"]):
                continue
            if b["start"] >= a["end"]:
                break
            if a["label"] != b["label"] or a["label_source"] != b["label_source"]:
                raise ValueError("conflicting annotation overlaps")
            raise ValueError("overlapping annotations must be consolidated")
    return result


def covers_normal(annotations, meter, signal, start, end):
    cursor = start
    for a in annotations:
        if a["meter_id"] != meter or a["signal"] != signal or a["end"] <= cursor:
            continue
        if (a["start"] > cursor or a["label"] != "reviewed_normal" or
                a["review_status"] != "approved"):
            continue
        cursor = max(cursor, a["end"])
        if cursor >= end:
            return True
    return False
