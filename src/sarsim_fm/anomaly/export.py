"""Versioned DERCONNECT file contract. No model imports are needed by consumers."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sarsim_fm.anomaly.provenance import file_hash

VERSION = "derconnect/v1"
IDENTITY = {k: "string" for k in
            ("schema_version", "run_id", "detector_id", "model_name", "meter_id", "signal")}
POINTS = {
    **IDENTITY, **dict.fromkeys(("timestamp_utc", "bin_end_utc", "available_at_utc",
                                "forecast_origin_utc"), "timestamp"),
    **dict.fromkeys(("horizon_steps", "sampling_seconds", "observation_count"), "int64"),
    **dict.fromkeys(("actual", "expected_power", "q01", "q10", "q50", "q90", "q99",
                    "anomaly_score", "threshold", "coverage_fraction"), "float64"),
    **dict.fromkeys(("unit", "score_name", "event_id", "anomaly_type", "type_source",
                    "prediction_status", "quality_flags", "calibration_id"), "string"),
    "anomaly_flag": "int64", "event_member": "bool",
}
EVENTS = {
    **IDENTITY, "event_id": "string",
    **dict.fromkeys(("start_utc", "end_utc", "detected_at_utc", "peak_timestamp_utc"), "timestamp"),
    **dict.fromkeys(("peak_score", "threshold_at_peak", "duration_seconds"), "float64"),
    "flagged_points": "int64", "anomaly_type": "string", "type_source": "string",
}
QUALITY = {**IDENTITY, "quality_event_id": "string", "start_utc": "timestamp",
           "end_utc": "timestamp", "reason": "string"}
HOURLY = {**IDENTITY, "timestamp_utc": "timestamp", "valid_minutes": "int64",
          "scored_minutes": "int64", "flagged_minutes": "int64", "event_count": "int64",
          "has_anomaly": "bool", "peak_raw_score": "float64", "event_ids": "string",
          "actual_power_mean": "float64", "forecast_point_mean": "float64",
          "eligible_bin_count": "int64", "coverage_fraction": "float64"}
FIELDS = {"predictions": POINTS, "events": EVENTS, "quality-events": QUALITY,
          "hourly-summary": HOURLY}
NULLABLE = {
    "predictions": {"actual", "expected_power", "q01", "q10", "q50", "q90", "q99",
                    "anomaly_score", "threshold", "anomaly_flag", "event_id",
                    "forecast_origin_utc", "calibration_id"},
    "events": set(), "quality-events": set(),
    "hourly-summary": {"has_anomaly", "peak_raw_score", "actual_power_mean",
                       "forecast_point_mean"},
}


def schema(name):
    types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(),
             "bool": pa.bool_(), "timestamp": pa.timestamp("us", tz="UTC")}
    return pa.schema([pa.field(k, types[v], nullable=k in NULLABLE[name])
                      for k, v in FIELDS[name].items()], metadata={b"schema_version": VERSION.encode()})


def write_table(path, rows, name):
    """Atomic Parquet replacement, including typed empty tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    for col, kind in FIELDS[name].items():
        if col not in frame:
            frame[col] = pd.Series(index=frame.index, dtype="object")
        if kind == "timestamp":
            frame[col] = pd.to_datetime(frame[col], utc=True)
    table = pa.Table.from_pandas(frame[list(FIELDS[name])], schema=schema(name), preserve_index=False)
    table.validate(full=True)
    for field in table.schema:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"null in required field {field.name}")
    tmp = path.with_suffix(".partial")
    pq.write_table(table, tmp)
    tmp.replace(path)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_bundle(run):
    run = Path(run)
    manifest = json.loads((run / "run-manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("run is incomplete")
    for name, digest in manifest["outputs"].items():
        if not (run / name).is_file() or file_hash(run / name) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return manifest


def hourly_summary(frame, events):
    """Display aggregates only; never manufacture hourly forecast quantiles."""
    result = []
    keys = list(IDENTITY)
    for _, group in frame.groupby(keys, sort=True):
        group = group.set_index("timestamp_utc")
        for hour, g in group.groupby(pd.Grouper(freq="h")):
            if g.empty:
                continue
            valid = g.actual.notna() & ~g.prediction_status.isin(
                ["invalid_input", "insufficient_coverage"])
            scored = g.anomaly_flag.notna()
            eligible = valid & g.q50.notna()
            e = events
            if not e.empty:
                e = e[(e.meter_id == g.meter_id.iloc[0]) &
                      (e.detector_id == g.detector_id.iloc[0]) &
                      (e.start_utc < hour + pd.Timedelta(hours=1)) & (e.end_utc > hour)]
            coverage = float(valid.sum() * g.sampling_seconds.iloc[0] / 3600)
            result.append({**{k: g[k].iloc[0] for k in keys}, "timestamp_utc": hour,
                           "valid_minutes": int(valid.sum()), "scored_minutes": int(scored.sum()),
                           "flagged_minutes": int(g.anomaly_flag.fillna(0).sum()),
                           "event_count": int((e.start_utc >= hour).sum()) if len(e) else 0,
                           "has_anomaly": bool(len(e)) if scored.any() else None,
                           "peak_raw_score": g.anomaly_score.max() if scored.any() else None,
                           "event_ids": json.dumps(sorted(e.event_id.tolist())) if len(e) else "[]",
                           "coverage_fraction": min(coverage, 1.0),
                           "actual_power_mean": float(g.loc[valid, "actual"].mean())
                           if coverage >= .8 else None,
                           "forecast_point_mean": float(g.loc[eligible, "q50"].mean())
                           if eligible.sum() * g.sampling_seconds.iloc[0] / 3600 >= .8 else None,
                           "eligible_bin_count": int(eligible.sum())})
    return result


def export_bundle(run, out, csv=True, hourly=True):
    """Export bounded prediction partitions from a verified, completed run."""
    run, out = Path(run), Path(out)
    manifest = verify_bundle(run)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    events = pd.read_parquet(run / "events.parquet")
    summaries = []
    for relative in manifest["prediction_parts"]:
        frame = pd.read_parquet(run / relative)
        target = out / relative
        write_table(target, frame, "predictions")
        if csv:
            write_csv(target.with_suffix(".csv"), frame, "predictions")
        if hourly:
            summaries.extend(hourly_summary(frame, events))
    for name in ("events", "quality-events"):
        frame = pd.read_parquet(run / f"{name}.parquet")
        write_table(out / f"{name}.parquet", frame, name)
        if csv:
            write_csv(out / f"{name}.csv", frame, name)
    if hourly:
        write_table(out / "hourly-summary.parquet", summaries, "hourly-summary")
    for name in manifest["outputs"]:
        if name not in manifest["prediction_parts"] and not (out / name).exists():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / name, target)
    exported = {**manifest, "source_run_manifest_sha256": file_hash(run / "run-manifest.json"),
                "outputs": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*"))
                            if p.is_file()}}
    write_json(out / "run-manifest.json", exported)
    write_json(out / "export-manifest.json", {
        "schema_version": VERSION, "run_id": manifest["run_id"], "complete": True,
        "outputs": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*"))
                    if p.is_file()}})


def write_csv(path, frame, name):
    frame = frame.copy()
    for key, kind in FIELDS[name].items():
        if kind == "timestamp":
            frame[key] = pd.to_datetime(frame[key], utc=True).map(
                lambda v: v.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if pd.notna(v) else None)
        if kind == "int64":
            frame[key] = frame[key].astype("Int64")
    frame.to_csv(path, index=False, na_rep="")
