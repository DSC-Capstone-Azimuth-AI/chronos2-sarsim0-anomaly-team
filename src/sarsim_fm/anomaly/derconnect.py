"""Bounded timestamp-aware DERCONNECT runner, separate from labelled benchmarks."""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.calibration import fit_thresholds
from sarsim_fm.anomaly.export import VERSION, hourly_summary, write_json, write_table
from sarsim_fm.anomaly.forecast import rolling_forecast
from sarsim_fm.anomaly.provenance import (
    derconnect_model_identity,
    environment,
    file_hash,
    object_hash,
    save_source_snapshot,
)
from sarsim_fm.anomaly.scores import (
    fit_residual_scale,
    interval_exceedance_score,
    robust_residual_score,
)
from sarsim_fm.config import validate_config
from sarsim_fm.data.annotations import covers_normal, load_annotations, window

LEVELS = [.01, .1, .5, .9, .99]
SIGNAL = "total_active_power"


def stage_manifest(cfg):
    root = Path(cfg["staged"])
    manifest = json.loads((root / "stage-manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("staging incomplete")
    if {k: v for k, v in manifest["preparation"].items() if k != "chunk_size"} != {
            k: v for k, v in cfg["preparation"].items() if k != "chunk_size"}:
        raise ValueError("preparation config differs from staged data")
    for name, digest in manifest["prepared_hashes"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"staged hash mismatch: {name}")
    return manifest


def dry_run(cfg, spec):
    validate_config(cfg)
    start, end = window(cfg["roles"]["evaluation"])
    context = cfg["detection"]["context_length"]
    _baseline_kind(spec, context)
    return {"meters": cfg["meters"], "signal": SIGNAL, "model": spec,
            "output_start": start.isoformat(), "output_end": end.isoformat(),
            "output_points": int((end - start).total_seconds() / 60) * len(cfg["meters"]),
            "context_minutes": context, "sampling_seconds": 60,
            "working_partition_points": 1440, "calibration_point_cap": cfg["calibration"]["max_points"],
            "weights_loaded": False, "estimated_forecasts_upper_bound":
            int((end - window(cfg["roles"]["scale"])[0]).total_seconds() / 60) * len(cfg["meters"])}


def _baseline_kind(spec, context):
    kind, _, period = spec.partition(":")
    if kind == "seasonal_naive":
        if not period or not period.isdigit() or int(period) < 1:
            raise ValueError("DERCONNECT seasonal_naive requires an explicit positive period")
        if context < int(period):
            raise ValueError("context_length must supply the true seasonal lag")
    if kind not in {"chronos2", "patchtst_fm", "seasonal_naive", "rolling_median"}:
        raise ValueError(f"unsupported DERCONNECT model: {kind}")
    return kind in {"seasonal_naive", "rolling_median"}


def _days(cfg, meter, stage):
    """Yield a complete grid with history; never load more than one prepared day."""
    first = window(cfg["roles"]["scale"])[0] - pd.Timedelta(minutes=cfg["detection"]["context_length"])
    end = window(cfg["roles"]["evaluation"])[1]
    available = set(stage["prepared_parts"])
    day = first.floor("D")
    while day < end:
        start, stop = max(first, day), min(end, day + pd.Timedelta(days=1))
        grid = pd.date_range(start, stop, freq="min", inclusive="left")
        relative = f"prepared/{meter}/{day:%Y-%m-%d}.parquet"
        if relative in available:
            frame = pd.read_parquet(Path(cfg["staged"]) / relative).set_index("timestamp_utc").reindex(grid)
        else:
            frame = pd.DataFrame(index=grid)
        defaults = {"actual": np.nan, "usable": False, "observation_count": 0,
                    "coverage_fraction": 0., "quality_flags": '["missing_data"]'}
        for key, value in defaults.items():
            if key not in frame:
                frame[key] = value
            else:
                frame[key] = frame[key].fillna(value)
        yield day, frame
        day += pd.Timedelta(days=1)


def _forecast_day(frame, forecaster, history, context, batch, *, predict=True):
    q = np.full((len(frame), 5), np.nan)
    valid = frame.usable.to_numpy(bool)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    if not valid[0]:
        history.clear()
    previous_end = 0
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if start > previous_end:
            history.clear()
        values = np.r_[list(history), frame.actual.iloc[start:end].to_numpy(float)]
        offset = len(history)
        if predict and len(values) > context:
            forecasts = rolling_forecast(forecaster, "segment", values, context, context, LEVELS, batch)
            q[start + context - offset:end] = forecasts.quantiles
        history.extend(frame.actual.iloc[start:end].to_numpy(float))
        previous_end = end
    if not valid[-1]:
        history.clear()
    return q


def _mask(index, bounds):
    start, end = bounds
    return (index >= start) & (index < end)


def _records(frame):
    for row in frame.to_dict("records"):
        yield {k: (v.isoformat() if isinstance(v, pd.Timestamp) else None
                   if v is pd.NA or v is pd.NaT or isinstance(v, float) and not np.isfinite(v) else v)
               for k, v in row.items()}


def _events(conn, identity, min_run, merge_gap):
    """Stream qualifying runs, then merge without crossing unavailable rows."""
    events = []
    run = None
    pending = None

    def finish():
        nonlocal pending
        if pending is None:
            return
        start, end, detected = pending
        rows = conn.execute("SELECT t,score,threshold FROM points WHERE meter=? AND t>=? AND t<? "
                            "ORDER BY score DESC,t LIMIT 1", (identity["meter_id"], start, end)).fetchone()
        peak_t, peak_score, threshold = rows
        event_id = object_hash([identity["run_id"], identity["meter_id"], start, end])[:24]
        conn.execute("UPDATE points SET event_id=? WHERE meter=? AND t>=? AND t<?",
                     (event_id, identity["meter_id"], start, end))
        flagged = conn.execute("SELECT count(*) FROM points WHERE meter=? AND t>=? AND t<? AND flag=1",
                               (identity["meter_id"], start, end)).fetchone()[0]
        events.append({**identity, "event_id": event_id, "start_utc": pd.Timestamp(start),
                       "end_utc": pd.Timestamp(end), "detected_at_utc": pd.Timestamp(detected),
                       "peak_timestamp_utc": pd.Timestamp(peak_t), "peak_score": peak_score,
                       "threshold_at_peak": threshold, "duration_seconds":
                       (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds(),
                       "flagged_points": flagged, "anomaly_type": "unknown",
                       "type_source": "not_classified"})
        pending = None

    def qualify():
        nonlocal run, pending
        if run and run[2] >= min_run:
            start, end, _ = run
            detected = (pd.Timestamp(start) + pd.Timedelta(minutes=min_run)).isoformat()
            if pending and (pd.Timestamp(start) - pd.Timestamp(pending[1])).total_seconds() <= merge_gap * 60:
                pending = (pending[0], end, pending[2])
            else:
                finish()
                pending = (start, end, detected)
        run = None

    for t, flag in conn.execute("SELECT t,flag FROM points WHERE meter=? ORDER BY t", (identity["meter_id"],)):
        if flag == 1:
            end = (pd.Timestamp(t) + pd.Timedelta(minutes=1)).isoformat()
            run = (run[0], end, run[2] + 1) if run else (t, end, 1)
        else:
            qualify()
            if flag is None:
                finish()
    qualify()
    finish()
    return events


def detect_derconnect(cfg, spec, out, *, forecaster=None, device="cpu", resume=False):
    """Resume deterministically recomputes incomplete work after identity verification.

    Completed runs are reused only after all checksums match. No partial forecasts are
    mixed; scratch partitions are disposable and no model cache is reused implicitly.
    """
    validate_config(cfg)
    out = Path(out)
    if out.exists() and not resume:
        raise FileExistsError(out)
    stage = stage_manifest(cfg)
    baseline = _baseline_kind(spec, cfg["detection"]["context_length"])
    annotations = load_annotations(cfg.get("annotations"))
    roles = {k: window(v) for k, v in cfg["roles"].items()}
    exploratory = cfg["calibration"]["exploratory"]
    for meter in cfg["meters"]:
        if meter not in stage["ranges"]:
            raise ValueError(f"requested meter absent from staging: {meter}")
        for role in ("scale", "calibration"):
            if not exploratory and not covers_normal(annotations, meter, SIGNAL, *roles[role]):
                raise ValueError(f"approved reviewed-normal {role} window required for {meter}")
    if forecaster is None:
        from sarsim_fm.models import load_derconnect_forecaster
        from sarsim_fm.runtime import resolve_runtime

        forecaster = load_derconnect_forecaster(
            spec, resolve_runtime(device, "fp32"), cfg["detection"]["context_length"],
        )
    model = derconnect_model_identity(spec, forecaster)
    context = cfg["detection"]["context_length"]
    effective_context = context
    if spec.startswith("patchtst_fm"):
        from sarsim_fm.models.patchtst_fm import forecast_slots

        effective_context = min(context, forecaster.model.config.context_length -
                                forecast_slots(forecaster.model.config, 1))
    env = environment()
    cfg_identity = {k: v for k, v in cfg.items() if k not in {"staged", "annotations"}}
    cfg_identity["preparation"] = {k: v for k, v in cfg["preparation"].items() if k != "chunk_size"}
    identity = {"schema_version": VERSION, "config": cfg_identity, "model": model,
                "preparation_id": stage["preparation_id"], "source_sha256": env["source_sha256"],
                "dependencies": env["packages"], "annotations_sha256":
                file_hash(cfg["annotations"]) if cfg.get("annotations") else None,
                "effective_context": effective_context, "mode": "rolling_one_step"}
    for key in ("selection_record", "calibration_record"):
        if cfg.get(key):
            identity[key] = file_hash(cfg[key])
    run_id = object_hash(identity)
    # A locked selection must be explicit about the complete detector, preparation and model.
    if cfg.get("locked"):
        selection = json.loads(Path(cfg["selection_record"]).read_text())
        if selection.get("detector") != cfg["detection"] or selection.get("model") != model:
            raise ValueError("locked selection does not match detector/model")
        if selection.get("preparation_id") != stage["preparation_id"]:
            raise ValueError("locked selection preparation mismatch")
    out = Path(out)
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        saved = json.loads((out / "identity.json").read_text())
        if saved != identity:
            raise ValueError("resume identity mismatch")
        if (out / "run-manifest.json").exists():
            from sarsim_fm.anomaly.export import verify_bundle
            return verify_bundle(out)
    else:
        out.mkdir(parents=True)
        write_json(out / "identity.json", identity)
    save_source_snapshot(out / "source-snapshot.zip")
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    conn = sqlite3.connect(scratch / "points.sqlite")
    conn.execute("DROP TABLE IF EXISTS points")
    conn.execute("CREATE TABLE points (meter TEXT,t TEXT,payload TEXT,score REAL,threshold REAL,"
                 "flag INTEGER,event_id TEXT,PRIMARY KEY(meter,t))")
    detector_id = object_hash({"model": model, "detection": cfg["detection"],
                               "calibration": cfg["calibration"], "effective_context": effective_context})[:24]
    calibration, all_events, quality_events, prediction_parts, summaries = {}, [], [], [], []
    try:
        for meter in cfg["meters"]:
            history = deque(maxlen=context)
            scale_y, scale_q, cal_y, cal_q = [], [], [], []
            spool = []
            for day, frame in _days(cfg, meter, stage):
                needs_forecasts = any(_mask(frame.index, roles[key]).any()
                                      for key in ("scale", "calibration", "evaluation"))
                q = _forecast_day(frame, forecaster, history, context, cfg["detection"]["batch_size"],
                                  predict=needs_forecasts)
                has_q = np.isfinite(q).all(axis=1)
                for role, ys, qs in (("scale", scale_y, scale_q), ("calibration", cal_y, cal_q)):
                    mask = _mask(frame.index, roles[role]) & has_q
                    ys.extend(frame.actual[mask].tolist())
                    qs.extend(q[mask].tolist())
                    if len(ys) > cfg["calibration"]["max_points"]:
                        raise ValueError("calibration point cap exceeded; explicitly raise configured cap")
                if _mask(frame.index, roles["evaluation"]).any():
                    selected = frame.loc[_mask(frame.index, roles["evaluation"])].copy()
                    for i, key in enumerate(("q01", "q10", "q50", "q90", "q99")):
                        selected[key] = q[_mask(frame.index, roles["evaluation"]), i]
                    name = scratch / f"{meter}-{day:%Y-%m-%d}.parquet"
                    selected.to_parquet(name)
                    spool.append(name)
            if not scale_y or not cal_y:
                raise ValueError(f"insufficient usable calibration/scale support: {meter}")
            scale = fit_residual_scale(scale_y, np.array(scale_q)[:, 2])

            def score(actual, quantiles, scale=scale):
                if baseline:
                    return robust_residual_score(actual, quantiles[:, 2], scale)
                return interval_exceedance_score(actual, quantiles, LEVELS)

            cs = score(np.array(cal_y), np.array(cal_q))
            threshold = fit_thresholds(cs, np.ones(len(cs)), cfg["calibration"]["target_fpr"]).thresholds[1]
            cal_record = {"meter_id": meter, "signal": SIGNAL, "threshold": threshold,
                          "residual_scale": scale, "scale_points": len(scale_y),
                          "calibration_points": len(cal_y), "baseline_review_status":
                          "unverified" if exploratory else "approved", "exploratory": exploratory,
                          "windows": {k: cfg["roles"][k] for k in ("scale", "calibration")},
                          "in_sample_fpr": float((cs > threshold).mean()),
                          "target_fpr": cfg["calibration"]["target_fpr"]}
            cal_id = object_hash({**cal_record, "identity": identity})
            calibration[meter] = {**cal_record, "calibration_id": cal_id}
            if cfg.get("locked"):
                frozen = json.loads(Path(cfg["calibration_record"]).read_text())
                expected = frozen.get(meter, {})
                if any(expected.get(k) != cal_record[k] for k in cal_record):
                    raise ValueError("locked calibration record mismatch")
            ident = {"schema_version": VERSION, "run_id": run_id, "detector_id": detector_id,
                     "model_name": spec, "meter_id": meter, "signal": SIGNAL}
            quality_open = None
            for path in spool:
                frame = pd.read_parquet(path)
                for t, row in frame.iterrows():
                    usable = bool(row.usable)
                    has_q = np.isfinite(row.q50)
                    status = ("scored" if has_q else "warmup") if usable else (
                        "invalid_input" if "invalid_phase" in row.quality_flags or "conflict" in row.quality_flags
                        else "insufficient_coverage")
                    end = t + pd.Timedelta(minutes=1)
                    s = float(score(np.array([row.actual]), np.array([[row.q01, row.q10, row.q50,
                                                                     row.q90, row.q99]]))[0]) if has_q else None
                    point = {**ident, "timestamp_utc": t, "bin_end_utc": end, "available_at_utc": end,
                             "forecast_origin_utc": t if has_q else None, "horizon_steps": 1,
                             "sampling_seconds": 60, "actual": row.actual, "expected_power": row.q50,
                             **{key: row[key] for key in ("q01", "q10", "q50", "q90", "q99")},
                             "unit": stage["unit"], "score_name": "robust_residual" if baseline
                             else "interval_exceedance", "anomaly_score": s,
                             "threshold": threshold if has_q else None,
                             "anomaly_flag": int(s > threshold) if s is not None else None,
                             "event_id": None, "event_member": False, "anomaly_type": "unknown",
                             "type_source": "not_classified", "prediction_status": status,
                             "observation_count": int(row.observation_count),
                             "coverage_fraction": float(row.coverage_fraction),
                             "quality_flags": row.quality_flags, "calibration_id": cal_id}
                    record = next(_records(pd.DataFrame([point])))
                    conn.execute("INSERT INTO points(meter,t,payload,score,threshold,flag) VALUES(?,?,?,?,?,?)",
                                 (meter, t.isoformat(), json.dumps(record), s,
                                  threshold if has_q else None, point["anomaly_flag"]))
                    reason = row.quality_flags if json.loads(row.quality_flags) else None
                    if quality_open and (reason != quality_open[2]):
                        quality_events.append(_quality(ident, *quality_open))
                        quality_open = None
                    if reason:
                        quality_open = (quality_open[0] if quality_open else t, end, reason)
                conn.commit()
            if quality_open:
                quality_events.append(_quality(ident, *quality_open))
            events = _events(conn, ident, cfg["detection"]["min_run_seconds"] // 60,
                             cfg["detection"]["merge_gap_seconds"] // 60)
            all_events.extend(events)
            conn.commit()
            for path in spool:
                date = path.stem[len(meter) + 1:]
                start = pd.Timestamp(date, tz="UTC")
                stop = start + pd.Timedelta(days=1)
                points = []
                for payload, event_id in conn.execute(
                        "SELECT payload,event_id FROM points WHERE meter=? AND t>=? AND t<? ORDER BY t",
                        (meter, start.isoformat(), stop.isoformat())):
                    record = json.loads(payload)
                    record.update(event_id=event_id, event_member=event_id is not None)
                    points.append(record)
                relative = f"predictions.parquet/{meter}/{date}.parquet"
                write_table(out / relative, points, "predictions")
                prediction_parts.append(relative)
                completed = pd.read_parquet(out / relative)
                summaries.extend(hourly_summary(completed, pd.DataFrame(events)))
        write_table(out / "events.parquet", all_events, "events")
        write_table(out / "quality-events.parquet", quality_events, "quality-events")
        write_table(out / "hourly-summary.parquet", summaries, "hourly-summary")
        write_json(out / "calibration.json", calibration)
        from sarsim_fm.anomaly.derconnect_evaluation import evaluate_files
        metrics = evaluate_files(out, prediction_parts, annotations)
        write_json(out / "metrics.json", metrics)
        outputs = prediction_parts + ["events.parquet", "quality-events.parquet", "hourly-summary.parquet",
                                       "calibration.json", "metrics.json", "identity.json", "source-snapshot.zip"]
        manifest = {**identity, "run_id": run_id, "detector_id": detector_id, "complete": True,
                    "resolved_config": cfg, "environment": env, "calibration": calibration,
                    "quantile_kind": "degenerate_point" if baseline else "model_quantiles",
                    "context_minutes": context, "effective_context_minutes": effective_context,
                    "prediction_parts": prediction_parts,
                    "partition_state": {p: "complete" for p in prediction_parts},
                    "outputs": {name: file_hash(out / name) for name in outputs}}
        write_json(out / "run-manifest.json", manifest)
        return manifest
    finally:
        conn.close()


def _quality(identity, start, end, reason):
    return {**identity, "quality_event_id": object_hash([identity, str(start), str(end), reason])[:24],
            "start_utc": start, "end_utc": end, "reason": reason}
