"""Read-only saved-output QA and bounded, explicitly requested model reproduction."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.derconnect_artifacts import BundleReader, ensure_new_output
from sarsim_fm.anomaly.export import write_json
from sarsim_fm.anomaly.provenance import environment, file_hash, object_hash
from sarsim_fm.anomaly.scores import interval_exceedance_score, rearrange_quantiles
from sarsim_fm.data.annotations import utc

LEVELS = [.01, .1, .5, .9, .99]
TARGETS = [f"2026-03-{day}T{time}:00Z" for day, time in
           [("02", "06:59"), ("02", "07:01"), ("02", "07:02"), ("02", "07:03"),
            ("04", "13:59"), ("04", "14:00"), ("04", "14:03"), ("06", "12:11"),
            ("02", "06:30"), ("04", "13:30")]]


def inspect_bundle(source, out, targets=None):
    reader = BundleReader(source)
    out = ensure_new_output(out)
    targets = [utc(t) for t in (TARGETS if targets is None else targets)]
    rows, stats, warnings = [], {}, []
    probabilistic = reader.manifest["quantile_kind"] != "degenerate_point"
    for _, _, f in reader.frames():
        for key, g in f.groupby(["detector_id", "meter_id", "signal"], sort=True):
            detector, meter, signal = key
            s = stats.setdefault(detector, {"meter_id": meter, "signal": signal, "points": 0,
                                           "scored": 0, "nonfinite_scored": 0, "crossings": 0,
                                           "negative_median_positive_actual": 0, "negative_q01": 0,
                                           "maximum_width": 0., "maximum_absolute_error": 0., "scopes": []})
            scope = {"meter_id": meter, "signal": signal}
            if scope not in s["scopes"]:
                s["scopes"].append(scope)
            scored = g.prediction_status.eq("scored") & g.anomaly_flag.notna()
            values = g.loc[scored, ["q01", "q10", "q50", "q90", "q99"]].to_numpy(float)
            finite = np.isfinite(values).all(axis=1) & np.isfinite(g.loc[scored, "actual"].to_numpy(float))
            s["points"] += len(g)
            s["scored"] += int(scored.sum())
            s["nonfinite_scored"] += int((~finite).sum())
            s["crossings"] += int((np.diff(values, axis=1) < 0).any(axis=1).sum())
            s["negative_median_positive_actual"] += int((scored & (g.actual > 0) & (g.q50 < 0)).sum())
            s["negative_q01"] += int((scored & (g.q01 < 0)).sum())
            finite_index = g.loc[scored].index[finite]
            if len(finite_index):
                width = g.loc[finite_index, "q99"] - g.loc[finite_index, "q01"]
                s["maximum_width"] = max(s["maximum_width"], float(width.max()))
                s["maximum_absolute_error"] = max(s["maximum_absolute_error"], float(
                    (g.loc[finite_index, "actual"] - g.loc[finite_index, "q50"]).abs().max()))
            for row in g.loc[g.timestamp_utc.isin(targets)].itertuples():
                rows.append({"detector_id": detector, "meter_id": meter, "signal": signal,
                             "timestamp_utc": row.timestamp_utc.isoformat(), "actual": number_or_none(row.actual),
                             "q01": number_or_none(row.q01), "q50": number_or_none(row.q50), "q99": number_or_none(row.q99),
                             "width98": number_or_none(row.q99 - row.q01), "score": number_or_none(row.anomaly_score),
                             "median_error_masked_by_interval": bool(row.anomaly_score == 0 and
                                                                     row.q50 != row.actual),
                             "raw_unrearranged_quantiles": None})
    for detector, s in stats.items():
        reasons = []
        if s["nonfinite_scored"]:
            reasons.append("nonfinite_scored_forecast")
        if s["crossings"]:
            reasons.append("exported_quantile_crossing")
        if probabilistic and s["negative_median_positive_actual"]:
            reasons.append("negative_median_against_positive_observations_requires_review")
        if not s["scored"]:
            reasons.append("no_scored_forecast_support")
        if not probabilistic:
            for scope in s["scopes"]:
                calibration = reader.manifest.get("calibration", {}).get(scope["meter_id"], {})
                scale = calibration.get("residual_scale")
                if (not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0 or
                        calibration.get("scale_points", 0) < 1 or calibration.get("calibration_points", 0) < 49):
                    reasons.append("missing_valid_residual_calibration")
        # A width/relative-error limit is deliberately not guessed from March data.
        # Probabilistic forecasts need reproduction/review even when coarse checks pass.
        status = "review_required" if reasons else "unverified" if probabilistic else "pass"
        s.update(status=status, reasons=reasons)
        warnings.extend({"detector_id": detector, "reason": reason} for reason in reasons)
    code = environment()["source_sha256"]
    identity = {**reader.source_identity(), "model_sha256": reader.manifest["model"]["sha256"],
                "code_sha256": code, "targets": [t.isoformat() for t in targets]}
    report = {"schema_version": "derconnect-forecast-qa/v1", "kind": "derconnect_forecast_qa",
              **identity, "diagnostic_id": object_hash(identity), "detectors": stats,
              "warnings": warnings, "reproduction_status": "not_run",
              "limits": ["Saved outputs do not contain unrearranged quantiles.",
                         "Negative power alone is not a fault or a universal QA failure.",
                         "Probabilistic QA requires reproduction and review; coverage alone is insufficient."],
              "cases": rows}
    write_json(out / "qa.json", report)
    pd.DataFrame(rows).to_csv(out / "cases.csv", index=False)
    plot_cases(out / "cases.svg", rows)
    write_json(out / "diagnostic-manifest.json", {"complete": True, "diagnostic_id": report["diagnostic_id"],
                                                 "source": reader.source_identity(),
                                                 "outputs": {n: file_hash(out / n) for n in ("qa.json", "cases.csv", "cases.svg")}})
    return report


def extract_context(staged, meter, target, length):
    """Only preceding valid UTC bins; checksums cover exactly the partitions read."""
    staged = Path(staged)
    manifest = json.loads((staged / "stage-manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("staging incomplete")
    target = utc(target)
    if type(length) is not int or not 1 <= length <= 8192:
        raise ValueError("context length must be 1..8192")
    start = target - pd.Timedelta(minutes=length)
    frames, hashes = [], {}
    for day in pd.date_range(start.floor("D"), target.floor("D"), freq="D"):
        name = f"prepared/{meter}/{day:%Y-%m-%d}.parquet"
        if day >= target:
            continue
        if name not in manifest["prepared_hashes"]:
            continue
        if file_hash(staged / name) != manifest["prepared_hashes"][name]:
            raise ValueError(f"context partition hash mismatch: {name}")
        f = pd.read_parquet(staged / name)
        frames.append(f[(f.timestamp_utc >= start) & (f.timestamp_utc < target)])
        hashes[name] = manifest["prepared_hashes"][name]
    if not frames:
        raise ValueError("insufficient prior context")
    f = pd.concat(frames).sort_values("timestamp_utc").reset_index(drop=True)
    grid = pd.date_range(start, target, freq="min", inclusive="left")
    if len(f) != length or not pd.DatetimeIndex(f.timestamp_utc).as_unit("us").equals(pd.DatetimeIndex(grid).as_unit("us")) or not f.usable.all():
        raise ValueError("insufficient uninterrupted prior context")
    x = f.actual.to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite context")
    return x, {"start_utc": start.isoformat(), "end_utc": target.isoformat(),
               "length": length, "minimum": float(x.min()), "maximum": float(x.max()),
               "context_sha256": object_hash(x.tolist()), "partition_hashes": hashes}


def reproduction_scope(source, targets=None):
    reader = BundleReader(source)
    targets = [utc(t) for t in (TARGETS if targets is None else targets)]
    if not 1 <= len(targets) <= 10 or len(set(targets)) != len(targets):
        raise ValueError("bounded reproduction requires 1..10 distinct targets")
    if len(reader.manifest["resolved_config"]["meters"]) != 1:
        raise ValueError("diagnostic scope is one meter")
    model = reader.manifest["model"]
    if not model.get("resolved_revision") or not model.get("files"):
        raise ValueError("reproduction requires pinned foundation model weights")
    return {"source": reader.source_identity(), "model": model, "targets": [t.isoformat() for t in targets],
            "contexts": [256, 1440], "batch_sizes": [1, 16], "maximum_target_predictions": len(targets) * 4,
            "weights_loaded": False}


def predict_cases(forecaster, contexts, actuals, batch_size):
    raw = np.asarray(forecaster.predict_quantiles(contexts, 1, LEVELS, batch_size,
                                                 np.ones(len(contexts), dtype=int)), dtype=float)
    if raw.shape != (len(contexts), 1, 5):
        raise ValueError("diagnostic adapter shape must be (cases, 1, 5)")
    finite = np.isfinite(raw).all(axis=(1, 2)) & np.isfinite(np.asarray(actuals))
    exported = np.full((len(contexts), 5), np.nan)
    scores = np.full(len(contexts), np.nan)
    if finite.any():
        exported[finite] = rearrange_quantiles(raw[finite])[:, 0, :]
        scores[finite] = interval_exceedance_score(np.asarray(actuals)[finite], exported[finite], LEVELS)
    return raw[:, 0, :], exported, scores


def reproduce(source, staged, out, *, device="cuda", targets=None, forecaster=None):
    """Explicit opt-in only. Default tests inject a fake adapter and never load weights."""
    reader = BundleReader(source)
    scope = reproduction_scope(source, targets)
    model = scope["model"]
    if reader.manifest["model"]["spec"].split(":", 1)[0] != "chronos2":
        raise ValueError("bounded reproduction currently supports Chronos-2")
    if out is not None and Path(out).exists():
        raise FileExistsError(out)
    if forecaster is None:
        from huggingface_hub import snapshot_download

        from sarsim_fm.models import load_forecaster
        from sarsim_fm.runtime import resolve_runtime
        snapshot = Path(snapshot_download(model["source_repository"], revision=model["resolved_revision"],
                                          allow_patterns=["*.json", "*.safetensors", "*.bin"]))
        for name, digest in model["files"].items():
            if file_hash(snapshot / name) != digest:
                raise ValueError(f"pinned model hash mismatch: {name}")
        forecaster = load_forecaster(f"chronos2:{snapshot}", resolve_runtime(device, "fp32"), 1440)
    out = ensure_new_output(out)
    records, missing = [], []
    config = reader.manifest["resolved_config"]
    if len(config["meters"]) != 1:
        raise ValueError("diagnostic scope is one meter")
    meter = config["meters"][0]
    detector = reader.manifest["detector_id"]
    signal = "total_active_power"
    for length in scope["contexts"]:
        contexts, selected, actuals, metadata = [], [], [], []
        for target in scope["targets"]:
            t = utc(target)
            row = reader.window(meter, signal, detector, t, t + pd.Timedelta(minutes=1))
            if row.empty:
                missing.append({"target": target, "context": length, "reason": "target_outside_source"})
                continue
            try:
                x, meta = extract_context(staged, meter, t, length)
            except ValueError as exc:
                missing.append({"target": target, "context": length, "reason": str(exc)})
                continue
            contexts.append(x)
            selected.append(target)
            metadata.append(meta)
            actuals.append(float(row.actual.iloc[0]))
        if not contexts:
            continue
        forecaster.context_length = length
        first = None
        for batch in scope["batch_sizes"]:
            raw, exported, scores = predict_cases(forecaster, contexts, actuals, batch)
            if first is None:
                first = raw.copy()
            for i, target in enumerate(selected):
                records.append({"target": target, "context": metadata[i], "batch_size": batch,
                                "actual": actuals[i], "raw_quantiles": [number_or_none(v) for v in raw[i]],
                                "exported_quantiles": [number_or_none(v) for v in exported[i]],
                                "score": number_or_none(scores[i]),
                                "raw_finite": bool(np.isfinite(raw[i]).all()),
                                "production_would_reject": bool(not np.isfinite(raw[i]).all()),
                                "raw_crossing": bool((np.diff(raw[i]) < 0).any()) if np.isfinite(raw[i]).all() else None,
                                "batch_difference_max_abs": number_or_none(np.max(np.abs(raw[i] - first[i]))),
                                "batch_difference_max_relative": number_or_none(np.max(np.abs(raw[i] - first[i]) /
                                                                            np.maximum(np.abs(first[i]), 1e-8)))})
    model_obj = getattr(getattr(forecaster, "pipeline", None), "model", None)
    parameter = next(model_obj.parameters(), None) if model_obj is not None else None
    report = {"schema_version": "derconnect-forecast-reproduction/v1", "scope": scope,
              "model_parameter_dtype": str(parameter.dtype) if parameter is not None else "test_double",
              "model_parameter_device": str(parameter.device) if parameter is not None else "test_double",
              "environment": environment(), "records": records, "unavailable": missing,
              "root_cause": "requires interpretation of raw/exported and batch/context comparisons",
              "target_predictions": len(records),
              "successful_target_predictions": sum(r["raw_finite"] for r in records),
              "failed_target_predictions": sum(not r["raw_finite"] for r in records)}
    write_json(out / "reproduction.json", report)
    flat = [{"target": r["target"], "context_length": r["context"]["length"], "batch_size": r["batch_size"],
             "q01": r["exported_quantiles"][0], "q50": r["exported_quantiles"][2],
             "q99": r["exported_quantiles"][4], "score": r["score"],
             "batch_difference_max_abs": r["batch_difference_max_abs"]} for r in records]
    pd.DataFrame(flat).to_csv(out / "cases.csv", index=False)
    plot_cases(out / "cases.svg", [{"timestamp_utc": r["target"], "actual": r["actual"],
               "q50": r["exported_quantiles"][2]} for r in records])
    write_json(out / "diagnostic-manifest.json", {"complete": True, "source": scope["source"],
                                                 "outputs": {n: file_hash(out / n) for n in
                                                             ("reproduction.json", "cases.csv", "cases.svg")}})
    return report


def number_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def plot_cases(path, rows):
    """Dependency-free scientific SVG; exact quantile values remain in the case records."""
    rows = [r for r in rows if r.get("actual") is not None and r.get("q50") is not None]
    if not rows:
        Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>No eligible target cases</title></svg>')
        return
    values = [r[k] for r in rows for k in ("actual", "q50")]
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1.)
    x = lambda i: 90 + i * 730 / max(len(rows) - 1, 1)
    y = lambda v: 280 - (v - lo) / span * 190
    markup = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="440" viewBox="0 0 900 440">',
              '<rect width="900" height="440" fill="white"/>',
              '<text x="90" y="30" font-size="18">Selected forecast cases: actual versus median</text>',
              '<text x="90" y="52" font-size="12">UTC target times; source power units unconfirmed; exact values in cases.csv</text>',
              '<path d="M90 80 V290 H840" fill="none" stroke="#333"/>',
              '<text x="10" y="180" font-size="12" transform="rotate(-90 10 180)">Power (source units)</text>']
    for value in (lo, (lo+hi)/2, hi):
        markup.append(f'<text x="18" y="{y(value):.2f}" font-size="11">{value:,.0f}</text>')
    for key, color, offset in (("actual", "#303846", 0), ("q50", "#2467a4", 170)):
        points = " ".join(f"{x(i):.2f},{y(r[key]):.2f}" for i,r in enumerate(rows))
        markup.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        markup.append(f'<text x="{90+offset}" y="420" font-size="13" fill="{color}">{key}</text>')
    stride = max(1, len(rows)//10)
    for i,r in enumerate(rows):
        if i % stride == 0:
            label=escape(str(r["timestamp_utc"])[5:16])
            markup.append(f'<text transform="translate({x(i):.2f} 315) rotate(40)" font-size="10">{label}</text>')
    markup.append('<text x="370" y="390" font-size="12">UTC target time (cases in recorded order)</text></svg>')
    Path(path).write_text("".join(markup))
