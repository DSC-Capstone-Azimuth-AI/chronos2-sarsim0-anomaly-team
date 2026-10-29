"""Reproducible anomaly runs. Synthetic labels never enter calibration masks."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.baselines import MedianMAD
from sarsim_fm.anomaly.calibration import (
    apply_thresholds,
    fit_thresholds,
    interval_coverage,
    normal_metrics,
)
from sarsim_fm.anomaly.events import build_events
from sarsim_fm.anomaly.forecast import cached_forecast
from sarsim_fm.anomaly.metrics import (
    event_metrics,
    paired_family_bootstrap,
    point_auprc,
    top_one_time,
    ucr_hit,
)
from sarsim_fm.anomaly.provenance import (
    ROOT,
    checkpoint_identity,
    environment,
    file_hash,
    object_hash,
    save_source_snapshot,
)
from sarsim_fm.anomaly.scores import (
    finite_array,
    fit_residual_scale,
    interval_exceedance_score,
    quantile_tail_score,
    robust_residual_score,
)
from sarsim_fm.evaluate import LEVELS

SCORES = ("interval_exceedance", "robust_residual", "quantile_tail")
DEFAULTS = {
    "context_length": 256, "mode": "rolling", "prediction_length": 18, "batch_size": 32,
    "score": "robust_residual", "target_fpr": 0.02,
    "events": {"min_run": 1, "merge_gap": 0, "release_threshold_ratio": 1.0},
}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _safe(value):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value))


def _resolved(cfg):
    cfg = json.loads(json.dumps(cfg))
    detector = {**DEFAULTS, **cfg.get("detector", {})}
    detector["events"] = {**DEFAULTS["events"], **detector["events"]}
    if cfg.get("selection_file"):
        selection = json.loads(Path(cfg["selection_file"]).read_text())
        if selection.get("selection_split") != "sim-anomaly-dev":
            raise ValueError("settings must be selected exclusively on sim-anomaly-dev")
        if object_hash(selection["detector"]) != selection["settings_sha256"]:
            raise ValueError("selection settings hash mismatch")
        if cfg.get("locked") and cfg.get("detector") and detector != selection["detector"]:
            raise ValueError("locked configuration disagrees with frozen synthetic selection")
        detector = selection["detector"]
        cfg["selection_sha256"] = file_hash(cfg["selection_file"])
    if cfg.get("locked", False) and not cfg.get("selection_file"):
        raise ValueError("locked evaluation requires a frozen synthetic selection_file")
    if detector["score"] not in SCORES:
        raise ValueError(f"unknown score; choose {SCORES}")
    cfg["detector"] = detector
    return cfg


def _datasets(cfg):
    """Yield (manifest, row iterator). UCR values are read one file at a time."""
    for source in cfg["datasets"]:
        path = Path(source)
        if path.suffix == ".json":
            from sarsim_fm.data.ucr_anomaly import iter_staged

            manifest = json.loads(path.read_text())
            yield {"path": str(path.resolve()), "sha256": file_hash(path), **manifest}, iter_staged(
                manifest, cfg.get("subset_ids"))
        else:
            frame = pd.read_parquet(path)
            required = {"item_id", "series", "clean_prefix_length", "season_length"}
            if required - set(frame):
                raise ValueError(f"{path}: missing columns {sorted(required - set(frame))}")
            if frame.empty or frame.item_id.duplicated().any():
                raise ValueError(f"{path}: empty dataset or duplicate item IDs")
            if cfg.get("subset_items"):
                frame = frame.loc[frame.item_id.isin(cfg["subset_items"])]
                if frame.empty:
                    raise ValueError("subset_items selected no series")
            sidecar = path.with_suffix(".manifest.json")
            manifest = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            if manifest.get("sha256", file_hash(path)) != file_hash(path):
                raise ValueError(f"{path}: dataset manifest hash mismatch")
            yield {**manifest, "name": path.stem, "path": str(path.resolve()),
                   "sha256": file_hash(path), "format": "synthetic"}, frame.to_dict("records")


def _score_frame(cache, prefix, calibration_start, scale):
    q, y, p = cache.quantiles, cache.actual, cache.levels
    frame = pd.DataFrame({"t": cache.time, "actual": y, "horizon": cache.forecast_horizon,
                          "scale_fit": cache.time < calibration_start,
                          "calibration": (cache.time >= calibration_start) & (cache.time < prefix),
                          "evaluation": cache.time >= prefix})
    frame["interval_exceedance"] = interval_exceedance_score(y, q, p)
    frame["robust_residual"] = robust_residual_score(y, q[:, LEVELS.index(0.5)], scale)
    frame["quantile_tail"] = quantile_tail_score(y, q, p)
    return frame


def _events(scores, horizons, thresholds, event_config, offset):
    # Normalization gives build_events a fixed entry threshold and horizon-aware release.
    boundary = np.array([thresholds.thresholds[int(h)] for h in horizons])
    normalized = scores / np.maximum(boundary, 1e-8)
    alerts = apply_thresholds(scores, horizons, thresholds)
    # For an exactly-zero threshold, any positive score is a crossing.
    normalized = np.where(alerts, np.maximum(normalized, np.nextafter(1.0, 2.0)),
                          np.minimum(normalized, 1.0))
    events = build_events(normalized, alerts, **event_config)
    return alerts, [replace(e, start=e.start + offset, end=e.end + offset,
                            peak_time=e.peak_time + offset) for e in events]


def _summary(records, seconds, cache_bytes):
    result = {"n_series": len(records), "runtime_seconds": seconds, "forecast_cache_bytes": cache_bytes}
    for key in ["event_tp", "event_fp", "event_fn", "normal_points", "normal_alerts"]:
        result[key] = sum(r[key] for r in records)
    tp, fp, fn = result["event_tp"], result["event_fp"], result["event_fn"]
    result.update(event_precision=tp / (tp + fp) if tp + fp else 0,
                  event_recall=tp / (tp + fn) if tp + fn else 0,
                  event_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    n = result["normal_points"]
    result["realized_fpr"] = result["normal_alerts"] / n if n else None
    result["false_alerts_per_1000"] = 1000 * result["realized_fpr"] if n else None
    for key in ["auprc", "coverage_80", "coverage_98", "detection_delay", "top_one_hit",
                "clean_prefix_false_alerts_per_1000"]:
        values = [r[key] for r in records if r.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def _run_detection(cfg, model_spec, out=None, device=None, overwrite=False, *, forecaster=None,
                   config_file=None):
    """Run all stages; injectable forecasters make causal integration tests inexpensive."""
    cfg = _resolved(cfg)
    d = cfg["detector"]
    out = Path(out or ROOT / "outputs" / "anomaly" / f"{cfg['name']}-{_safe(model_spec)}")
    if (out / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"{out} is completed; pass --overwrite to replace run artifacts")
    if overwrite and (out / "summary.json").exists():
        (out / "summary.json").unlink()
    # Keep forecasts reusable but never mix old event/score outputs with a resumed run.
    for directory in ["scores", "forecasts"]:
        (out / directory).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if forecaster is None:
        if model_spec == "rolling_mad":
            forecaster = MedianMAD()
        else:
            from sarsim_fm.models import load_forecaster
            from sarsim_fm.runtime import resolve_runtime

            forecaster = load_forecaster(model_spec, resolve_runtime(device or "auto"), d["context_length"])
    identity = checkpoint_identity(model_spec, forecaster)
    env = environment()
    save_source_snapshot(out / "source-snapshot.zip")
    run = {"config": cfg, "config_file": config_file, "environment": env,
           "checkpoint": identity, "model": model_spec, "device": device or "auto",
           "seeds": [], "status": "running"}
    _write(out / "run-config.json", run)
    records, all_events, manifests, pending = [], [], [], []
    seen, calibration_by_dataset = set(), {}
    for manifest, rows in _datasets(cfg):
        manifests.append(manifest)
        name = manifest["name"]
        is_ucr = manifest.get("format") == "ucr"
        for row in rows:
            item = str(row["item_id"])
            if _safe(item) != item:
                raise ValueError("item_id must contain only letters, digits, underscores, dots or dashes")
            if item in seen:
                raise ValueError(f"duplicate item_id across datasets: {item}")
            seen.add(item)
            series = finite_array(row["series"], "series")
            prefix = int(row["clean_prefix_length"])
            if series.ndim != 1 or not 1 < prefix < len(series):
                raise ValueError(f"{item}: invalid clean prefix or series shape")
            calibration_start = int(np.floor(0.8 * prefix)) if is_ucr else int(
                row.get("calibration_start", prefix // 2))
            # Fit residual scale using earlier history, independent of threshold-calibration scores.
            start = max(1, int(0.6 * prefix)) if is_ucr else max(1, calibration_start // 2)
            if not 1 <= start < calibration_start < prefix:
                raise ValueError(f"{item}: calibration requires history and a clean interval")
            # SeasonalNaive(None) uses this period; Chronos ignores the season metadata.
            forecaster.season_length = int(row["season_length"])
            meta_model = {"checkpoint": identity, "code_sha": env["git_sha"],
                          "implementation_sha256": object_hash({
                              k: v for k, v in env["source_files"].items()
                              if "/models/" in k or k.endswith(("/forecast.py", "/baselines.py"))})}
            cache_key = object_hash({"model": meta_model, "item_id": item})[:20]
            shared = Path(cfg.get("cache_dir", ROOT / "outputs" / "anomaly-cache")) / f"{cache_key}.parquet"
            cache = cached_forecast(shared, forecaster, item, series, model=meta_model,
                                    start=start, context_length=d["context_length"], mode=d["mode"],
                                    prediction_length=d["prediction_length"], levels=LEVELS,
                                    batch_size=d["batch_size"])
            for src in [shared, shared.with_suffix(".json")]:
                shutil.copy2(src, out / "forecasts" / f"{_safe(item)}{src.suffix}")
            calibration = (cache.time >= calibration_start) & (cache.time < prefix)
            if not calibration.any():
                raise ValueError(f"{item}: empty clean calibration interval")
            scale_fit = cache.time < calibration_start
            scale = fit_residual_scale(cache.actual[scale_fit],
                                       cache.quantiles[scale_fit, LEVELS.index(0.5)])
            scores = _score_frame(cache, prefix, calibration_start, scale)
            # UCR: thresholds remain per file. Synthetic: pool clean prefixes within each split.
            group = item if is_ucr else name
            calibration_by_dataset.setdefault(group, []).append(
                (scores.loc[calibration, d["score"]].to_numpy(), cache.forecast_horizon[calibration]))
            scores.to_parquet(out / "scores" / f"{_safe(item)}.parquet", index=False)
            pending.append({"row": {k: v for k, v in row.items() if k not in {"series", "clean_series"}},
                            "item": item, "dataset": name, "group": group, "is_ucr": is_ucr,
                            "prefix": prefix, "scale": scale})
            if row.get("seed") is not None:
                run["seeds"].append(int(row["seed"]))
            print(f"Forecasted {item}", flush=True)
    if not pending:
        raise ValueError("no series selected")
    models = {group: fit_thresholds(np.concatenate([s for s, _ in batches]),
                                    np.concatenate([h for _, h in batches]), d["target_fpr"])
              for group, batches in calibration_by_dataset.items()}
    for entry in pending:
        item, row, prefix = entry["item"], entry["row"], entry["prefix"]
        path = out / "scores" / f"{_safe(item)}.parquet"
        frame = pd.read_parquet(path)
        model = models[entry["group"]]
        suffix = frame.t.to_numpy() >= prefix
        raw = frame[d["score"]].to_numpy()
        horizons = frame.horizon.to_numpy()
        alerts, events = _events(raw[suffix], horizons[suffix], model, d["events"], prefix)
        label = []
        if row.get("anomaly_start") is not None and not pd.isna(row["anomaly_start"]):
            a, b = int(row["anomaly_start"]), int(row["anomaly_end"])
            if not prefix <= a < b <= int(frame.t.iloc[-1]) + 1:
                raise ValueError(f"{item}: anomaly interval must lie after clean prefix within series")
            label = [(a, b)]
        times = frame.t.to_numpy()[suffix]
        truth = np.zeros(len(times), bool)
        for a, b in label:
            truth |= (times >= a) & (times < b)
        calibration_mask = frame.calibration.to_numpy()
        calibration_alerts = apply_thresholds(raw[calibration_mask], horizons[calibration_mask], model)
        metric = {"item_id": item, "dataset": entry["dataset"], "family": row.get("family"),
                  "seed": row.get("seed"), **event_metrics(events, label),
                  **normal_metrics(alerts[~truth]), "auprc": point_auprc(truth, raw[suffix]),
                  "clean_prefix_false_alerts_per_1000": float(calibration_alerts.mean() * 1000),
                  "clean_prefix_scored_points": int(calibration_mask.sum()),
                  "clean_prefix_alerts": int(calibration_alerts.sum()),
                  "clean_prefix_metric_scope": "calibration_in_sample",
                  "residual_scale": entry["scale"], "threshold_model": asdict(model)}
        # Coverage is evaluated on held-out normal suffix points, excluding labels.
        forecast = pd.read_parquet(out / "forecasts" / f"{_safe(item)}.parquet")
        q = forecast[[f"q{round(p * 100):02d}" for p in LEVELS]].to_numpy()
        if (~truth).any():
            metric.update(interval_coverage(frame.actual.to_numpy()[suffix][~truth], q[suffix][~truth], LEVELS))
        if entry["is_ucr"]:
            timestamp = top_one_time(events, times, raw[suffix])
            metric.update(top_one_time=timestamp, top_one_hit=ucr_hit(timestamp, *label[0]))
        frame["alert"] = apply_thresholds(raw, horizons, model)
        frame["label"] = False
        frame.loc[suffix, "label"] = truth
        frame.to_parquet(path, index=False)
        records.append(metric)
        all_events.extend({"item_id": item, "dataset": entry["dataset"], **asdict(e)} for e in events)
    pd.DataFrame(all_events, columns=["item_id", "dataset", "start", "end", "peak_time", "peak_score"]).to_parquet(
        out / "events.parquet", index=False)
    _write(out / "dataset-manifest.json", manifests)
    _write(out / "per-series.json", records)
    active = {entry["item"] for entry in pending}
    for folder in ["scores", "forecasts"]:
        for previous in (out / folder).iterdir():
            if previous.suffix in {".parquet", ".json"} and previous.stem not in active:
                previous.unlink()
    total = _summary(records, time.perf_counter() - started,
                     sum(p.stat().st_size for p in (out / "forecasts").glob("*.parquet")))
    datasets = {name: _summary([r for r in records if r["dataset"] == name], None, None)
                for name in sorted({r["dataset"] for r in records})}
    families = {name: _summary([r for r in records if r.get("family") == name], None, None)
                for name in sorted({r["family"] for r in records if r.get("family")})}
    run.update(status="completed", runtime_seconds=total["runtime_seconds"])
    _write(out / "run-config.json", run)
    summary = {"model": model_spec, "config_sha256": object_hash(cfg), "overall": total,
               "datasets": datasets, "families": families}
    _write(out / "summary.json", summary)
    print(json.dumps(total, indent=2))
    return summary


def run_detection(cfg, model_spec, out=None, device=None, overwrite=False, *, forecaster=None,
                  config_file=None):
    """Exclusive per-run writer lock; OS releases it automatically after a crash."""
    import fcntl

    out = Path(out or ROOT / "outputs" / "anomaly" / f"{cfg['name']}-{_safe(model_spec)}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{out} already has an active writer") from exc
        return _run_detection(cfg, model_spec, out, device, overwrite,
                              forecaster=forecaster, config_file=config_file)


def collect_results(root):
    runs, paired = [], []
    for path in sorted(Path(root).glob("**/summary.json")):
        if (path.parent / "run-config.json").exists():
            runs.append({"run": str(path.parent), **json.loads(path.read_text())})
    for i, left in enumerate(runs):
        for right in runs[i + 1:]:
            a = {r["item_id"]: r for r in json.loads((Path(left["run"]) / "per-series.json").read_text())
                 if r.get("top_one_hit") is not None}
            b = {r["item_id"]: r for r in json.loads((Path(right["run"]) / "per-series.json").read_text())
                 if r.get("top_one_hit") is not None}
            common = sorted(a.keys() & b.keys())
            if common:
                paired.append({"left": left["run"], "right": right["run"], "n_pairs": len(common),
                               **paired_family_bootstrap([a[k]["top_one_hit"] for k in common],
                                                        [b[k]["top_one_hit"] for k in common],
                                                        [a[k]["family"] for k in common])})
    return {"runs": runs, "paired_localization_comparisons": paired}
