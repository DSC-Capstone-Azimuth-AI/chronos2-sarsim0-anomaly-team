"""Leakage-free batched sequential inference and content-addressed forecast caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.scores import finite_array, rearrange_quantiles


@dataclass(frozen=True)
class ForecastCache:
    item_id: str
    time: np.ndarray
    actual: np.ndarray
    quantiles: np.ndarray
    levels: np.ndarray
    forecast_horizon: np.ndarray


def source_hash(series) -> str:
    x = np.ascontiguousarray(finite_array(series, "series"), dtype="<f8")
    return hashlib.sha256(x.tobytes()).hexdigest()


def _forecast(forecaster, item_id, series, start, context_length, prediction_length, levels, batch_size):
    x = finite_array(series, "series")
    if x.ndim != 1 or not 1 <= start < len(x):
        raise ValueError("series must be one-dimensional and start must be inside it after history")
    for value, name in [(start, "start"), (context_length, "context_length"),
                        (prediction_length, "prediction_length"), (batch_size, "batch_size")]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    p = finite_array(levels, "levels")
    if p.ndim != 1 or len(p) < 2 or (np.diff(p) <= 0).any() or ((p <= 0) | (p >= 1)).any():
        raise ValueError("levels must be strictly increasing in (0, 1)")
    q = np.empty((len(x) - start, len(p)))
    horizons = np.empty(len(x) - start, dtype=np.int64)
    origins = range(start, len(x), prediction_length)
    for offset in range(0, len(origins), batch_size):
        batch = origins[offset:offset + batch_size]
        # Only one batch of contexts exists at once. Copies protect the input from model mutation.
        contexts = [x[max(0, t - context_length):t].copy() for t in batch]
        seasons = np.full(len(batch), getattr(forecaster, "season_length", None) or 1)
        pred = rearrange_quantiles(forecaster.predict_quantiles(
            contexts, prediction_length, list(levels), batch_size, seasons,
        ))
        if pred.shape != (len(batch), prediction_length, len(p)):
            raise ValueError("forecaster output shape must be (batch, prediction_length, n_levels)")
        for t, row in zip(batch, pred, strict=True):
            n = min(prediction_length, len(x) - t)
            q[t - start:t - start + n] = row[:n]
            horizons[t - start:t - start + n] = np.arange(1, n + 1)
    return ForecastCache(str(item_id), np.arange(start, len(x)), x[start:].copy(), q, p, horizons)


def rolling_forecast(forecaster, item_id, series, start, context_length, levels, batch_size):
    """Forecast each t using only series[:t]."""
    return _forecast(forecaster, item_id, series, start, context_length, 1, levels, batch_size)


def block_forecast(forecaster, item_id, series, start, context_length, prediction_length,
                   levels, batch_size):
    """Forecast disjoint blocks; each point retains its one-based horizon."""
    return _forecast(forecaster, item_id, series, start, context_length, prediction_length,
                     levels, batch_size)


def cached_forecast(path: Path, forecaster, item_id, series, *, model, start,
                    context_length, mode="rolling", prediction_length=18, levels, batch_size=32):
    """Reuse only matching sidecars and verified Parquet bytes; otherwise regenerate."""
    path = Path(path)
    if mode not in {"rolling", "block"}:
        raise ValueError("inference mode must be rolling or block")
    meta = {
        "schema_version": 1, "model": model, "item_id": str(item_id), "start": start,
        "context_length": context_length, "mode": mode, "prediction_length": prediction_length,
        "levels": list(levels), "batch_size": batch_size, "source_sha256": source_hash(series),
        "season_length": getattr(forecaster, "season_length", None),
    }
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        stored = json.loads(sidecar.read_text())
        digest = stored.pop("parquet_sha256", None)
        if stored == meta and digest == hashlib.sha256(path.read_bytes()).hexdigest():
            frame = pd.read_parquet(path)
            qcols = [f"q{round(p * 100):02d}" for p in levels]
            return ForecastCache(str(item_id), frame.t.to_numpy(), frame.actual.to_numpy(),
                                 frame[qcols].to_numpy(), np.array(levels),
                                 frame.forecast_horizon.to_numpy())
    result = _forecast(forecaster, item_id, series, start, context_length,
                       1 if mode == "rolling" else prediction_length, levels, batch_size)
    frame = pd.DataFrame({"item_id": result.item_id, "t": result.time, "actual": result.actual})
    for i, p in enumerate(levels):
        frame[f"q{round(p * 100):02d}"] = result.quantiles[:, i]
    frame["forecast_horizon"] = result.forecast_horizon
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    sidecar.write_text(json.dumps({**meta, "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                                 indent=2, sort_keys=True))
    return result
