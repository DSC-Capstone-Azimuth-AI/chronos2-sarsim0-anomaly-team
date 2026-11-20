"""SarSim0 configuration, the training stream, and fixed held-out simulated eval sets."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

# SarSim0's numba `parallel=True` kernels segfault next to torch's OpenMP on macOS; numba's own
# workqueue layer avoids the clash. Must be set before numba compiles anything.
if sys.platform == "darwin":
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import pandas as pd
import torch
from sarsim0 import SarSim0Generator, SarSim0StreamingDataset, SarSimConfig
from torch.utils.data import IterableDataset

from sarsim_fm.data.eval_sets import EVAL_DIR, write_eval_set

# Applied under every `sarsim:` block. Univariate, no covariates, no padding: the setup the
# original Chronos-2 experiments used. Everything else falls back to SarSimConfig (paper Table 9).
SARSIM_DEFAULTS = {
    "series_length": 6000,
    "burn_in": 200,
    "multivariate_prob": 0.0,
    "with_covariates_prob": 0.0,
    "pad_max": 0,
}

_TUPLE_FIELDS = {f.name for f in fields(SarSimConfig) if f.name.endswith("_range")}

_PROFILE_KEYS = {"grid_profile"}

def _grid_profile(overrides: dict | None) -> str | None:
    return (overrides or {}).get("grid_profile")


def apply_hourly_grid_profile(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Layer a randomized weekday/weekend building schedule onto a SarSim0 series."""
    n = len(series)
    t = np.arange(n, dtype=np.float64) + int(rng.integers(0, 168))
    hour = t % 24.0
    day = (t // 24.0) % 7.0  # 0..4 weekdays, 5..6 weekend

    # Trapezoidal daily duty cycle: sharp morning ramp, flat daytime plateau, evening ramp-down
    start_h = rng.uniform(5.0, 8.5)
    ramp_up = rng.uniform(1.0, 2.5)
    end_h = rng.uniform(16.5, 22.5)
    ramp_dn = rng.uniform(1.0, 2.5)

    up = np.clip((hour - start_h) / ramp_up, 0.0, 1.0)
    dn = np.clip((end_h - hour) / ramp_dn, 0.0, 1.0)
    daily = np.minimum(up, dn)

    # 5 active weekdays + 2 reduced-load weekend days
    weekend_scale = rng.uniform(0.15, 0.55)
    day_scale = np.where(day < 5.0, 1.0, weekend_scale)

    baseload = rng.uniform(30.0, 80.0)
    swing = baseload * rng.uniform(0.6, 1.8)
    schedule = baseload + swing * daily * day_scale

    # Standardize raw SarSim0 output (clipped to avoid +/-1e6 outliers) as heteroscedastic noise
    raw = np.nan_to_num(series.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(raw, [2, 98])
    raw = np.clip(raw, lo, hi)
    resid = (raw - np.mean(raw)) / (np.std(raw) + 1e-6)

    # Noise scales with active load so nights/weekends stay quiet and weekdays vary more
    noise_amp = baseload * rng.uniform(0.03, 0.08) + swing * daily * day_scale * rng.uniform(0.06, 0.16)
    return np.maximum(schedule + resid * noise_amp, baseload * 0.5)

def sarsim_config(context_length: int, prediction_length: int, overrides: dict | None = None) -> SarSimConfig:
    clean = {k: v for k, v in (overrides or {}).items() if k not in _PROFILE_KEYS}
    params = {**SARSIM_DEFAULTS, **clean}
    params = {k: tuple(v) if k in _TUPLE_FIELDS else v for k, v in params.items()}
    if "seasonality_pairs" in params:
        params["seasonality_pairs"] = [tuple(p) for p in params["seasonality_pairs"]]
    return SarSimConfig(
        context_window=context_length, prediction_window=prediction_length, **params
    )
    
class _GridProfileStream(IterableDataset):
    def __init__(self, base: SarSim0StreamingDataset, profile: str, seed: int):
        self.base = base
        self.profile = profile
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(self.seed + 10_000 + worker_id)
        for batch in self.base:
            ctx = batch["context"].numpy()
            tgt = batch["future_target"].numpy()
            c_len = ctx.shape[1]
            out_ctx, out_tgt = np.empty_like(ctx), np.empty_like(tgt)
            for i in range(ctx.shape[0]):
                full = np.concatenate([ctx[i], tgt[i]])
                shaped = apply_hourly_grid_profile(full, rng).astype(ctx.dtype)
                out_ctx[i], out_tgt[i] = shaped[:c_len], shaped[c_len:]
            yield {**batch, "context": torch.from_numpy(out_ctx), "future_target": torch.from_numpy(out_tgt)}
            
def training_stream(data_cfg: dict) -> IterableDataset:
    """Infinite stream of Chronos-2 batches. Worker `i` is seeded with `seed + i`."""
    base = SarSim0StreamingDataset(
        context_length=data_cfg["context_length"],
        prediction_length=data_cfg["prediction_length"],
        batch_size=data_cfg["batch_size"],
        config=sarsim_config(
            data_cfg["context_length"], data_cfg["prediction_length"], data_cfg.get("sarsim")
        ),
        seed=data_cfg["seed"],
    )
    profile = _grid_profile(data_cfg.get("sarsim"))
    if profile == "hourly":
        return _GridProfileStream(base, profile, data_cfg["seed"])
    if profile is not None:
        raise ValueError(f"Unknown sarsim.grid_profile: {profile!r}")
    return base

def make_heldout_split(name: str, split: dict, defaults: dict) -> None:
    """Generate one held-out split, one series per generator call.

    SarSim0 draws a single set of SARIMA parameters per vectorized call, so a large
    `generate_batch` would hold only a handful of distinct processes.
    """
    spec = {**defaults, **split, "sarsim": {**defaults.get("sarsim", {}), **split.get("sarsim", {})}}
    config = sarsim_config(spec["context_length"], spec["prediction_length"], spec["sarsim"])
    generator = SarSim0Generator(config=config, seed=spec["seed"])
    
    profile = _grid_profile(spec["sarsim"])
    rng = np.random.default_rng(spec["seed"] + 10_000)
    
    contexts, targets = [], []
    for _ in range(spec["n_series"]):
        context, target = generator.generate_batch(batch_size=1)
        c, t_arr = np.asarray(context[0]), np.asarray(target[0])
        if profile == "hourly":
            shaped = apply_hourly_grid_profile(np.concatenate([c, t_arr]), rng)
            c, t_arr = shaped[: len(c)], shaped[len(c) :]
        contexts.append(c.tolist())
        targets.append(t_arr.tolist())

    frame = pd.DataFrame(
        {
            "item_id": [f"{name}-{i:06d}" for i in range(spec["n_series"])],
            "context": contexts,
            "target": targets,
            "season_length": spec["season_length"],
        }
    )
    path = write_eval_set(frame, name)
    (EVAL_DIR / f"{name}.manifest.json").write_text(json.dumps({"name": name, **spec}, indent=2))
    print(f"Wrote {path} ({len(frame)} series, seed {spec['seed']})")


def draw_clean_series(generator, length, context_length, max_retries, rejections):
    """Reject invalid raw outputs before concatenation/injection, with bounded retries."""
    from sarsim_fm.data.anomaly_injection import clean_rejection_reason

    for _ in range(max_retries):
        context, target = generator.generate_batch(batch_size=1)
        x = np.concatenate([np.asarray(context[0]).ravel(), np.asarray(target[0]).ravel()])
        reason = clean_rejection_reason(x, context_length)
        if len(x) < length:
            reason = "too_short"
        if reason is None:
            return x[:length].astype(np.float64)
        rejections[reason] += 1
    raise ValueError(f"clean generation exhausted {max_retries} retries: {dict(rejections)}")


def make_anomaly_split(name: str, spec: dict, out: Path) -> Path:
    """Each series has its own SarSim0 seed; normal splits have empty labels."""
    import hashlib

    from sarsim_fm.data.anomaly_injection import KINDS, inject_anomaly

    length, prefix = int(spec["series_length"]), int(spec["clean_prefix_length"])
    context = int(spec["context_length"])
    if not 1 <= context < prefix < length:
        raise ValueError("require 1 <= context_length < clean_prefix_length < series_length")
    if spec["n_series"] < 1 or spec.get("max_retries", 30) < 1:
        raise ValueError("n_series and max_retries must be positive")
    rows, rejections = [], Counter({"non_finite": 0, "clipped": 0, "all_zero_context": 0, "too_short": 0})
    for i in range(spec["n_series"]):
        seed = int(spec["seed"]) + i
        rng = np.random.default_rng(seed)
        config = sarsim_config(prefix, length - prefix, spec.get("sarsim"))
        generator = SarSim0Generator(config=config, seed=seed)
        clean = draw_clean_series(generator, length, context, spec.get("max_retries", 30), rejections)
        if spec.get("normal", False):
            series, start, end, kind, severity = clean.copy(), None, None, "normal", 0.0
        else:
            period = int(spec["season_length"])
            duration = max(1, round(rng.uniform(*spec["duration_periods"]) * period))
            duration = min(duration, length - prefix)
            start = int(rng.integers(prefix, length - duration + 1))
            kind = KINDS[i % len(KINDS)]
            severity = float(rng.uniform(*spec["severity_range"]))
            series, label = inject_anomaly(clean, rng, kind, severity, duration, start, period)
            end = label.end
        rows.append({
            "item_id": f"{name}-{i:06d}", "series": series.tolist(), "clean_series": clean.tolist(),
            "anomaly_start": start, "anomaly_end": end, "anomaly_kind": kind,
            "severity": severity, "seed": seed, "season_length": spec["season_length"],
            "clean_prefix_length": prefix,
        })
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    manifest = {
        "name": name, "config": spec, "seed_range": [spec["seed"], spec["seed"] + len(rows)],
        "n_series": len(rows), "rejections": dict(sorted(rejections.items())),
        "index_convention": "zero-based half-open", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {path} ({len(rows)} series; rejected {sum(rejections.values())})")
    return path
