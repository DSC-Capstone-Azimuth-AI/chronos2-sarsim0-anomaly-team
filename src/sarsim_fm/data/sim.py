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
from sarsim0 import SarSim0Generator, SarSim0StreamingDataset, SarSimConfig

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


def sarsim_config(context_length: int, prediction_length: int, overrides: dict | None = None) -> SarSimConfig:
    params = {**SARSIM_DEFAULTS, **(overrides or {})}
    params = {k: tuple(v) if k in _TUPLE_FIELDS else v for k, v in params.items()}
    if "seasonality_pairs" in params:
        params["seasonality_pairs"] = [tuple(p) for p in params["seasonality_pairs"]]
    return SarSimConfig(
        context_window=context_length, prediction_window=prediction_length, **params
    )


def training_stream(data_cfg: dict) -> SarSim0StreamingDataset:
    """Infinite stream of Chronos-2 batches. Worker `i` is seeded with `seed + i`."""
    return SarSim0StreamingDataset(
        context_length=data_cfg["context_length"],
        prediction_length=data_cfg["prediction_length"],
        batch_size=data_cfg["batch_size"],
        config=sarsim_config(
            data_cfg["context_length"], data_cfg["prediction_length"], data_cfg.get("sarsim")
        ),
        seed=data_cfg["seed"],
    )


def make_heldout_split(name: str, split: dict, defaults: dict) -> None:
    """Generate one held-out split, one series per generator call.

    SarSim0 draws a single set of SARIMA parameters per vectorized call, so a large
    `generate_batch` would hold only a handful of distinct processes.
    """
    spec = {**defaults, **split, "sarsim": {**defaults.get("sarsim", {}), **split.get("sarsim", {})}}
    config = sarsim_config(spec["context_length"], spec["prediction_length"], spec["sarsim"])
    generator = SarSim0Generator(config=config, seed=spec["seed"])

    contexts, targets = [], []
    for _ in range(spec["n_series"]):
        context, target = generator.generate_batch(batch_size=1)
        contexts.append(context[0].tolist())
        targets.append(target[0].tolist())

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
