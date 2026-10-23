"""SarSim0 configuration, the training stream, and fixed held-out simulated eval sets."""

from __future__ import annotations

import json
from dataclasses import fields

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
