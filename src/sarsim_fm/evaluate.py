"""Score any forecaster on any eval set; save per-series quantiles and a summary."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.data.eval_sets import EvalSet
from sarsim_fm.metrics import WQL_LEVELS, coverage, mase, quantile_loss, scaled_quantile_loss, smape

# Saved for every series. The tails (0.01/0.99) are what forecast-residual anomaly detection uses.
LEVELS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def evaluate(forecaster, eval_set: EvalSet, batch_size: int) -> tuple[pd.DataFrame, dict]:
    levels = np.asarray(LEVELS)
    wql_idx = [LEVELS.index(q) for q in WQL_LEVELS]
    median_idx = LEVELS.index(0.5)

    start = time.perf_counter()
    forecasts = forecaster.predict_quantiles(
        eval_set.contexts, eval_set.horizon, list(LEVELS), batch_size, eval_set.season_lengths
    )
    seconds = time.perf_counter() - start

    rows = []
    wql_num, wql_den = 0.0, 0.0
    for i, item_id in enumerate(eval_set.item_ids):
        actual, history, q = eval_set.targets[i], eval_set.contexts[i], forecasts[i]
        season = int(eval_set.season_lengths[i])
        median = q[:, median_idx]
        wql_num += 2.0 * quantile_loss(actual, q[:, wql_idx], levels[wql_idx]).mean()
        wql_den += np.abs(actual).sum()
        rows.append(
            {
                "item_id": item_id,
                "smape": smape(actual, median),
                "mase": mase(actual, median, history, season),
                "sql": scaled_quantile_loss(actual, q[:, wql_idx], levels[wql_idx], history, season),
                "coverage_80": coverage(actual, q[:, LEVELS.index(0.1)], q[:, LEVELS.index(0.9)]),
                "coverage_98": coverage(actual, q[:, 0], q[:, -1]),
                "quantiles": q.tolist(),
            }
        )

    per_series = pd.DataFrame(rows)
    summary = {
        "n_series": len(per_series),
        "horizon": eval_set.horizon,
        "smape": float(per_series["smape"].mean()),
        "mase": float(per_series["mase"].mean()),
        "mase_valid_series": int(per_series["mase"].notna().sum()),
        "sql": float(per_series["sql"].mean()),
        "wql":float(wql_num / wql_den) if wql_den > 0 else float("nan"),
        "coverage_80": float(per_series["coverage_80"].mean()),
        "coverage_98": float(per_series["coverage_98"].mean()),
        "predict_seconds": round(seconds, 1),
    }
    return per_series, summary


def save_eval(out_dir: Path, name: str, per_series: pd.DataFrame, summary: dict, meta: dict) -> None:
    """Write `<out_dir>/eval/<name>.parquet` and merge into `<out_dir>/eval/summary.json`."""
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    per_series.attrs["quantile_levels"] = list(LEVELS)
    per_series.to_parquet(eval_dir / f"{name}.parquet", index=False)

    summary_path = eval_dir / "summary.json"
    all_summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    all_summaries[name] = {**summary, **meta, "quantile_levels": list(LEVELS)}
    summary_path.write_text(json.dumps(all_summaries, indent=2))
