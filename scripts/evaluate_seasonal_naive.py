from pathlib import Path

import numpy as np
import pandas as pd


def seasonal_naive(
    history: np.ndarray,
    horizon: int = 18,
    seasonality: int = 12,
) -> np.ndarray:
    cycle = history[-seasonality:]
    repeats = int(np.ceil(horizon / seasonality))
    return np.tile(cycle, repeats)[:horizon]


def smape(actual, forecast):
    denominator = np.abs(actual) + np.abs(forecast)

    values = np.divide(
        2.0 * np.abs(actual - forecast),
        denominator,
        out=np.zeros_like(actual, dtype=np.float64),
        where=denominator != 0,
    )

    return float(np.mean(values))


def mase(actual, forecast, history, seasonality=12):
    differences = np.abs(
        history[seasonality:] - history[:-seasonality]
    )

    scale = np.mean(differences)

    if not np.isfinite(scale) or scale <= 1e-12:
        return np.nan

    return float(np.mean(np.abs(actual - forecast)) / scale)


data = pd.read_parquet("data/m4/monthly.parquet")
rows = []

for row in data.itertuples(index=False):
    history = np.asarray(row.train_values, dtype=np.float32)
    actual = np.asarray(row.test_values, dtype=np.float32)

    forecast = seasonal_naive(history)

    rows.append(
        {
            "item_id": row.item_id,
            "smape": smape(actual, forecast),
            "mase": mase(actual, forecast, history),
            "forecast": forecast.tolist(),
        }
    )

results = pd.DataFrame(rows)

output_dir = Path("outputs/baselines")
output_dir.mkdir(parents=True, exist_ok=True)

results.to_parquet(
    output_dir / "seasonal-naive-monthly.parquet",
    index=False,
)

summary = {
    "mean_smape": results["smape"].mean(),
    "mean_mase": results["mase"].mean(),
    "valid_mase_series": results["mase"].notna().sum(),
}

print(summary)