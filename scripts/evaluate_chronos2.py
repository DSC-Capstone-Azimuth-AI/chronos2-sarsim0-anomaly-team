from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline


def smape(actual, forecast):
    denominator = np.abs(actual) + np.abs(forecast)

    return np.mean(
        np.divide(
            2.0 * np.abs(actual - forecast),
            denominator,
            out=np.zeros_like(actual, dtype=np.float64),
            where=denominator != 0,
        )
    )


def mase(actual, forecast, history, seasonality=12):
    scale = np.mean(
        np.abs(history[seasonality:] - history[:-seasonality])
    )

    if not np.isfinite(scale) or scale <= 1e-12:
        return np.nan

    return np.mean(np.abs(actual - forecast)) / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="amazon/chronos-2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    data = pd.read_parquet("data/m4/monthly.parquet")

    contexts = [
        np.asarray(values, dtype=np.float32)
        for values in data["train_values"]
    ]

    actuals = [
        np.asarray(values, dtype=np.float32)
        for values in data["test_values"]
    ]

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map="cuda",
    )

    _, median_forecasts = pipeline.predict_quantiles(
        inputs=contexts,
        prediction_length=18,
        quantile_levels=[0.1, 0.5, 0.9],
        context_length=512,
        batch_size=args.batch_size,
    )

    rows = []

    for item_id, history, actual, forecast in zip(
        data["item_id"],
        contexts,
        actuals,
        median_forecasts,
    ):
        forecast = (
            forecast.detach()
            .cpu()
            .numpy()
            .squeeze()
            .astype(np.float32)
        )

        rows.append(
            {
                "item_id": item_id,
                "smape": float(smape(actual, forecast)),
                "mase": float(mase(actual, forecast, history)),
                "forecast": forecast.tolist(),
            }
        )

    results = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)

    print("Mean sMAPE:", results["smape"].mean())
    print("Mean MASE:", results["mase"].mean())


if __name__ == "__main__":
    main()