"""Forecast metrics. Point metrics take the median forecast; WQL and coverage use quantiles."""

from __future__ import annotations

import numpy as np

# Levels used for WQL, matching Chronos and GiftEval.
WQL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE as a fraction in [0, 2]. Multiply by 100 for the M4 convention."""
    denominator = np.abs(actual) + np.abs(forecast)
    values = np.divide(
        2.0 * np.abs(actual - forecast),
        denominator,
        out=np.zeros_like(actual, dtype=np.float64),
        where=denominator != 0,
    )
    return float(np.mean(values))


def naive_scale(history: np.ndarray, season_length: int) -> float:
    """In-sample seasonal-naive MAE. NaN when undefined or ~0 (e.g. constant history)."""
    if len(history) <= season_length:
        return np.nan
    scale = float(np.mean(np.abs(history[season_length:] - history[:-season_length])))
    return scale if np.isfinite(scale) and scale > 1e-12 else np.nan


def mase(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray, season_length: int) -> float:
    return float(np.mean(np.abs(actual - forecast))) / naive_scale(history, season_length)


def quantile_loss(actual: np.ndarray, quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pinball loss summed over time, per level. `quantiles` has shape (horizon, n_levels)."""
    diff = actual[:, None] - quantiles
    return np.sum(np.maximum(levels * diff, (levels - 1.0) * diff), axis=0)


def scaled_quantile_loss(
    actual: np.ndarray, quantiles: np.ndarray, levels: np.ndarray, history: np.ndarray, season_length: int
) -> float:
    """Per-series probabilistic analogue of MASE: 2 x mean pinball / naive scale.

    Equals MASE for a point forecast. Unlike aggregate WQL it is not dominated by a few
    large-magnitude series, which matters for SarSim0 (some series run up to its ±1e6 clip).
    """
    loss = 2.0 * quantile_loss(actual, quantiles, levels).mean() / len(actual)
    return float(loss) / naive_scale(history, season_length)


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals inside [lower, upper]."""
    return float(np.mean((actual >= lower) & (actual <= upper)))
