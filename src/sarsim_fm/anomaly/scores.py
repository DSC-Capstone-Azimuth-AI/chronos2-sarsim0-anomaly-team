"""Finite anomaly scores. Quantile axes always correspond to increasing levels."""

from __future__ import annotations

import numpy as np


def finite_array(value, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def rearrange_quantiles(quantiles: np.ndarray) -> np.ndarray:
    """Return nondecreasing quantiles along the final axis, without mutation."""
    q = finite_array(quantiles, "quantiles")
    if q.ndim < 1 or q.shape[-1] < 2:
        raise ValueError("quantiles must have a final axis with at least two levels")
    return np.sort(q, axis=-1)


def _inputs(actual, quantiles, levels, eps):
    positive(eps, "eps")
    y = finite_array(actual, "actual")
    q = rearrange_quantiles(quantiles)
    p = finite_array(levels, "levels")
    if p.ndim != 1 or len(p) != q.shape[-1]:
        raise ValueError("levels shape must match the final quantile axis")
    if np.any(np.diff(p) <= 0) or np.any((p <= 0) | (p >= 1)):
        raise ValueError("levels must be strictly increasing within (0, 1)")
    if y.shape != q.shape[:-1]:
        raise ValueError("actual shape must equal quantiles.shape[:-1]")
    return y, q, p


def quantile_at(quantiles, levels, level):
    """Linear interpolation at a level within the supplied quantile grid."""
    if not levels[0] <= level <= levels[-1]:
        raise ValueError(f"requested level {level} is outside the quantile grid")
    right = min(int(np.searchsorted(levels, level, side="right")), len(levels) - 1)
    left = max(0, right - 1)
    weight = (level - levels[left]) / (levels[right] - levels[left])
    return quantiles[..., left] * (1 - weight) + quantiles[..., right] * weight


def interval_exceedance_score(actual, quantiles, levels, lower=0.01, upper=0.99, eps=1e-8):
    """Zero inside the interval; outside distance divided by interval width."""
    y, q, p = _inputs(actual, quantiles, levels, eps)
    if not lower < upper:
        raise ValueError("lower must be less than upper")
    lo, hi = quantile_at(q, p, lower), quantile_at(q, p, upper)
    return np.maximum(np.maximum(lo - y, y - hi), 0) / np.maximum(hi - lo, eps)


def fit_residual_scale(actual, median_forecast, eps=1e-8) -> float:
    """1.4826 times centered residual MAD, floored by eps; clean data only."""
    positive(eps, "eps")
    y, median = finite_array(actual, "actual"), finite_array(median_forecast, "median_forecast")
    if y.shape != median.shape or y.size == 0:
        raise ValueError("actual and median_forecast must have matching nonempty shapes")
    residual = y - median
    return float(max(1.4826 * np.median(np.abs(residual - np.median(residual))), eps))


def robust_residual_score(actual, median_forecast, scale, eps=1e-8):
    """Absolute median residual divided by clean residual scale."""
    positive(eps, "eps")
    positive(scale, "scale")
    y, median = finite_array(actual, "actual"), finite_array(median_forecast, "median_forecast")
    if y.shape != median.shape:
        raise ValueError("actual and median_forecast shapes must match")
    return np.abs(y - median) / max(scale, eps)


def quantile_tail_score(actual, quantiles, levels, eps=1e-8):
    """-log(two-sided interpolated tail probability).

    Outside the grid use exponential continuation with the nearest positive
    quantile spacing. Tied quantiles use the midpoint CDF and an eps scale.
    This avoids constant scores for all observations beyond q01/q99.
    """
    y, q, p = _inputs(actual, quantiles, levels, eps)
    values, rows = y.ravel(), q.reshape(-1, len(p))
    index = np.arange(len(values))
    right = np.clip(np.sum(rows < values[:, None], axis=1), 1, len(p) - 1)
    left = right - 1
    lo, hi = rows[index, left], rows[index, right]
    weight = np.clip((values - lo) / np.maximum(hi - lo, eps), 0, 1)
    cdf = p[left] + weight * (p[right] - p[left])
    ties = rows == values[:, None]
    count = ties.sum(axis=1)
    cdf = np.where(count > 0, (ties * p).sum(axis=1) / np.maximum(count, 1), cdf)
    log_tail = np.log(np.maximum(np.minimum(cdf, 1 - cdf), np.finfo(float).tiny))
    gaps_lo = rows - rows[:, :1]
    widths_lo = np.min(np.where(gaps_lo > 0, gaps_lo, np.inf), axis=1)
    gaps_hi = rows[:, -1:] - rows
    widths_hi = np.min(np.where(gaps_hi > 0, gaps_hi, np.inf), axis=1)
    widths_lo = np.where(np.isfinite(widths_lo), np.maximum(widths_lo, eps), eps)
    widths_hi = np.where(np.isfinite(widths_hi), np.maximum(widths_hi, eps), eps)
    log_tail = np.where(values < rows[:, 0],
                        np.log(p[0]) - (rows[:, 0] - values) / widths_lo, log_tail)
    log_tail = np.where(values > rows[:, -1],
                        np.log1p(-p[-1]) - (values - rows[:, -1]) / widths_hi, log_tail)
    return np.maximum(0, -np.log(2.0) - log_tail).reshape(y.shape)
