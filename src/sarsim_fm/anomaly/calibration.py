"""Horizon-specific split-conformal thresholds and normal-point diagnostics."""

from dataclasses import dataclass

import numpy as np

from sarsim_fm.anomaly.scores import _inputs, finite_array, quantile_at

TARGET_FPRS = (0.005, 0.01, 0.02)


@dataclass(frozen=True)
class ThresholdModel:
    target_fpr: float
    thresholds: dict[int, float]
    calibration_counts: dict[int, int]


def _arrays(scores, horizons):
    s, h = finite_array(scores, "scores"), finite_array(horizons, "horizons")
    if s.ndim != 1 or h.shape != s.shape or not len(s):
        raise ValueError("scores and horizons must be matching nonempty one-dimensional arrays")
    if (h < 1).any() or (h != np.floor(h)).any():
        raise ValueError("horizons must be positive integers")
    return s, h.astype(int)


def fit_thresholds(scores, horizons, target_fpr):
    """k=ceil((n+1)(1-alpha)) order statistic; alert strictly above it.

    The caller supplies clean calibration observations only. Reject k>n rather
    than substituting contaminated observations or another horizon's threshold.
    Serial dependence means the usual exchangeability guarantee is approximate.
    """
    s, h = _arrays(scores, horizons)
    if target_fpr not in TARGET_FPRS:
        raise ValueError(f"target_fpr must be one of {TARGET_FPRS}")
    thresholds, counts = {}, {}
    for horizon in np.unique(h):
        values = s[h == horizon]
        n = len(values)
        k = int(np.ceil((n + 1) * (1 - target_fpr) - 1e-12))
        if k > n:
            raise ValueError(f"horizon {horizon}: insufficient calibration samples ({n}) "
                             f"for target_fpr={target_fpr}; need >= {round(1 / target_fpr) - 1}")
        thresholds[int(horizon)] = float(np.partition(values, k - 1)[k - 1])
        counts[int(horizon)] = n
    return ThresholdModel(float(target_fpr), thresholds, counts)


def apply_thresholds(scores, horizons, model):
    s, h = _arrays(scores, horizons)
    missing = set(h) - set(model.thresholds)
    if missing:
        raise ValueError(f"uncalibrated forecast horizons: {sorted(missing)}")
    return s > np.array([model.thresholds[int(i)] for i in h])


def normal_metrics(alerts):
    a = np.asarray(alerts)
    if a.dtype != bool or a.ndim != 1:
        raise ValueError("alerts must be a one-dimensional Boolean array")
    rate = float(a.mean()) if len(a) else None
    return {"normal_points": len(a), "normal_alerts": int(a.sum()),
            "realized_fpr": rate, "false_alerts_per_1000": None if rate is None else rate * 1000}


def interval_coverage(actual, quantiles, levels):
    y, q, p = _inputs(actual, quantiles, levels, 1e-8)
    return {f"coverage_{percent}": float(np.mean((y >= quantile_at(q, p, lo)) &
                                                 (y <= quantile_at(q, p, hi))))
            for percent, lo, hi in [(80, 0.1, 0.9), (98, 0.01, 0.99)]}
