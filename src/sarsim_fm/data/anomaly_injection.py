"""Deterministic, non-mutating anomaly injection with half-open labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sarsim_fm.anomaly.scores import finite_array, positive

KINDS = (
    "spike", "dip", "level_shift", "variance_burst", "trend_change",
    "seasonal_disruption", "dropout", "stuck_sensor", "reversed_segment", "time_warp",
)


@dataclass(frozen=True)
class AnomalyInterval:
    start: int
    end: int
    kind: str
    severity: float


def clean_rejection_reason(series, context_length: int) -> str | None:
    x = np.asarray(series)
    if not np.isfinite(x).all():
        return "non_finite"
    if (np.abs(x) >= 1e6).any():
        return "clipped"
    if not np.any(x[:context_length]):
        return "all_zero_context"
    return None


def inject_anomaly(clean, rng, kind, severity, duration, location, season_length=24):
    """Return a copy plus label; integer duration/location are measured in samples.

    The config generator converts period-relative durations to samples. Local
    scale uses preceding history, or the initial clean neighborhood at location 0.
    """
    x = finite_array(clean, "clean")
    positive(severity, "severity")
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("clean must be a one-dimensional series with at least two points")
    if kind not in KINDS:
        raise ValueError(f"unknown anomaly kind {kind!r}; expected one of {KINDS}")
    if not isinstance(duration, (int, np.integer)) or duration < 1:
        raise ValueError("duration must be a positive integer")
    if not isinstance(location, (int, np.integer)) or not 0 <= location < len(x):
        raise ValueError("location must be an integer inside the series")
    if location + duration > len(x) or season_length < 1:
        raise ValueError("anomaly interval is out of bounds or season_length is invalid")
    start, end = int(location), int(location + duration)
    history = x[max(0, start - max(32, 4 * season_length)):start]
    if not len(history):
        history = x[:max(32, 4 * season_length)]
    center = np.median(history)
    scale = max(1.4826 * np.median(np.abs(history - center)), 1e-8)
    amplitude = severity * scale
    out = x.copy()
    segment = x[start:end]
    if kind == "spike":
        out[start:end] += amplitude * np.maximum(0.1, np.hanning(duration + 2)[1:-1])
    elif kind == "dip":
        out[start:end] -= amplitude * np.maximum(0.1, np.hanning(duration + 2)[1:-1])
    elif kind == "level_shift":
        out[start:end] += amplitude
    elif kind == "variance_burst":
        out[start:end] += rng.normal(0, amplitude, duration)
    elif kind == "trend_change":
        out[start:end] += np.linspace(amplitude / duration, amplitude, duration)
    elif kind == "seasonal_disruption":
        out[start:end] += amplitude * np.cos(2 * np.pi * np.arange(duration) / season_length)
    elif kind == "dropout":
        out[start:end] = 0.0
    elif kind == "stuck_sensor":
        out[start:end] = x[max(0, start - 1)]
    elif kind == "reversed_segment":
        out[start:end] = segment[::-1]
    elif kind == "time_warp":
        grid = np.linspace(0, 1, duration)
        out[start:end] = np.interp(grid ** (1 + severity), grid, segment)
    if not np.isfinite(out).all():
        raise ValueError("injection produced non-finite values; reduce severity")
    return out, AnomalyInterval(start, end, kind, float(severity))
