"""Model registry.

A forecaster spec is `<type>` or `<type>:<path-or-hub-id>`, e.g.
    seasonal_naive
    chronos2:amazon/chronos-2
    chronos2:outputs/chronos2-ft-cpu/checkpoint-final
    patchtst:outputs/patchtst-scratch-20k/checkpoint-final

To add a model family, write `models/<name>.py` with `load(path, runtime, context_length)`
returning a `Forecaster`, plus `train(cfg, runtime, run_dir) -> checkpoint path` if trainable,
then register both below (and `timed_steps(cfg, runtime, steps)` for `sfm check`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from sarsim_fm.models import chronos2, naive, patchtst
from sarsim_fm.runtime import Runtime


class Forecaster(Protocol):
    def predict_quantiles(
        self,
        contexts: list[np.ndarray],
        horizon: int,
        levels: list[float],
        batch_size: int,
        season_lengths: np.ndarray,
    ) -> np.ndarray:
        """Return quantile forecasts with shape (n_series, horizon, len(levels))."""
        ...


LOADERS: dict[str, Callable[..., Forecaster]] = {
    "chronos2": chronos2.load,
    "patchtst": patchtst.load,
    "seasonal_naive": naive.load,
}

TRAINERS: dict[str, Callable] = {
    "chronos2": chronos2.train,
    "patchtst": patchtst.train,
}

TIMERS: dict[str, Callable] = {
    "chronos2": chronos2.timed_steps,
    "patchtst": patchtst.timed_steps,
}


def load_forecaster(spec: str, runtime: Runtime, context_length: int) -> Forecaster:
    kind, _, path = spec.partition(":")
    if kind not in LOADERS:
        raise ValueError(f"Unknown model type {kind!r}; known: {sorted(LOADERS)}")
    return LOADERS[kind](path or None, runtime, context_length)


def get_trainer(kind: str) -> Callable:
    if kind not in TRAINERS:
        raise ValueError(f"Model type {kind!r} is not trainable; trainable: {sorted(TRAINERS)}")
    return TRAINERS[kind]


def get_timer(kind: str) -> Callable:
    if kind not in TIMERS:
        raise ValueError(f"`sfm check` timing does not support {kind!r}; supported: {sorted(TIMERS)}")
    return TIMERS[kind]
