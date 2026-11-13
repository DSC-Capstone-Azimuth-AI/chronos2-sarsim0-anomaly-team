"""Model registry.

A forecaster spec is `<type>` or `<type>:<path-or-hub-id>`, e.g.
    seasonal_naive
    chronos2:amazon/chronos-2
    chronos2:outputs/chronos2-ft-cpu/checkpoint-final
    patchtst_fm                                   (= patchtst_fm:ibm-granite/granite-timeseries-patchtst-fm-r2)
    patchtst_fm:outputs/patchtst-fm-ft-5k/checkpoint-final

To add a model family, write `models/<name>.py` with `load(path, runtime, context_length)`
returning a `Forecaster`, plus `train(cfg, runtime, run_dir) -> checkpoint path` if trainable,
then register both below (and `timed_steps(cfg, runtime, steps)` for `sfm check`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from sarsim_fm.models import chronos2, median, naive, patchtst_fm
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
    "patchtst_fm": patchtst_fm.load,
    "seasonal_naive": naive.load,
    "rolling_median": median.load,
}

TRAINERS: dict[str, Callable] = {
    "chronos2": chronos2.train,
    "patchtst_fm": patchtst_fm.train,
}

TIMERS: dict[str, Callable] = {
    "chronos2": chronos2.timed_steps,
    "patchtst_fm": patchtst_fm.timed_steps,
}


def load_forecaster(spec: str, runtime: Runtime, context_length: int) -> Forecaster:
    kind, _, path = spec.partition(":")
    if kind not in LOADERS:
        raise ValueError(f"Unknown model type {kind!r}; known: {sorted(LOADERS)}")
    return LOADERS[kind](path or None, runtime, context_length)


def load_derconnect_forecaster(spec: str, runtime: Runtime, context_length: int) -> Forecaster:
    """Pin a Hub snapshot before loading; config private fields are not provenance.

    Transformers versions need not preserve `_commit_hash` on a loaded model.
    Resolve config.json to an immutable snapshot, download that exact revision,
    and load its local directory so the recorded identity describes actual weights.
    Legacy benchmark loading retains its existing behavior.
    """
    kind, _, location = spec.partition(":")
    defaults = {"chronos2": "amazon/chronos-2",
                "patchtst_fm": "ibm-granite/granite-timeseries-patchtst-fm-r2"}
    if kind not in defaults:
        return load_forecaster(spec, runtime, context_length)
    location = location or defaults[kind]
    if Path(location).is_dir():
        return load_forecaster(spec, runtime, context_length)
    from huggingface_hub import hf_hub_download, snapshot_download

    config_path = Path(hf_hub_download(location, "config.json"))
    snapshot = config_path.parent
    if snapshot.parent.name != "snapshots" or not snapshot.name:
        raise ValueError("could not resolve an immutable Hub snapshot from config.json")
    revision = snapshot.name
    resolved = snapshot_download(location, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.bin"])
    if Path(resolved) != snapshot:
        raise ValueError("downloaded snapshot differs from resolved config revision")
    forecaster = load_forecaster(f"{kind}:{resolved}", runtime, context_length)
    forecaster.resolved_hub_checkpoint = {
        "repository": location, "revision": revision, "path": str(snapshot),
    }
    return forecaster


def get_trainer(kind: str) -> Callable:
    if kind not in TRAINERS:
        raise ValueError(f"Model type {kind!r} is not trainable; trainable: {sorted(TRAINERS)}")
    return TRAINERS[kind]


def get_timer(kind: str) -> Callable:
    if kind not in TIMERS:
        raise ValueError(f"`sfm check` timing does not support {kind!r}; supported: {sorted(TIMERS)}")
    return TIMERS[kind]
