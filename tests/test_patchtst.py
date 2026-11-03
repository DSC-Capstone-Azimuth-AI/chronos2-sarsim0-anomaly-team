import numpy as np
import pytest
import torch

from sarsim_fm.anomaly.forecast import rolling_forecast
from sarsim_fm.models.patchtst import (
    PatchTSTForecaster,
    PatchTSTStream,
    build_model,
    fill_unobserved,
    model_inputs,
)

LEVELS = [0.01, 0.1, 0.5, 0.9, 0.99]
TINY = {
    "model": {"init": "scratch", "architecture": {"d_model": 16, "num_attention_heads": 2,
                                                  "num_hidden_layers": 1, "ffn_dim": 32}},
    "data": {"context_length": 64, "prediction_length": 6, "seed": 0},
}


def tiny_forecaster(context_length=64):
    return PatchTSTForecaster(build_model(TINY).eval(), torch.device("cpu"), context_length)


def test_fill_unobserved_carries_values_over_gaps():
    nan = float("nan")
    values = torch.tensor([[nan, nan, 3.0, nan, 5.0], [nan] * 5])
    observed = torch.isfinite(values)
    filled = fill_unobserved(torch.nan_to_num(values), observed)
    assert filled.tolist() == [[3.0, 3.0, 3.0, 3.0, 5.0], [0.0] * 5]


def test_model_inputs_right_align_and_mask_padding():
    values, observed = model_inputs([np.arange(1, 4, dtype=np.float32), np.arange(10.0)], 5)
    assert values[0].tolist() == [1, 1, 1, 2, 3]
    assert observed[0].tolist() == [False, False, True, True, True]
    assert values[1].tolist() == [5, 6, 7, 8, 9] and observed[1].all()


def test_forecaster_handles_short_and_gappy_histories():
    rng = np.random.default_rng(0)
    contexts = [rng.normal(size=200), rng.normal(size=5), np.r_[np.nan, rng.normal(size=30), np.nan]]
    q = tiny_forecaster().predict_quantiles(contexts, 4, LEVELS, 2, np.ones(3))
    assert q.shape == (3, 4, len(LEVELS)) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) > 0).all()


def test_forecaster_rejects_horizon_beyond_training():
    with pytest.raises(ValueError, match="prediction_length"):
        tiny_forecaster().predict_quantiles([np.ones(10)], 7, LEVELS, 1, np.ones(1))


def test_forecaster_uses_only_requested_context():
    forecaster = tiny_forecaster(context_length=8)
    history = np.random.default_rng(1).normal(size=50)
    changed = history.copy()
    changed[:-8] += 100.0
    a, b = (forecaster.predict_quantiles([h], 6, LEVELS, 1, np.ones(1)) for h in (history, changed))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_rolling_anomaly_forecast_runs_on_patchtst():
    series = np.sin(np.arange(120) / 3.0)
    cache = rolling_forecast(tiny_forecaster(), "x", series, 100, 64, LEVELS, 8)
    assert cache.quantiles.shape == (20, len(LEVELS)) and (cache.forecast_horizon == 1).all()


def test_stream_masks_a_truncated_prefix():
    data_cfg = {"context_length": 64, "prediction_length": 6, "batch_size": 16, "seed": 0,
                "short_context": {"prob": 1.0, "min_length": 16}}
    batch = next(iter(PatchTSTStream(data_cfg)))
    mask = batch["past_observed_mask"][..., 0]
    assert batch["past_values"].shape == (16, 64, 1) and batch["future_values"].shape == (16, 6, 1)
    # Observed positions form one suffix per row, at least min_length long.
    assert (mask.int().diff(dim=1) >= 0).all() and (mask.sum(dim=1) >= 16).all()
    assert (mask.sum(dim=1) < 64).any()
