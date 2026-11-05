import numpy as np
import pytest
import torch

from sarsim_fm.anomaly.forecast import rolling_forecast
from sarsim_fm.models.patchtst_fm import (
    PatchTSTFMForecaster,
    PatchTSTFMForSarSim0,
    forecast_slots,
    forecast_window,
)
from sarsim_fm.models.vendor.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPrediction

LEVELS = [0.01, 0.1, 0.5, 0.9, 0.99]
# r2's layout (strided conformer, Hamming windows, 99 quantiles) at toy size; 32 forecast slots.
TINY = {"context_length": 256, "prediction_length": 16, "d_patch": 16, "patch_stride": 8,
        "patch_loss_windowing": "hamming", "d_model": 32, "n_head": 2, "n_layer": 2,
        "block_type": "conformer", "conv_kernel_size": 3, "prehead_norm": True, "pretrain_mask_cont": 2}


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PatchTSTFMForSarSim0(PatchTSTFMConfig(**TINY)).eval()


def decode_window(model, context, horizon):
    """Quantiles from a `forecast_window` pass, post-processed the way IBM's fast forecast path is."""
    backbone, config = model.backbone, model.config
    window = forecast_window(config, context)
    out = backbone(**window, return_loss=False)
    q = backbone._overlap_add(out.quantile_outputs, original_length=config.context_length,
                              stride=config.patch_stride)
    q = backbone.norm_fn.inverse_transform(q.permute(0, 2, 1), mean=out.loc, std=out.scale)
    slots = forecast_slots(config, horizon)
    return q[:, :, -slots:][:, :, :horizon]


def test_window_layout():
    config = PatchTSTFMConfig(**TINY)
    context = torch.arange(1.0, 41.0).reshape(2, 20)
    context[0, 5] = float("nan")
    future = torch.full((2, 6), 99.0)
    w = forecast_window(config, context, future)
    slots = forecast_slots(config, 6)
    assert slots == 32 and w["context_length"] == 20 + slots
    assert w["inputs"].shape == (2, config.context_length)
    left = config.context_length - 20 - slots
    assert w["pad_mask"][:, :left].all() and not w["pad_mask"][:, left:].any()
    assert not w["pred_mask"][:, :-slots].any() and w["pred_mask"][:, -slots:].all()
    # The NaN is missing, the future fills the first 6 slots, and the other 26 slots have no target.
    assert w["miss_mask"][0, left + 5] and w["miss_mask"].sum(dim=1).tolist() == [27, 26]
    assert (w["inputs"][:, -slots:-slots + 6] == 99).all()


def test_window_cuts_history_to_fit_with_the_slots():
    config = PatchTSTFMConfig(**TINY)
    w = forecast_window(config, torch.randn(1, 1000))
    assert w["context_length"] == config.context_length and not w["pad_mask"].any()


def test_training_window_matches_ibm_inference():
    """Training sees exactly the input layout that `PatchTSTFMForPrediction.forward` builds."""
    model = tiny_model()
    context = torch.randn(3, 100).cumsum(dim=1)
    with torch.no_grad():
        ours = decode_window(model, context, horizon=10)
        ibm = model(past_values=context, prediction_length=10, return_dict=True).quantile_outputs[..., 0]
    torch.testing.assert_close(ours, ibm, rtol=1e-4, atol=1e-4)


def test_future_values_do_not_leak_into_predictions():
    model = tiny_model()
    context = torch.randn(2, 100)
    with torch.no_grad():
        a = model.backbone(**forecast_window(model.config, context, torch.zeros(2, 18))).quantile_outputs
        b = model.backbone(**forecast_window(model.config, context, torch.full((2, 18), 1e3))).quantile_outputs
    torch.testing.assert_close(a, b)


def test_loss_trains():
    model = tiny_model().train()
    t = torch.arange(118.0)
    series = torch.stack([torch.sin(t / 4) * 3 + i for i in range(4)])
    batch = {"past_values": series[:, :100], "future_values": series[:, 100:]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_checkpoint_loads_as_ibm_prediction_class(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    loaded = PatchTSTFMForPrediction.from_pretrained(tmp_path).eval()
    context = torch.randn(2, 80)
    with torch.no_grad():
        a = model(past_values=context, prediction_length=8, return_dict=True).quantile_outputs
        b = loaded(past_values=context, prediction_length=8, return_dict=True).quantile_outputs
    torch.testing.assert_close(a, b)


def test_forecaster_handles_short_gappy_and_mixed_length_histories():
    forecaster = PatchTSTFMForecaster(tiny_model(), torch.device("cpu"), context_length=200)
    contexts = [np.random.default_rng(0).normal(size=150), np.array([1.0, np.nan, 3.0, 4.0]), np.ones(1)]
    q = forecaster.predict_quantiles(contexts, horizon=12, levels=LEVELS, batch_size=2,
                                     season_lengths=np.ones(3))
    assert q.shape == (3, 12, len(LEVELS)) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= -1e-5).all()


def test_forecaster_returns_model_quantiles_on_the_grid():
    model = tiny_model()
    forecaster = PatchTSTFMForecaster(model, torch.device("cpu"), context_length=200)
    context = np.random.default_rng(1).normal(size=90).astype(np.float32)
    q = forecaster.predict_quantiles([context], horizon=5, levels=LEVELS, batch_size=1, season_lengths=[1])
    with torch.no_grad():
        ibm = model(past_values=torch.from_numpy(context)[None], prediction_length=5,
                    quantile_levels=LEVELS, return_dict=True).quantile_outputs[0, :, :, 0]
    np.testing.assert_allclose(q[0], ibm.numpy().T, rtol=1e-5, atol=1e-5)


def test_rolling_anomaly_forecast_runs_on_patchtst_fm():
    forecaster = PatchTSTFMForecaster(tiny_model(), torch.device("cpu"), context_length=64)
    series = np.sin(np.arange(120) / 5.0)
    cache = rolling_forecast(forecaster, "s", series, start=40, context_length=64,
                             levels=LEVELS, batch_size=16)
    assert cache.quantiles.shape == (80, len(LEVELS))


def test_pretrained_rejects_architecture_overrides():
    from sarsim_fm.models.patchtst_fm import build_model

    with pytest.raises(ValueError, match="scratch"):
        build_model({"init": "pretrained", "architecture": {"n_layer": 2}}, seed=0)


def test_batching_with_longer_series_does_not_change_a_forecast():
    forecaster = PatchTSTFMForecaster(tiny_model(), torch.device("cpu"), context_length=200)
    rng = np.random.default_rng(2)
    short, long = rng.normal(size=30), rng.normal(size=180)
    alone = forecaster.predict_quantiles([short], 8, LEVELS, 1, [1])
    batched = forecaster.predict_quantiles([long, short], 8, LEVELS, 2, [1, 1])
    np.testing.assert_allclose(batched[1], alone[0], rtol=1e-4, atol=1e-4)


def test_scratch_rebuilds_derived_config_fields(monkeypatch):
    from sarsim_fm.models import patchtst_fm

    monkeypatch.setattr(patchtst_fm.PatchTSTFMConfig, "from_pretrained",
                        classmethod(lambda cls, base: cls(**TINY)))
    model = patchtst_fm.build_model(
        {"init": "scratch", "architecture": {"context_length": 128, "n_layer": 1}}, seed=0
    )
    assert model.config.n_patch == (128 - 16) // 8 + 1 and len(model.backbone.blocks) == 1
    assert len(model.config.quantile_levels) == 99
