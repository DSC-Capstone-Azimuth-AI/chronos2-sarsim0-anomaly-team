import numpy as np
import pytest

from sarsim_fm.config import load_config
from sarsim_fm.metrics import WQL_LEVELS, coverage, mase, scaled_quantile_loss, smape
from sarsim_fm.models.naive import SeasonalNaive
from sarsim_fm.runtime import resolve_runtime

rng = np.random.default_rng(0)
history = rng.normal(size=60).cumsum().astype(np.float32)
actual = rng.normal(size=18).astype(np.float32)
point = rng.normal(size=18).astype(np.float32)


def test_mase_matches_original_formula():
    scale = np.mean(np.abs(history[12:] - history[:-12]))
    assert mase(actual, point, history, 12) == pytest.approx(np.mean(np.abs(actual - point)) / scale)


def test_mase_nan_for_constant_history():
    assert np.isnan(mase(actual, point, np.ones(30, np.float32), 1))


def test_sql_equals_mase_for_point_forecast():
    levels = np.asarray(WQL_LEVELS)
    quantiles = np.repeat(point[:, None], len(levels), axis=1)
    assert scaled_quantile_loss(actual, quantiles, levels, history, 1) == pytest.approx(
        mase(actual, point, history, 1)
    )


def test_smape_and_coverage_basics():
    assert smape(actual, actual) == 0.0
    assert coverage(actual, actual - 1, actual + 1) == 1.0


def test_seasonal_naive_repeats_last_season():
    out = SeasonalNaive(None).predict_quantiles(
        [np.arange(24, dtype=np.float32)], 18, [0.1, 0.5, 0.9], 1, np.array([12])
    )
    assert out.shape == (1, 18, 3)
    np.testing.assert_array_equal(out[0, :, 1], np.r_[np.arange(12, 24), np.arange(12, 18)])


def test_config_inherit_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train: {max_steps: 10, learning_rate: 1.0e-5}\ndata: {seed: 1}\n")
    (tmp_path / "run.yaml").write_text("inherit: base.yaml\ntrain: {max_steps: 20}\n")
    cfg = load_config(tmp_path / "run.yaml", ["data.seed=7", "eval.datasets=[a, b]"])
    assert cfg["train"] == {"max_steps": 20, "learning_rate": 1e-5}
    assert cfg["data"]["seed"] == 7
    assert cfg["eval"]["datasets"] == ["a", "b"]


def test_cpu_runtime_is_fp32():
    runtime = resolve_runtime("cpu", "auto")
    assert runtime.device.type == "cpu" and not runtime.bf16 and not runtime.tf32
