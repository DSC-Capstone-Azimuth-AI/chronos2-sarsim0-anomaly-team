import numpy as np
import pytest

from sarsim_fm.anomaly.scores import (
    fit_residual_scale,
    interval_exceedance_score,
    quantile_tail_score,
    rearrange_quantiles,
    robust_residual_score,
)
from sarsim_fm.evaluate import LEVELS


def test_rearrangement_and_interval():
    p = np.array(LEVELS)
    q = np.tile(np.linspace(-2, 2, len(p))[::-1], (5, 1))
    original = q.copy()
    assert (np.diff(rearrange_quantiles(q)) >= 0).all()
    np.testing.assert_array_equal(q, original)
    s = interval_exceedance_score(np.array([0, 2, 3, 4, -4]), q, p)
    np.testing.assert_allclose(s, [0, 0, 0.25, 0.5, 0.5])


def test_tail_monotone_both_tails_and_ties():
    p = np.array(LEVELS)
    y = np.array([0, 1, 2, 4, 10])
    q = np.tile(np.linspace(-2, 2, len(p)), (len(y), 1))
    for sign in [-1, 1]:
        assert (np.diff(quantile_tail_score(sign * y, q, p)) > 0).all()
    assert np.isfinite(quantile_tail_score(y, q * 0, p)).all()


@pytest.mark.parametrize("noise", [0, 1e-12])
def test_constant_residual_scale(noise):
    y = np.ones(10) + noise * np.arange(10)
    scale = fit_residual_scale(y, np.ones(10))
    assert np.isfinite(robust_residual_score(y, np.zeros(10), scale)).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite(bad):
    with pytest.raises(ValueError, match="finite"):
        rearrange_quantiles([[0, bad]])
    with pytest.raises(ValueError, match="finite"):
        fit_residual_scale([bad], [0])
    with pytest.raises(ValueError, match="finite"):
        robust_residual_score([bad], [0], 1)


def test_shapes():
    with pytest.raises(ValueError, match="shape"):
        interval_exceedance_score(np.zeros(2), np.zeros((3, 13)), LEVELS)
    with pytest.raises(ValueError, match="levels"):
        quantile_tail_score(np.zeros(3), np.zeros((3, 13)), [0.1, 0.9])
    with pytest.raises(ValueError, match="shapes"):
        robust_residual_score([1], [1, 2], 1)
