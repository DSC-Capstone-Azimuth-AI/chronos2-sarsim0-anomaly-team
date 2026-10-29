import numpy as np
import pytest

from sarsim_fm.anomaly.calibration import (
    apply_thresholds,
    fit_thresholds,
    interval_coverage,
    normal_metrics,
)
from sarsim_fm.evaluate import LEVELS


def test_conformal_rank_and_monotonicity():
    s = np.arange(1000)
    models = [fit_thresholds(s, np.ones(1000), a) for a in [0.005, 0.01, 0.02]]
    assert [m.thresholds[1] for m in models] == [995, 990, 980]
    assert models[0].calibration_counts == {1: 1000}


def test_horizons_and_normal_metrics():
    model = fit_thresholds(np.r_[np.zeros(200), np.ones(200) * 10],
                           np.repeat([1, 2], 200), 0.01)
    alerts = apply_thresholds([1, 1, 11, 10], [1, 2, 2, 2], model)
    np.testing.assert_array_equal(alerts, [True, False, True, False])
    assert normal_metrics(alerts)["false_alerts_per_1000"] == 500
    assert normal_metrics(alerts)["realized_fpr"] == 0.5
    assert interval_coverage(np.zeros(2), np.zeros((2, 13)), LEVELS)["coverage_98"] == 1
    with pytest.raises(ValueError, match="uncalibrated"):
        apply_thresholds([1], [3], model)


def test_invalid_and_insufficient():
    with pytest.raises(ValueError, match="target_fpr"):
        fit_thresholds([1], [1], 0.5)
    with pytest.raises(ValueError, match="insufficient"):
        fit_thresholds(np.arange(10), np.ones(10), 0.005)
    assert fit_thresholds(np.arange(199), np.ones(199), 0.005).thresholds[1] == 198
