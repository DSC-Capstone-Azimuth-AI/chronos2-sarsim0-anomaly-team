import numpy as np
import pytest

from sarsim_fm.anomaly.events import DetectedEvent
from sarsim_fm.anomaly.metrics import (
    event_metrics,
    paired_family_bootstrap,
    point_auprc,
    top_one_time,
    ucr_hit,
)


def test_duplicate_detections_and_delay():
    events = [DetectedEvent(12, 14, 13, 4), DetectedEvent(16, 18, 17, 8),
              DetectedEvent(30, 32, 30, 2)]
    m = event_metrics(events, [(10, 20)])
    assert (m["event_tp"], m["event_fp"], m["detection_delay"]) == (1, 2, 2)
    assert m["event_f1"] == 0.5
    assert top_one_time(events, np.arange(3), [1, 2, 3]) == 16
    assert top_one_time([], [10, 11, 12], [1, 3, 2]) == 11


def test_average_precision_ties_and_empty_positive():
    y = np.array([True, False, True, False])
    assert point_auprc(y, [4, 3, 2, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert point_auprc(y, np.ones(4)) == 0.5
    assert point_auprc(y & False, np.ones(4)) is None


def test_ucr_and_paired_bootstrap():
    assert not ucr_hit(0, 100, 101)
    assert ucr_hit(1, 100, 101)
    assert not ucr_hit(201, 100, 101)
    assert ucr_hit(500, 100, 301)
    result = paired_family_bootstrap([1, 1, 1], [0, 0, 0], ["a", "a", "b"], n_bootstrap=20)
    assert result["ci95"] == [1, 1] and result["n_families"] == 2
