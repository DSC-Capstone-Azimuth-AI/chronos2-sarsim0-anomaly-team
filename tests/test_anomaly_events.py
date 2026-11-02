import numpy as np
import pytest

from sarsim_fm.anomaly.events import build_events


def test_adjacent_merge_filter_peak():
    s = np.array([0, 2, 5, 0, 3, 4, 0, 2, 0])
    events = build_events(s, s > 1, min_run=2)
    assert [(e.start, e.end) for e in events] == [(1, 3), (4, 6)]
    assert (events[0].peak_time, events[0].peak_score) == (2, 5)
    assert [(e.start, e.end) for e in build_events(s, s > 1, min_run=2, merge_gap=1)] == [(1, 6)]
    assert build_events(np.zeros(2), np.zeros(2, bool)) == []
    assert build_events([], np.array([], bool)) == []


def test_hysteresis():
    s = np.array([0, 2, 0.8, 0.6, 0.3, 2])
    events = build_events(s, s > 1, release_threshold_ratio=0.5)
    assert [(e.start, e.end) for e in events] == [(1, 4), (5, 6)]
    with pytest.raises(ValueError, match="normalized"):
        build_events([4, 5], np.array([False, True]), release_threshold_ratio=0.5)
