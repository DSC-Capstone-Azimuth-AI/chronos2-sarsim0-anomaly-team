"""Threshold crossings grouped into half-open events."""

from dataclasses import dataclass

import numpy as np

from sarsim_fm.anomaly.scores import finite_array


@dataclass(frozen=True)
class DetectedEvent:
    start: int
    end: int
    peak_time: int
    peak_score: float


def build_events(scores, alerts, min_run=1, merge_gap=0, release_threshold_ratio=1.0):
    """Group crossings; filter short entry runs, extend with hysteresis, then merge.

    With hysteresis (<1), scores must be normalized to the point's entry
    threshold (entry=1). The pipeline always passes normalized scores. At 1,
    only the supplied alert mask determines membership. A gap of k means k
    unalerted samples. Peak ties resolve to the earliest timestamp.
    """
    s, a = finite_array(scores, "scores"), np.asarray(alerts)
    if s.ndim != 1 or a.shape != s.shape or a.dtype != bool:
        raise ValueError("scores and Boolean alerts must be matching one-dimensional arrays")
    if not isinstance(min_run, int) or min_run < 1 or not isinstance(merge_gap, int) or merge_gap < 0:
        raise ValueError("min_run must be a positive integer and merge_gap a nonnegative integer")
    if not np.isfinite(release_threshold_ratio) or not 0 < release_threshold_ratio <= 1:
        raise ValueError("release_threshold_ratio must be within (0, 1]")
    if release_threshold_ratio < 1 and np.any(a != (s > 1)):
        raise ValueError("hysteresis requires scores normalized to entry threshold 1")
    edges = np.diff(np.r_[False, a, False].astype(int))
    intervals = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if end - start < min_run:
            continue
        if release_threshold_ratio < 1:
            while end < len(s) and s[end] > release_threshold_ratio:
                end += 1
        if intervals and start - intervals[-1][1] <= merge_gap:
            intervals[-1][1] = max(intervals[-1][1], int(end))
        else:
            intervals.append([int(start), int(end)])
    result = []
    for start, end in intervals:
        peak = start + int(np.argmax(s[start:end]))
        result.append(DetectedEvent(start, end, peak, float(s[peak])))
    return result
