"""One-to-one event matching, point average precision, and UCR localization."""

from __future__ import annotations

import numpy as np

from sarsim_fm.anomaly.scores import finite_array


def event_metrics(events, labels):
    """At most one detection per label is a TP; duplicate detections are FPs.

    Chronological detections greedily match the earliest-ending unmatched
    overlapping label. Delay is the first overlapping timestamp minus label start.
    """
    labels = sorted(labels, key=lambda x: x[1])
    matched, delays = set(), []
    for event in sorted(events, key=lambda e: e.start):
        for i, (start, end) in enumerate(labels):
            if i not in matched and event.start < end and start < event.end:
                matched.add(i)
                delays.append(max(event.start, start) - start)
                break
    tp, n, m = len(matched), len(events), len(labels)
    precision = tp / n if n else 0.0
    recall = tp / m if m else 0.0
    return {"event_tp": tp, "event_fp": n - tp, "event_fn": m - tp,
            "event_precision": precision, "event_recall": recall,
            "event_f1": 2 * tp / (n + m) if n + m else 0.0,
            "detection_delay": float(np.mean(delays)) if delays else None,
            "detection_delays": delays}


def point_auprc(labels, scores):
    """Average precision (step integral), aggregating ties; None without positives."""
    y, s = np.asarray(labels), finite_array(scores, "scores")
    if y.dtype != bool or y.ndim != 1 or y.shape != s.shape:
        raise ValueError("Boolean labels and scores must have matching one-dimensional shapes")
    if not y.any():
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * tp / (ends + 1)))


def top_one_time(events, time, scores):
    """Center of highest peak-score event; otherwise highest raw score timestamp."""
    if events:
        best = max(events, key=lambda e: e.peak_score)
        return (best.start + best.end - 1) // 2
    s = finite_array(scores, "scores")
    if s.ndim != 1 or not len(s) or np.shape(time) != s.shape:
        raise ValueError("time and scores must be matching nonempty vectors")
    return int(time[int(np.argmax(s))])


def ucr_hit(timestamp, start, end, tolerance=100):
    """Archive tolerance: max(100, anomaly length) outside each half-open boundary."""
    if not 0 <= start < end or tolerance < 0:
        raise ValueError("invalid interval or tolerance")
    slack = max(tolerance, end - start)
    # Archive slides use strict inequalities in original one-based inclusive coordinates.
    return bool(start + 1 - slack < timestamp + 1 < end + slack)


def paired_family_bootstrap(left, right, families, seed=42, n_bootstrap=2000):
    """Resample mnemonic families; retain all paired files in a sampled family."""
    a, b = finite_array(left, "left"), finite_array(right, "right")
    f = np.asarray(families)
    if a.ndim != 1 or a.shape != b.shape or a.shape != f.shape or not len(a):
        raise ValueError("paired outcomes and families must have matching nonempty shapes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    groups = [np.flatnonzero(f == key) for key in np.unique(f)]
    delta, rng = a - b, np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        estimates.append(float(delta[indices].mean()))
    return {"mean_difference": float(delta.mean()), "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "n_families": len(groups), "seed": seed, "n_bootstrap": n_bootstrap}
