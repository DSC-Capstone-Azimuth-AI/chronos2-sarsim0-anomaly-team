"""Select detector settings exclusively on saved synthetic-development forecasts."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.calibration import TARGET_FPRS, fit_thresholds
from sarsim_fm.anomaly.metrics import event_metrics
from sarsim_fm.anomaly.pipeline import SCORES, _events, _write
from sarsim_fm.anomaly.provenance import file_hash, object_hash


def select_settings(run_dir: Path, out: Path):
    """Freeze the best event F1 under a held-out normal false-alarm constraint.

    Every candidate is evaluated on the dev suffix. Normal-test and UCR data are
    not loaded. Ties favor lower normal alert volume then simpler event grouping.
    """
    run = json.loads((run_dir / "run-config.json").read_text())
    if run.get("status") != "completed":
        raise ValueError("selection requires a completed development run")
    manifest = json.loads((run_dir / "dataset-manifest.json").read_text())
    if len(manifest) != 1 or manifest[0]["name"] != "sim-anomaly-dev":
        raise ValueError("selection requires exactly the sim-anomaly-dev split")
    frames = [pd.read_parquet(path) for path in sorted((run_dir / "scores").glob("*.parquet"))]
    if not frames or any("label" not in f for f in frames):
        raise ValueError("selection requires a completed development run")
    candidates = []
    for score, fpr in itertools.product(SCORES, TARGET_FPRS):
        s = np.concatenate([f.loc[f.calibration, score] for f in frames])
        h = np.concatenate([f.loc[f.calibration, "horizon"] for f in frames])
        thresholds = fit_thresholds(s, h, fpr)
        for min_run, merge_gap, ratio in itertools.product([1, 2, 3], [0, 2, 6], [1.0, 0.75]):
            counts = {"event_tp": 0, "event_fp": 0, "event_fn": 0}
            normal, false = 0, 0
            event_cfg = {"min_run": min_run, "merge_gap": merge_gap, "release_threshold_ratio": ratio}
            for frame in frames:
                test = frame.loc[frame.evaluation]
                alerts, events = _events(test[score].to_numpy(), test.horizon.to_numpy(),
                                         thresholds, event_cfg, int(test.t.iloc[0]))
                times = test.t.to_numpy()[test.label.to_numpy()]
                labels = [(int(times[0]), int(times[-1]) + 1)] if len(times) else []
                m = event_metrics(events, labels)
                for key in counts:
                    counts[key] += m[key]
                normal += int((~test.label).sum())
                false += int(alerts[~test.label.to_numpy()].sum())
            tp, fp, fn = (counts[k] for k in ["event_tp", "event_fp", "event_fn"])
            candidates.append({"score": score, "target_fpr": fpr, "events": event_cfg,
                               "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                               "normal_fpr": false / normal, **counts})
    eligible = [c for c in candidates if c["normal_fpr"] <= 2 * c["target_fpr"]]
    if not eligible:
        raise ValueError("no dev candidate meets <=2x target held-out normal FPR; inspect dev diagnostics")
    best = max(eligible, key=lambda c: (c["event_f1"], -c["normal_fpr"],
                                      -c["events"]["merge_gap"], -c["events"]["min_run"]))
    detector = {**run["config"]["detector"], **{k: best[k] for k in ["score", "target_fpr", "events"]}}
    selection = {"selection_split": "sim-anomaly-dev", "detector": detector,
                 "settings_sha256": object_hash(detector), "source_run": str(run_dir.resolve()),
                 "source_run_config_sha256": file_hash(run_dir / "run-config.json"),
                 "source_manifest_sha256": file_hash(run_dir / "dataset-manifest.json"),
                 "selection_model": run["model"], "objective": "max event F1 subject to dev normal FPR <= 2x target",
                 "chosen": best, "candidates": candidates}
    if out.exists():
        raise FileExistsError(f"{out} already freezes settings; choose a new experiment path")
    _write(out, selection)
    print(json.dumps({k: v for k, v in selection.items() if k != "candidates"}, indent=2))
    return selection
