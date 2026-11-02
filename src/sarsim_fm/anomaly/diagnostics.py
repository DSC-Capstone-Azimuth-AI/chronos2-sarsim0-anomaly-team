"""Post-lock failure plots and mnemonic-family summaries. Never alters settings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_diagnostics(run_dir: Path, max_plots=20):
    """Plot failed localization / missed events after a completed locked evaluation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run-config.json").read_text())
    if not run["config"].get("locked") or run.get("status") != "completed":
        raise ValueError("diagnostics require a completed locked evaluation")
    records = json.loads((run_dir / "per-series.json").read_text())
    events = pd.read_parquet(run_dir / "events.parquet")
    score_name = run["config"]["detector"]["score"]
    failures = [r for r in records if r.get("top_one_hit") is False or r["event_fn"] > 0]
    directory = run_dir / "diagnostics"
    directory.mkdir(exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    (directory / "families.json").write_text(json.dumps(summary["families"], indent=2))
    for record in failures[:max_plots]:
        item = record["item_id"]
        frame = pd.read_parquet(run_dir / "scores" / f"{item}.parquet")
        forecast = pd.read_parquet(run_dir / "forecasts" / f"{item}.parquet")
        fig, (signal, scores) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        # Downsample display only, retaining true labels/events and raw output artifacts.
        indices = np.arange(0, len(frame), max(1, len(frame) // 10000))
        visible = forecast.iloc[indices]
        signal.plot(visible.t, visible.actual, lw=0.6, label="Observed")
        signal.fill_between(visible.t, visible.q01, visible.q99, alpha=0.2, label="98% forecast interval")
        label = frame.loc[frame.label, "t"]
        if len(label):
            for axis in [signal, scores]:
                axis.axvspan(label.iloc[0], label.iloc[-1] + 1, alpha=0.2, color="red", label="Labeled anomaly")
        scores.plot(frame.t.iloc[indices], frame[score_name].iloc[indices], lw=0.7, label=score_name)
        thresholds = record["threshold_model"]["thresholds"]
        threshold = np.array([thresholds[str(int(h))] for h in frame.horizon])
        scores.plot(frame.t.iloc[indices], threshold[indices], ls="--", label="Clean calibrated threshold")
        for event in events.loc[events.item_id == item].itertuples():
            scores.axvspan(event.start, event.end, alpha=0.1, color="orange")
        if record.get("top_one_time") is not None:
            scores.axvline(record["top_one_time"], color="black", ls=":", label="Top-one location")
        signal.set(title=f"{item} · {run['model']}", ylabel="Signal value (source units)")
        scores.set(xlabel="Time (zero-based sample)", ylabel=f"{score_name} severity")
        signal.legend(loc="upper right")
        scores.legend(loc="upper right")
        fig.text(0.01, 0.01, f"Source: {run_dir.name}; full scored evaluation span; display downsampled only", fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(directory / f"{item}.png", dpi=140)
        plt.close(fig)
    return {"failures": len(failures), "plots": min(len(failures), max_plots), "path": str(directory)}
