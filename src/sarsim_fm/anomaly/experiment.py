"""Ordered, resumable evaluation; training is deliberately a separate decision."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.pipeline import _datasets, _resolved, _write, collect_results, run_detection
from sarsim_fm.anomaly.provenance import ROOT, environment, file_hash, object_hash
from sarsim_fm.anomaly.selection import select_settings
from sarsim_fm.config import load_config


def compare_pilot(cfg, model_spec, out, device=None):
    """Same fixed dev items, both modes; save runtimes and all three score differences."""
    out = Path(out)
    summaries = {}
    for mode in ["rolling", "block"]:
        current = {**cfg, "name": f"pilot-{mode}",
                   "detector": {**cfg.get("detector", {}), "mode": mode}}
        path = out / mode
        summaries[mode] = run_detection(current, model_spec, path, device, overwrite=True)
    differences = {}
    for path in sorted((out / "rolling" / "scores").glob("*.parquet")):
        a, b = pd.read_parquet(path), pd.read_parquet(out / "block" / "scores" / path.name)
        if not np.array_equal(a.t, b.t):
            raise ValueError("pilot time alignment differs")
        differences[path.stem] = {key: {"mean_abs_difference": float(np.abs(a[key] - b[key]).mean()),
                                       "max_abs_difference": float(np.abs(a[key] - b[key]).max())}
                                  for key in ["interval_exceedance", "robust_residual", "quantile_tail"]}
    result = {"model": model_spec, "summaries": summaries, "score_differences": differences,
              "reference_mode": "rolling", "block_stride": cfg.get("detector", {}).get("prediction_length", 18)}
    _write(out / "comparison.json", result)
    return result


def validate_completed_run(path, config, model):
    """Validate saved results against current data, independent of checkout location."""
    path = Path(path)
    resolved = _resolved(config)
    saved = json.loads((path / "run-config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if (saved.get("status") != "completed" or saved["model"] != model
            or summary["config_sha256"] != object_hash(resolved)):
        raise ValueError(f"{path}: completed run differs from requested settings")
    previous = json.loads((path / "dataset-manifest.json").read_text())
    current = [manifest for manifest, _ in _datasets(resolved)]
    for manifest in current:
        if manifest.get("format") == "ucr":
            for entry in manifest["series"]:
                if file_hash(entry["source_path"]) != entry["sha256"]:
                    raise ValueError(f"{path}: staged UCR source changed")

    def identity(manifest):
        if manifest.get("format") == "ucr":
            # Restaging changes source paths, not the archive's data or annotations.
            entries = [{k: v for k, v in entry.items() if k != "source_path"}
                       for entry in manifest["series"]]
            return manifest["name"], object_hash(entries)
        return manifest["name"], manifest["sha256"]

    if sorted(map(identity, previous)) != sorted(map(identity, current)):
        raise ValueError(f"{path}: source changed since completed run")
    if ":" in model and Path(model.partition(":")[2]).is_dir():
        from sarsim_fm.anomaly.provenance import checkpoint_identity

        if checkpoint_identity(model)["sha256"] != saved["checkpoint"]["sha256"]:
            raise ValueError(f"{path}: checkpoint changed since completed run")


def run_experiment(device=None, include_ucr=True, external_only=False, checkpoint=None,
                   pretrained_checkpoint=None):
    """Resume matching completed runs; refuse silent changes to locked settings.

    Generate splits with make-anomaly-sim and stage UCR before invoking. Run on a
    GPU for the complete rolling Chronos archive evaluation.
    """
    root = ROOT / "outputs" / "anomaly"
    if external_only and not include_ucr:
        raise ValueError("external_only cannot be combined with synthetic_only")
    optional = Path(checkpoint) if checkpoint else ROOT / "outputs" / "chronos2-ft-500" / "checkpoint-final"
    if checkpoint and not optional.is_dir():
        raise FileNotFoundError(f"optional checkpoint not found: {optional}")
    models = ["rolling_mad", "seasonal_naive", "chronos2:amazon/chronos-2"]
    if pretrained_checkpoint:
        if not Path(pretrained_checkpoint).is_dir():
            raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained_checkpoint}")
        models[2] = f"chronos2:{pretrained_checkpoint}"
    if optional.exists():
        models.append(f"chronos2:{optional}")
    state = {"started": time.time(), "runs": [], "optional_checkpoint_available": optional.exists(),
             "scope": "external" if external_only else "all" if include_ucr else "synthetic",
             "status": "running"}

    def execute(config, model, path):
        path = root / path
        state["current_run"] = str(path)
        _write(root / "experiment-state.json", state)
        if (path / "summary.json").exists():
            validate_completed_run(path, config, model)
            print(f"Reusing completed run {path}", flush=True)
        else:
            run_detection(config, model, path, device)
        state["runs"].append(str(path))
        _write(root / "experiment-state.json", state)

    selection_path = root / "synthetic-selection.json"
    if external_only:
        if not selection_path.exists():
            raise FileNotFoundError("external evaluation requires the transferred frozen selection")
    else:
        dev = load_config(ROOT / "configs" / "anomaly" / "synthetic-chronos2.yaml")
        for model in models:
            execute(dev, model, f"development-{models.index(model)}")
        if not selection_path.exists():
            select_settings(root / "development-2", selection_path)
        locked = load_config(ROOT / "configs" / "anomaly" / "synthetic-locked.yaml")
        for model in models:
            execute(locked, model, f"locked-synthetic-{models.index(model)}")
    if include_ucr:
        ucr = load_config(ROOT / "configs" / "anomaly" / "ucr-chronos2.yaml")
        smoke = {**ucr, "subset_ids": [4, 5, 173]}
        for model in models:
            execute(smoke, model, f"ucr-smoke-{models.index(model)}")
        for model in models:
            execute(ucr, model, f"ucr-full-{models.index(model)}")
    state.update(status="completed", runtime_seconds=time.time() - state["started"])
    _write(root / "experiment-state.json", state)
    _write(root / "comparison.json", collect_results(root))
    return state


def record_forecasting_guardrail():
    """Append the available forecasting pilot without replacing the initial checkout record."""
    path = ROOT / "outputs" / "anomaly-baseline" / "baseline.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    results = {}
    for model in ["seasonal_naive", "chronos2_amazon_chronos-2"]:
        summary = ROOT / "outputs" / "baselines" / model / "eval" / "summary.json"
        if summary.exists():
            results[model] = json.loads(summary.read_text())
    datasets = {}
    for name in ["sim_iid", "sim_ood_season"]:
        source = ROOT / "data" / "eval" / f"{name}.parquet"
        manifest = source.with_suffix(".manifest.json")
        if source.exists():
            datasets[name] = {"sha256": file_hash(source), "manifest": json.loads(manifest.read_text())}
    record["followup_synthetic_guardrail"] = {
        "environment": environment(), "datasets": datasets, "forecasting_metrics": results,
        "commands": ["uv run sfm make-sim --set defaults.n_series=20",
                     "uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2"],
        "limitation": "20-series synthetic pilot only; M4 and the optional fine-tuned checkpoint are unavailable"}
    _write(path, record)
    return record
