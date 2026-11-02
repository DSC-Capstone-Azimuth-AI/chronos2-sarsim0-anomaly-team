import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sarsim_fm.anomaly.baselines import MedianMAD
from sarsim_fm.anomaly.experiment import validate_completed_run
from sarsim_fm.anomaly.pipeline import run_detection


def test_completed_runs_can_move_to_a_new_checkout(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    monkeypatch.chdir(original)
    pd.DataFrame([{"item_id": "normal", "series": np.random.default_rng(0).normal(size=1000),
                   "clean_prefix_length": 600, "season_length": 12}]).to_parquet("data.parquet")
    cfg = {"name": "test", "datasets": ["data.parquet"], "cache_dir": "cache"}
    run_detection(cfg, "rolling_mad", original / "run", forecaster=MedianMAD())
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy2(original / "data.parquet", moved / "data.parquet")
    shutil.copytree(original / "run", moved / "run")
    # Saved absolute path is deliberately unusable on the new host.
    manifest = moved / "run" / "dataset-manifest.json"
    records = json.loads(manifest.read_text())
    records[0]["path"] = "/not/on/this/machine/data.parquet"
    manifest.write_text(json.dumps(records))
    monkeypatch.chdir(moved)
    validate_completed_run(moved / "run", cfg, "rolling_mad")
    pd.DataFrame([{"item_id": "normal", "series": np.ones(1000),
                   "clean_prefix_length": 600, "season_length": 12}]).to_parquet("data.parquet")
    with pytest.raises(ValueError, match="source changed"):
        validate_completed_run(moved / "run", cfg, "rolling_mad")
