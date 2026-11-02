import numpy as np
import pandas as pd
import pytest

from sarsim_fm.anomaly.baselines import MedianMAD
from sarsim_fm.anomaly.pipeline import _resolved, run_detection
from sarsim_fm.anomaly.selection import select_settings


def test_selection_only_uses_dev_and_freezes_settings(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=1200)
    x[1000:1010] += 50
    path = tmp_path / "sim-anomaly-dev.parquet"
    pd.DataFrame([{"item_id": "dev-a", "series": x, "season_length": 12,
                   "clean_prefix_length": 800, "anomaly_start": 1000,
                   "anomaly_end": 1010}]).to_parquet(path)
    cfg = {"name": "dev", "datasets": [str(path)], "cache_dir": str(tmp_path / "cache")}
    run_detection(cfg, "rolling_mad", tmp_path / "dev", forecaster=MedianMAD())
    frozen = tmp_path / "selection.json"
    result = select_settings(tmp_path / "dev", frozen)
    assert len(result["candidates"]) == 162
    locked = _resolved({"name": "test", "locked": True, "selection_file": str(frozen)})
    assert locked["detector"] == result["detector"]
    with pytest.raises(FileExistsError, match="freezes"):
        select_settings(tmp_path / "dev", frozen)
    # Selection must reject any mixed dev/test run even if scores are present.
    (tmp_path / "dev" / "dataset-manifest.json").write_text('[{"name": "sim-anomaly-test"}]')
    with pytest.raises(ValueError, match="sim-anomaly-dev"):
        select_settings(tmp_path / "dev", tmp_path / "invalid.json")
