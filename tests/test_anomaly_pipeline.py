import json

import numpy as np
import pandas as pd
import pytest

from sarsim_fm.anomaly.baselines import MedianMAD
from sarsim_fm.anomaly.pipeline import collect_results, run_detection


def dataset(tmp_path, name="sim-anomaly-dev"):
    rng = np.random.default_rng(3)
    x = rng.normal(size=700)
    x[550:560] += 20
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame([{"item_id": "a", "series": x.tolist(), "clean_prefix_length": 400,
                   "season_length": 12, "anomaly_start": 550, "anomaly_end": 560, "seed": 9}]).to_parquet(path)
    return {"name": name, "datasets": [str(path)], "cache_dir": str(tmp_path / "cache"),
            "detector": {"context_length": 32, "target_fpr": 0.02}}


def test_pipeline_artifacts_and_overwrite(tmp_path):
    cfg = dataset(tmp_path)
    out = tmp_path / "runs" / "test"
    result = run_detection(cfg, "rolling_mad", out, forecaster=MedianMAD())
    assert result["overall"]["event_tp"] == 1
    for name in ["run-config.json", "dataset-manifest.json", "summary.json", "events.parquet",
                 "forecasts/a.parquet", "forecasts/a.json", "scores/a.parquet"]:
        assert (out / name).exists()
    record = json.loads((out / "per-series.json").read_text())[0]
    assert record["threshold_model"]["calibration_counts"] == {"1": 200}
    assert json.loads((out / "run-config.json").read_text())["seeds"] == [9]
    assert len(collect_results(tmp_path / "runs")["runs"]) == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        run_detection(cfg, "rolling_mad", out, forecaster=MedianMAD())


def test_anomaly_values_cannot_change_calibration(tmp_path):
    cfg = dataset(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    run_detection(cfg, "rolling_mad", a, forecaster=MedianMAD())
    path = cfg["datasets"][0]
    frame = pd.read_parquet(path)
    x = np.array(frame.at[0, "series"])
    x[400:] *= 1e5
    frame.at[0, "series"] = x
    frame.to_parquet(path)
    run_detection(cfg, "rolling_mad", b, forecaster=MedianMAD())
    ra, rb = [json.loads((p / "per-series.json").read_text())[0] for p in [a, b]]
    assert ra["threshold_model"] == rb["threshold_model"]
    assert ra["residual_scale"] == rb["residual_scale"]


def test_locked_requires_selection(tmp_path):
    cfg = dataset(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        run_detection({**cfg, "locked": True}, "rolling_mad", tmp_path / "run")


def test_run_writer_lock(tmp_path):
    import fcntl

    cfg = dataset(tmp_path)
    out = tmp_path / "run"
    out.mkdir()
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            run_detection(cfg, "rolling_mad", out, forecaster=MedianMAD())


def test_ucr_uses_only_clean_prefix_partitions(tmp_path):
    from sarsim_fm.anomaly.provenance import file_hash

    x = np.random.default_rng(0).normal(size=1300)
    x[1100:1110] += 15
    source = tmp_path / "001_UCR_Anomaly_demo_1000_1101_1110.txt"
    source.write_text("\n".join(map(str, x)))
    manifest = {"name": "ucr-anomaly", "format": "ucr", "series": [{
        "dataset_id": 1, "name": "demo", "train_end": 1000, "anomaly_start": 1101,
        "anomaly_end": 1110, "source_path": str(source), "sha256": file_hash(source), "family": "demo"}]}
    staged = tmp_path / "ucr.json"
    staged.write_text(json.dumps(manifest))
    cfg = {"name": "smoke", "datasets": [str(staged)], "cache_dir": str(tmp_path / "cache"),
           "detector": {"target_fpr": 0.005}}
    out = tmp_path / "ucr"
    run_detection(cfg, "rolling_mad", out, forecaster=MedianMAD())
    scores = pd.read_parquet(out / "scores" / "ucr-001.parquet")
    np.testing.assert_array_equal(scores.loc[scores.calibration, "t"], np.arange(800, 1000))
    np.testing.assert_array_equal(scores.loc[scores.scale_fit, "t"], np.arange(600, 800))
    assert scores.loc[scores.label, "t"].tolist() == list(range(1100, 1110))
    row = json.loads((out / "per-series.json").read_text())[0]
    assert row["threshold_model"]["calibration_counts"] == {"1": 200}
