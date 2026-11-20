import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/derconnect/forecast_quality.py"
spec = importlib.util.spec_from_file_location("forecast_quality", SCRIPT)
fq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fq)

N, PREFIX, SEASON = 24 * 28, 24 * 14, 24


def _stage(root: Path, meter: str) -> str:
    t = np.arange(N)
    series = 50 + 10 * np.sin(2 * np.pi * t / SEASON) + np.random.default_rng(1).normal(0, 1, N)
    filled = np.zeros(N, bool)
    filled[PREFIX + 5] = True  # one interpolated evaluation point, which must be excluded
    rel = f"data/derconnect-{meter}.parquet"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": meter, "series": [series.tolist()], "clean_prefix_length": PREFIX,
                  "season_length": SEASON}, index=[0]).to_parquet(path, index=False)
    pd.DataFrame({"time_local": pd.date_range("2026-01-05", periods=N, freq="1h"), "kw": series,
                  "filled": filled}).to_parquet(path.with_name(f"{path.stem}-times.parquet"))
    return rel


def _run(root: Path, name: str, model: str, datasets: list[str], noise: float) -> Path:
    run = root / name
    (run / "forecasts").mkdir(parents=True)
    rng = np.random.default_rng(2)
    for rel in datasets:
        row = pd.read_parquet(root / rel).iloc[0]
        y = np.asarray(row["series"])
        t = np.arange(SEASON, N)
        median = y[t] + rng.normal(0, noise, len(t))
        frame = pd.DataFrame({"item_id": row["item_id"], "t": t, "actual": y[t]})
        for col, level in zip(fq.QCOLS, fq.LEVELS):
            frame[col] = median + (level - 0.5) * 6 * max(noise, 1)
        frame["forecast_horizon"] = 1
        frame.to_parquet(run / "forecasts" / f"{row['item_id']}.parquet", index=False)
    (run / "run-config.json").write_text(json.dumps(
        {"status": "completed", "model": model, "config": {"datasets": datasets}}))
    return run


def test_ranks_models_and_scores_only_evaluation_rows(tmp_path):
    datasets = [_stage(tmp_path, m) for m in ["E1", "E2"]]
    _run(tmp_path, "good", "good", datasets, noise=0.5)
    _run(tmp_path, "bad", "bad", datasets, noise=5.0)

    points = {m: fq.load_run(tmp_path / m, tmp_path)[1] for m in ["good", "bad"]}
    good = points["good"]
    assert good.t.min() == PREFIX                       # no calibration rows
    assert PREFIX + 5 not in set(good[good.meter == "E1"].t)  # filled point dropped
    assert len(good) == 2 * (N - PREFIX - 1)

    overall = fq.summarize(pd.concat([f.assign(model=m) for m, f in points.items()]), ["model"])
    by = overall.set_index("model")
    assert by.loc["good", "sql"] < by.loc["bad", "sql"]
    assert by.loc["good", "mase"] < by.loc["bad", "mase"]

    comparison = fq.paired(points, "good", "week", n_boot=500, seed=0).iloc[0]
    assert comparison.model == "bad" and comparison.d_sql > 0 and comparison.d_sql_lo > 0
    assert comparison.n_units == 2 * 2                  # 2 meters x 2 evaluation weeks


def test_merges_runs_split_by_meter_and_rejects_overlap(tmp_path):
    e1, e2 = _stage(tmp_path, "E1"), _stage(tmp_path, "E2")
    _run(tmp_path, "m-E1", "m", [e1], noise=1.0)
    _run(tmp_path, "m-E2", "m", [e2], noise=1.0)
    _run(tmp_path, "m-again", "m", [e1], noise=1.0)

    points = fq.collect([tmp_path / "m-E1", tmp_path / "m-E2"], tmp_path)
    assert set(points) == {"m"} and set(points["m"].meter) == {"E1", "E2"}
    with pytest.raises(ValueError, match="already scored"):
        fq.collect([tmp_path / "m-E1", tmp_path / "m-again"], tmp_path)


def test_rejects_incomplete_run(tmp_path):
    datasets = [_stage(tmp_path, "E1")]
    run = _run(tmp_path, "r", "m", datasets, noise=1.0)
    config = json.loads((run / "run-config.json").read_text())
    (run / "run-config.json").write_text(json.dumps({**config, "status": "running"}))
    with pytest.raises(ValueError, match="not completed"):
        fq.load_run(run, tmp_path)
