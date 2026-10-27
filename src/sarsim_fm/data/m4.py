"""Stage M4 subsets from the Mcompetitions/M4-methods CSVs into the common eval format.

Monthly is the zero-shot test set; Quarterly is the real-data validation set for model selection
(sweeps, checkpoint choice), so Monthly never steers it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.data.eval_sets import EVAL_DIR, write_eval_set
from sarsim_fm.runtime import ROOT

M4_ROOT = ROOT / "external" / "M4-data"
# Official M4 counts, horizons and seasonal periods per frequency.
FREQUENCIES = {
    "monthly": {"n_series": 48_000, "horizon": 18, "season_length": 12},
    "quarterly": {"n_series": 24_000, "horizon": 8, "season_length": 4},
}


def _read_wide_series(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    ids = frame.iloc[:, 0].astype(str)
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    # Rows are right-padded with NaN to the longest series.
    return {item_id: row[~np.isnan(row)] for item_id, row in zip(ids, values)}


def stage_m4(freq: str = "monthly", m4_root: Path = M4_ROOT) -> Path:
    spec = FREQUENCIES[freq]
    n_series, horizon, season_length = spec["n_series"], spec["horizon"], spec["season_length"]
    name, title = f"m4_{freq}", freq.capitalize()
    dataset = m4_root / "Dataset"
    train_path = dataset / "Train" / f"{title}-train.csv"
    test_path = dataset / "Test" / f"{title}-test.csv"
    info_path = dataset / "M4-info.csv"
    for path in (train_path, test_path, info_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} (see README: Download the M4 data)")

    train = _read_wide_series(train_path)
    test = _read_wide_series(test_path)
    if len(train) != n_series or set(train) != set(test):
        raise ValueError(f"Expected {n_series} matching train/test series; got {len(train)}/{len(test)}")
    bad = [k for k, v in test.items() if len(v) != horizon]
    if bad:
        raise ValueError(f"Unexpected test horizons for {bad[:10]}")

    info = pd.read_csv(info_path).rename(columns={"M4id": "item_id"})
    info["item_id"] = info["item_id"].astype(str)

    ordered = sorted(train, key=lambda item_id: int(item_id[1:]))
    frame = pd.DataFrame(
        {
            "item_id": ordered,
            "context": [train[i].tolist() for i in ordered],
            "target": [test[i].tolist() for i in ordered],
            "season_length": season_length,
        }
    ).merge(info, on="item_id", how="left", validate="one_to_one")

    path = write_eval_set(frame, name)

    lengths = frame["context"].map(len)
    manifest = {
        "dataset": f"M4 {title}",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(m4_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "number_of_series": len(frame),
        "forecast_horizon": horizon,
        "season_length": season_length,
        "context": train_path.name,
        "target": test_path.name,
        "context_length_min": int(lengths.min()),
        "context_length_median": float(lengths.median()),
        "context_length_max": int(lengths.max()),
    }
    (EVAL_DIR / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return path
