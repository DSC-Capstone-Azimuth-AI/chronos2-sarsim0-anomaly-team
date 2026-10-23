"""Common on-disk format for every evaluation set.

`data/eval/<name>.parquet` with one row per series:
    item_id        str
    context        list[float]   history given to the model
    target         list[float]   values to forecast (fixed horizon per set)
    season_length  int           used for MASE scaling and seasonal naive
Extra columns (e.g. M4 metadata) are allowed and ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.runtime import ROOT

EVAL_DIR = ROOT / "data" / "eval"
REQUIRED_COLUMNS = ("item_id", "context", "target", "season_length")


@dataclass
class EvalSet:
    name: str
    item_ids: list[str]
    contexts: list[np.ndarray]
    targets: list[np.ndarray]
    season_lengths: np.ndarray

    @property
    def horizon(self) -> int:
        return len(self.targets[0])

    def __len__(self) -> int:
        return len(self.item_ids)


def eval_set_path(name: str) -> Path:
    return EVAL_DIR / f"{name}.parquet"


def write_eval_set(frame: pd.DataFrame, name: str) -> Path:
    validate_eval_frame(frame, name)
    path = eval_set_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def validate_eval_frame(frame: pd.DataFrame, name: str) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame["item_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate item_id values")

    horizons = frame["target"].map(len)
    if horizons.nunique() != 1:
        raise ValueError(f"{name}: targets must share one horizon, got {sorted(horizons.unique())}")

    for column in ("context", "target"):
        finite = frame[column].map(lambda v: bool(np.isfinite(np.asarray(v, np.float32)).all()))
        if not finite.all():
            raise ValueError(f"{name}: non-finite values in {column}")


def load_eval_set(name: str) -> EvalSet:
    path = eval_set_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with `sfm stage-m4` or `sfm make-sim`")
    frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    return EvalSet(
        name=name,
        item_ids=frame["item_id"].astype(str).tolist(),
        contexts=[np.asarray(v, dtype=np.float32) for v in frame["context"]],
        targets=[np.asarray(v, dtype=np.float32) for v in frame["target"]],
        season_lengths=frame["season_length"].to_numpy(dtype=np.int64),
    )
