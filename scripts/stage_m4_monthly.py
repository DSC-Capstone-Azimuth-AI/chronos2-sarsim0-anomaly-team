from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
M4_ROOT = ROOT / "external" / "M4-data"
OUTPUT_DIR = ROOT / "data" / "m4"

TRAIN_PATH = (
    M4_ROOT / "Dataset" / "Train" / "Monthly-train.csv"
)
TEST_PATH = (
    M4_ROOT / "Dataset" / "Test" / "Monthly-test.csv"
)
INFO_PATH = M4_ROOT / "Dataset" / "M4-info.csv"


def read_wide_series(path: Path) -> dict[str, list[float]]:
    frame = pd.read_csv(path)
    id_column = frame.columns[0]
    series = {}

    for _, row in frame.iterrows():
        item_id = str(row[id_column])

        values = (
            pd.to_numeric(row.iloc[1:], errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float32)
        )

        if not np.isfinite(values).all():
            raise ValueError(f"{item_id} contains invalid values")

        series[item_id] = values.tolist()

    return series


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH, INFO_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    train = read_wide_series(TRAIN_PATH)
    test = read_wide_series(TEST_PATH)

    if len(train) != 48_000:
        raise ValueError(f"Expected 48,000 train series; got {len(train)}")

    if len(test) != 48_000:
        raise ValueError(f"Expected 48,000 test series; got {len(test)}")

    if set(train) != set(test):
        raise ValueError("Train and test series IDs do not match")

    bad_horizons = {
        item_id: len(values)
        for item_id, values in test.items()
        if len(values) != 18
    }

    if bad_horizons:
        raise ValueError(
            f"Unexpected test horizons: {list(bad_horizons.items())[:10]}"
        )

    info = pd.read_csv(INFO_PATH)
    info["M4id"] = info["M4id"].astype(str)
    info = info[info["M4id"].isin(train)].copy()

    if len(info) != 48_000:
        raise ValueError(
            f"Expected 48,000 metadata rows; got {len(info)}"
        )

    info = info.rename(columns={"M4id": "item_id"})

    ordered_ids = sorted(
        train,
        key=lambda value: int(value[1:]),
    )

    staged = pd.DataFrame(
        {
            "item_id": ordered_ids,
            "train_values": [train[item_id] for item_id in ordered_ids],
            "test_values": [test[item_id] for item_id in ordered_ids],
        }
    )

    staged["n_train"] = staged["train_values"].map(len)
    staged["horizon"] = 18
    staged["seasonal_period"] = 12

    staged = staged.merge(
        info,
        on="item_id",
        how="left",
        validate="one_to_one",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "monthly.parquet"
    staged.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    commit = subprocess.check_output(
        ["git", "-C", str(M4_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    manifest = {
        "dataset": "M4 Monthly",
        "source_commit": commit,
        "number_of_series": len(staged),
        "forecast_horizon": 18,
        "seasonal_period": 12,
        "training_input": "Monthly-train.csv",
        "evaluation_target": "Monthly-test.csv",
        "minimum_train_length": int(staged["n_train"].min()),
        "median_train_length": float(staged["n_train"].median()),
        "maximum_train_length": int(staged["n_train"].max()),
    }

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(staged[["item_id", "n_train", "horizon"]].head())
    print(manifest)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()