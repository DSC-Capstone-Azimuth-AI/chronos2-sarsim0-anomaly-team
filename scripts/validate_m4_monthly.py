from pathlib import Path

import numpy as np
import pandas as pd


path = Path("data/m4/monthly.parquet")
data = pd.read_parquet(path)

assert len(data) == 48_000
assert data["item_id"].nunique() == 48_000
assert data["horizon"].eq(18).all()
assert data["seasonal_period"].eq(12).all()
assert data["test_values"].map(len).eq(18).all()
assert data["train_values"].map(len).eq(data["n_train"]).all()

for column in ("train_values", "test_values"):
    valid = data[column].map(
        lambda values: np.isfinite(
            np.asarray(values, dtype=np.float32)
        ).all()
    )

    assert valid.all(), f"Invalid values found in {column}"

print("M4 Monthly validation passed")
print("Series:", len(data))
print("Minimum train length:", data["n_train"].min())
print("Median train length:", data["n_train"].median())
print("Maximum train length:", data["n_train"].max())