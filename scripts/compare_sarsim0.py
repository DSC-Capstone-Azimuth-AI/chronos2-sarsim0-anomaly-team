"""Compare staged hourly DERConnect meters against a generated SarSim0 split.

    uv run --locked --extra cpu --extra notebooks python scripts/compare_sarsim0.py \
        --sim data/eval/sim_derconnect_1h_test.parquet \
        --out outputs/hourly_derconnect_vs_sarsim0_1h.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUILDINGS = ["E2451", "E3020", "E3510", "E7080", "E7085"]


def znorm(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sim",
        type=Path,
        default=Path("data/eval/sim_derconnect_1h_test.parquet"),
        help="Path to the generated SarSim0 parquet file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/hourly_derconnect_vs_sarsim0_1h.png"),
        help="Output path for the comparison PNG.",
    )
    parser.add_argument(
        "--sim-indices",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("IDX1", "IDX2"),
        help="Two SarSim0 series indices to plot (default: 0 1).",
    )
    args = parser.parse_args()

    der_series = {}
    for b in BUILDINGS:
        row = pd.read_parquet(f"data/anomaly/derconnect-{b}.parquet").iloc[0]
        der_series[f"{b} ({row['building_type']})"] = np.asarray(row["series"], dtype=float)

    sim_df = pd.read_parquet(args.sim)
    sim_series = [np.concatenate([r.context, r.target]) for r in sim_df.itertuples()]

    print(f"=== Summary Stats (Hourly DERConnect vs. {args.sim.stem}) ===")
    for name, s in der_series.items():
        skew = pd.Series(s).skew()
        print(
            f"{name:22s} | min={s.min():7.1f}  median={np.median(s):7.1f}  "
            f"max={s.max():7.1f}  skew={skew:+.2f}"
        )
    for i in range(min(5, len(sim_series))):
        s = sim_series[i]
        skew = pd.Series(s).skew()
        label = f"SarSim0 #{i}"
        print(
            f"{label:22s} | min={s.min():7.1f}  median={np.median(s):7.1f}  "
            f"max={s.max():7.1f}  skew={skew:+.2f}"
        )

    i0, i1 = args.sim_indices
    _, axes = plt.subplots(4, 4, figsize=(18, 12))
    columns = [
        ("DERConnect E2451\n(instructional)", der_series["E2451 (instructional)"][:512]),
        ("DERConnect E3510\n(office)", der_series["E3510 (office)"][:512]),
        (f"{args.sim.stem} #{i0}", sim_series[i0][:512]),
        (f"{args.sim.stem} #{i1}", sim_series[i1][:512]),
    ]

    for col, (title, raw) in enumerate(columns):
        z = znorm(raw)
        axes[0, col].plot(z, lw=0.9)
        axes[0, col].set_title(f"{title}\nRaw: [{raw.min():.1f}, {raw.max():.1f}]")
        if col == 0:
            axes[0, col].set_ylabel("Z-scored load")

        axes[1, col].plot(z[:168], lw=1.1, color="teal")
        axes[1, col].set_title("First 168h (1 Week Zoom)")
        if col == 0:
            axes[1, col].set_ylabel("Z-scored load")

        axes[2, col].hist(z, bins=35, density=True, alpha=0.75, color="slateblue")
        axes[2, col].set_title("Value Distribution")
        if col == 0:
            axes[2, col].set_ylabel("Density")

        axes[3, col].plot(pd.Series(z).rolling(24).std(), lw=1.0, color="darkorange")
        axes[3, col].set_title("24h Rolling Std")
        if col == 0:
            axes[3, col].set_ylabel("Rolling Std")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()