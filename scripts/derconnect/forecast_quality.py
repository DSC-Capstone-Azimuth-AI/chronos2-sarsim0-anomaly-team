"""E1: compare forecast quality of several models on staged DERConnect meters.

Reads completed `sfm detect` runs (one per model) on the same staged data, e.g. from
configs/anomaly/derconnect-hourly.yaml, and scores their saved forecasts. It loads no models.

    uv run sfm detect configs/anomaly/derconnect-hourly.yaml --model chronos2:amazon/chronos-2
    uv run sfm detect configs/anomaly/derconnect-hourly.yaml \
        --model chronos2:azimuth-anomaly-detection/chronos2-ft-500
    uv run python scripts/derconnect/forecast_quality.py \
        outputs/anomaly/derconnect-hourly-* --reference chronos2:amazon/chronos-2

Only evaluation rows are scored: at or after each meter's clean_prefix_length (Mar-May for the
hourly config), so nothing used to fit thresholds is counted. Interpolated (`filled`) actuals are
excluded. MASE and scaled quantile loss (SQL) divide by the meter's in-sample seasonal-naive MAE
on its prefix, so the scale is identical across models and comparisons are fair.

The paired comparison resamples whole blocks (meter-weeks by default) because errors on nearby
hours are correlated. Writes per-meter.csv, overall.csv, paired.csv and report.md to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
QCOLS = ["q01", "q05", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90", "q95", "q99"]
LEVELS = np.array([int(c[1:]) / 100 for c in QCOLS])


def naive_scale(history: np.ndarray, season: int) -> float:
    """In-sample seasonal-naive MAE, as in sarsim_fm.metrics.naive_scale."""
    if len(history) <= season:
        return np.nan
    scale = float(np.mean(np.abs(history[season:] - history[:-season])))
    return scale if np.isfinite(scale) and scale > 1e-12 else np.nan


def load_meters(datasets: list[str], data_root: Path) -> dict[str, dict]:
    """Staged series, prefix, scale and local timestamps for each meter, keyed by item_id."""
    meters = {}
    for source in datasets:
        path = Path(source) if Path(source).is_absolute() else data_root / source
        row = pd.read_parquet(path).iloc[0]
        series = np.asarray(row["series"], dtype=np.float64)
        prefix, season = int(row["clean_prefix_length"]), int(row["season_length"])
        times_path = path.with_name(f"{path.stem}-times.parquet")
        times = pd.read_parquet(times_path) if times_path.exists() else None
        meters[str(row["item_id"])] = {
            "prefix": prefix, "season": season, "scale": naive_scale(series[:prefix], season),
            "time_local": None if times is None else pd.to_datetime(times.time_local).to_numpy(),
            "filled": None if times is None else times.filled.to_numpy(bool)}
    return meters


def load_run(run: Path, data_root: Path) -> tuple[str, pd.DataFrame]:
    """One row per scored point: meter, t, block keys, scaled absolute error and pinball loss."""
    config = json.loads((run / "run-config.json").read_text())
    if config.get("status") != "completed":
        raise ValueError(f"{run}: run is not completed")
    meters = load_meters(config["config"]["datasets"], data_root)
    frames = []
    for meter, info in meters.items():
        f = pd.read_parquet(run / "forecasts" / f"{meter}.parquet")
        f = f[f.t >= info["prefix"]]
        t = f.t.to_numpy()
        keep = np.ones(len(f), bool) if info["filled"] is None else ~info["filled"][t]
        f, t = f[keep], t[keep]
        q = np.sort(f[QCOLS].to_numpy(), axis=1)  # non-crossing, as the detector scores them
        y = f.actual.to_numpy()
        diff = y[:, None] - q
        pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=1)
        time = info["time_local"][t] if info["time_local"] is not None else None
        frames.append(pd.DataFrame({
            "meter": meter, "t": t,
            "day": pd.to_datetime(time).normalize() if time is not None else t // info["season"],
            "week": pd.to_datetime(time).to_period("W").start_time if time is not None
            else t // (7 * info["season"]),
            "abs_err": np.abs(y - q[:, QCOLS.index("q50")]) / info["scale"],
            "sql": 2 * pinball / info["scale"],
            "in80": (y >= q[:, QCOLS.index("q10")]) & (y <= q[:, QCOLS.index("q90")]),
            "in98": (y >= q[:, QCOLS.index("q01")]) & (y <= q[:, QCOLS.index("q99")])}))
    return config["model"], pd.concat(frames, ignore_index=True)


def collect(runs: list[Path], data_root: Path) -> dict[str, pd.DataFrame]:
    """Scored points per model. Runs of one model split by meter (e.g. one NRP Job per building)
    are merged; scoring the same meter twice for a model is an error."""
    points: dict[str, pd.DataFrame] = {}
    for run in runs:
        model, frame = load_run(run, data_root)
        if model in points:
            overlap = set(points[model].meter) & set(frame.meter)
            if overlap:
                raise ValueError(f"{run}: meters {sorted(overlap)} already scored for {model}")
            frame = pd.concat([points[model], frame], ignore_index=True)
        points[model] = frame
    return points


def summarize(points: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return points.groupby(by).agg(
        n=("t", "size"), mase=("abs_err", "mean"), sql=("sql", "mean"),
        coverage_80=("in80", "mean"), coverage_98=("in98", "mean")).reset_index()


def paired(points: dict[str, pd.DataFrame], reference: str, block: str, n_boot: int,
           seed: int) -> pd.DataFrame:
    """Model minus reference on shared points; bootstrap CI over (meter, block) units."""
    rng = np.random.default_rng(seed)
    ref = points[reference].set_index(["meter", "t"])
    rows = []
    for model, frame in points.items():
        if model == reference:
            continue
        joined = frame.set_index(["meter", "t"]).join(ref, rsuffix="_ref", how="inner")
        units = joined.groupby(["meter", block]).agg(
            mase=("abs_err", "mean"), mase_ref=("abs_err_ref", "mean"),
            sql=("sql", "mean"), sql_ref=("sql_ref", "mean"))
        row = {"model": model, "reference": reference, "block": block, "n_units": len(units),
               "n_points": len(joined)}
        for metric in ["mase", "sql"]:
            delta = (units[metric] - units[f"{metric}_ref"]).to_numpy()
            boots = rng.choice(delta, size=(n_boot, len(delta))).mean(axis=1)
            row.update({f"d_{metric}": delta.mean(),
                        f"d_{metric}_lo": np.quantile(boots, 0.025),
                        f"d_{metric}_hi": np.quantile(boots, 0.975),
                        f"{metric}_units_improved": float((delta < 0).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def markdown(overall: pd.DataFrame, comparison: pd.DataFrame, reference: str | None) -> str:
    lines = ["# E1: DERConnect forecast quality", "",
             "Evaluation rows only (after each meter's clean prefix), interpolated points excluded.",
             "MASE and SQL: lower is better. Coverage: nominal 0.80 and 0.98.", "",
             "| Model | Points | MASE | SQL | Cov 80 | Cov 98 |", "|---|---:|---:|---:|---:|---:|"]
    for r in overall.sort_values("sql").itertuples():
        lines.append(f"| `{r.model}` | {r.n} | {r.mase:.4f} | {r.sql:.4f} | "
                     f"{r.coverage_80:.3f} | {r.coverage_98:.3f} |")
    if reference and not comparison.empty:
        block = comparison.block.iloc[0]
        lines += ["", f"Paired against `{reference}` (negative = better; 95% CI over meter-{block}s):",
                  "", "| Model | ΔMASE [95% CI] | ΔSQL [95% CI] | Units improved (SQL) |",
                  "|---|---|---|---:|"]
        for r in comparison.sort_values("d_sql").itertuples():
            lines.append(f"| `{r.model}` | {r.d_mase:+.4f} [{r.d_mase_lo:+.4f}, {r.d_mase_hi:+.4f}] | "
                         f"{r.d_sql:+.4f} [{r.d_sql_lo:+.4f}, {r.d_sql_hi:+.4f}] | "
                         f"{r.sql_units_improved:.0%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path,
                        help="completed `sfm detect` run folders; runs of one model split by meter are merged")
    parser.add_argument("--reference", help="model spec to compare against, e.g. chronos2:amazon/chronos-2")
    parser.add_argument("--block", choices=["day", "week"], default="week",
                        help="bootstrap unit within each meter (default: week)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT,
                        help="base for relative dataset paths in run-config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "e1-forecast-quality")
    args = parser.parse_args()

    points = collect(args.runs, args.data_root)
    if args.reference and args.reference not in points:
        raise ValueError(f"--reference {args.reference} is not among the runs: {sorted(points)}")

    every = pd.concat([f.assign(model=m) for m, f in points.items()], ignore_index=True)
    per_meter = summarize(every, ["model", "meter"])
    overall = summarize(every, ["model"])
    comparison = (paired(points, args.reference, args.block, args.n_boot, args.seed)
                  if args.reference else pd.DataFrame())

    args.out.mkdir(parents=True, exist_ok=True)
    per_meter.to_csv(args.out / "per-meter.csv", index=False)
    overall.to_csv(args.out / "overall.csv", index=False)
    comparison.to_csv(args.out / "paired.csv", index=False)
    report = markdown(overall, comparison, args.reference)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
