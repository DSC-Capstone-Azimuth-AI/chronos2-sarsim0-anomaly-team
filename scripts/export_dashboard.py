"""Export one `sfm detect` run on DERConnect into flat files for the dashboard team.

    uv run python scripts/export_dashboard.py \
        outputs/anomaly/derconnect-hourly-chronos2-ft-5k \
        --model-name chronos2-ft-5k

Writes outputs/dashboard/<model-name>/{predictions,forecasts,events}.{csv,parquet} and a README.md.
Detector outputs index time by step (`t`); timestamps come from the -times.parquet files that
scripts/stage_derconnect.py writes next to each staged series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = "America/Los_Angeles"
QUANTILES = ["q01", "q05", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90", "q95", "q99"]


def to_utc(local: pd.Series) -> pd.Series:
    # The skipped spring-forward hour (Mar 8, 2 a.m.) has no UTC time: it is NaT and filled=True.
    return local.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="raise").dt.tz_convert("UTC")


def export(run: Path, model: str, out: Path, formats=("csv", "parquet")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "run-config.json").read_text())
    per_series = {s["item_id"]: s for s in json.loads((run / "per-series.json").read_text())}
    manifests = {m["name"]: m for m in json.loads((run / "dataset-manifest.json").read_text())}
    events_all = pd.read_parquet(run / "events.parquet")
    datasets = {Path(d).stem: ROOT / d for d in config["config"]["datasets"]}
    preds, fcsts, events = [], [], []
    for meter in sorted(per_series):
        manifest = manifests[per_series[meter]["dataset"]]
        path = datasets[manifest["name"]]
        times = pd.read_parquet(path.with_name(f"{path.stem}-times.parquet"))
        times["time_utc"] = to_utc(times.time_local)
        thresholds = {int(h): v for h, v in per_series[meter]["threshold_model"]["thresholds"].items()}

        s = pd.read_parquet(run / "scores" / f"{meter}.parquet")
        s = s.join(times, on="t")
        score = config["config"]["detector"]["score"]
        preds.append(pd.DataFrame({
            "meter": meter, "time_local": s.time_local, "time_utc": s.time_utc, "model": model,
            "actual_kw": s.actual, "score": s[score], "threshold": s.horizon.map(thresholds),
            "flag": s.alert.astype(int),
            "period": s.evaluation.map({True: "detection", False: "calibration"}),
            "filled": s.filled}))

        f = pd.read_parquet(run / "forecasts" / f"{meter}.parquet").join(times, on="t")
        fcsts.append(f.assign(meter=meter, model=model).rename(columns={"actual": "actual_kw"})[
            ["meter", "time_local", "time_utc", "model", "actual_kw", *QUANTILES]])

        step = pd.Timedelta(manifest["freq"])
        fq = f.set_index("t")
        side = pd.Series(s.actual.to_numpy() > fq.loc[s.t, "q50"].to_numpy(), index=s.t)[s.alert.to_numpy()]
        for n, e in enumerate(events_all[events_all.item_id == meter].itertuples(), 1):
            sides = {"above" if v else "below" for v in side.loc[e.start:e.end - 1]}
            peak = fq.loc[e.peak_time]
            start, end = times.time_local[e.start], times.time_local[e.end - 1] + step
            events.append({
                "event_id": f"{model}-{meter}-{n:03d}", "meter": meter, "model": model,
                "start": start, "end": end, "start_utc": times.time_utc[e.start],
                "end_utc": to_utc(pd.Series([end]))[0],
                "duration_h": (end - start) / pd.Timedelta("1h"),
                "peak_time": times.time_local[e.peak_time], "peak_score": e.peak_score,
                "peak_actual_kw": peak.actual, "peak_expected_kw": peak.q50,
                "direction": sides.pop() if len(sides) == 1 else "mixed"})

    tables = {"predictions": pd.concat(preds, ignore_index=True),
              "forecasts": pd.concat(fcsts, ignore_index=True),
              "events": pd.DataFrame(events)}
    for name, table in tables.items():
        if "parquet" in formats:
            table.to_parquet(out / f"{name}.parquet", index=False)
        if "csv" in formats:
            table.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
    detector = config["config"]["detector"]
    freq = next(iter(manifests.values()))["freq"]
    unit = "hour" if freq == "1h" else f"{freq} reading"
    (out / "README.md").write_text(README.format(
        formats=" and ".join(f.upper() if f == "csv" else f.title() for f in formats),
        model=model, checkpoint=config["checkpoint"]["spec"], run=run.name, freq=freq, unit=unit,
        meters=", ".join(sorted(per_series)), first=tables["predictions"].time_local.min(),
        mode=detector["mode"],
        score=detector["score"], fpr=detector["target_fpr"], events=detector["events"],
        prefix_end=next(iter(manifests.values()))["prefix_end_local"],
        n_pred=len(tables["predictions"]), n_events=len(events),
        coverage="\n".join(f"| {m} | {v['coverage_80']:.1%} | {v['coverage_98']:.1%} | {v['realized_fpr']:.1%} |"
                            for m, v in per_series.items())))
    print(out, {k: len(v) for k, v in tables.items()})


README = """# DERConnect anomaly detection: {model}

Detector outputs for meters {meters}, Jan–May 2026, one row per {unit}.
Each table is written as {formats}.

- Model: `{model}` (checkpoint `{checkpoint}`), run folder `{run}`
- Forecast mode: `{mode}`: each {unit} is forecast one step ({freq}) ahead
- Score: `{score}`; threshold set for a {fpr:.0%} false-alarm rate on the normal period
- Events: {events}
- {n_pred} scored rows, {n_events} events

## Signal and time

- `actual_kw` is total active power (phases A+B+C) in kW, averaged over each {freq} interval. A
  row's time is the start of its interval (hourly: 13:00 covers 13:00–14:00).
- `time_local` is America/Los_Angeles wall-clock time, so a building's schedule sits at the same
  time of day across the daylight-saving change. `time_utc` is the same instant in UTC.
- Mar 8, 2:00–3:00 a.m. does not exist locally (clocks sprang forward). That hour is
  interpolated: `filled = True` and `time_utc` is empty. Other gaps in the meter data are also
  interpolated and marked `filled = True`.

## Files

**predictions**: one row per meter per {unit}.
`meter, time_local, time_utc, model, actual_kw, score, threshold, flag, period, filled`

- `flag = 1` when `score > threshold`.
- `period = calibration` for rows before {prefix_end}. That stretch is *assumed* normal and is
  used to set the threshold; flags there are expected false alarms (about {fpr:.0%}). Show or count
  flags only where `period = detection` (Mar 1 – May 31).
- Rows start {first}: the pipeline scores only the last three quarters of the normal period (the
  first quarter is history for the first forecasts).

**forecasts**: `meter, time_local, time_utc, model, actual_kw, q01 … q99` (13 quantiles of the
expected kW; q50 is the median). Plot q10–q90 and q01–q99 as bands around `actual_kw`.

**events**: consecutive flagged rows grouped into one event (detection period only).
`event_id, meter, model, start, end, start_utc, end_utc, duration_h, peak_time, peak_score,
peak_actual_kw, peak_expected_kw, direction`

- `end` is exclusive: a one-hour event at 13:00 has `end = 14:00`. `duration_h` is in hours.
- `direction` is `above` or `below` the forecast median for all flagged rows, else `mixed`.

## Caveats

- **Candidates, not confirmed anomalies.** DERConnect has no labels. Events need checking against
  facilities records.
- **No anomaly types.** The detector finds anomalous intervals; it does not classify them as spikes,
  dips, level shifts, etc. `direction` and `duration_h` are descriptive hints only.
- **No precision, recall or F1 for this data.** Without labels they cannot be computed, and the
  metrics in the run folder's `summary.json` count every event as a false positive. Do not show them.
- **What can be reported:** how often actual load fell inside the forecast bands (near 80% for
  q10–q90 and 98% for q01–q99 means well-calibrated forecasts) and the share of detection-period
  rows flagged (target {fpr:.0%}):

| Meter | Inside q10–q90 | Inside q01–q99 | Rows flagged |
|---|---|---|---|
{coverage}

- **Assumed-normal period.** Thresholds assume Jan–Feb was normal for every meter. Anything unusual
  in that stretch raises the threshold and hides similar events later.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="sfm detect output folder")
    parser.add_argument("--model-name", required=True, help="name shown on the dashboard")
    parser.add_argument("--out", type=Path, help="default: outputs/dashboard/<model-name>")
    parser.add_argument("--formats", nargs="+", default=["csv", "parquet"], choices=["csv", "parquet"])
    args = parser.parse_args()
    export(args.run.resolve(), args.model_name,
           args.out or ROOT / "outputs" / "dashboard" / args.model_name, args.formats)


if __name__ == "__main__":
    main()
