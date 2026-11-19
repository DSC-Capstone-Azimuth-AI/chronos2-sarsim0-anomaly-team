"""Stage DERConnect building meters for `sfm detect`: one hourly total-power series per building.

    uv run python scripts/stage_derconnect.py ~/Downloads/Haojian_Data.zip
    uv run python scripts/stage_derconnect.py <zip> --freq 15min --prefix-end 2026-02-01
    uv run python scripts/stage_derconnect.py <zip> --freq 2s --buildings E3020 \
        --start 2026-03-09 --end 2026-03-12 --prefix-end 2026-03-10     # native resolution, 3 days

Reads the 2-second CSVs straight from the zip (no unpacking), sums Active_Power_A/B/C, converts W to
kW and averages per interval of local wall-clock time (America/Los_Angeles), so building schedules
keep the same position every day across the daylight-saving change; the skipped spring-forward hour
is interpolated like any other gap. Writes, per building, data/anomaly/derconnect-<ID>[-<freq>].parquet
(the detector input), a .manifest.json and a -<freq>-times.parquet with timestamps for plotting.

There are no anomaly labels. `clean_prefix_length` covers everything before --prefix-end (local
time): the detector treats it as normal and fits its residual scale and thresholds there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "anomaly"
TZ = "America/Los_Angeles"
POWER = ["Active_Power_A", "Active_Power_B", "Active_Power_C"]
# Building type per facilities management.
BUILDINGS = {"E2451": "instructional", "E3020": "mixed", "E3510": "office", "E7080": "office",
             "E7085": "office"}
MEMBER = re.compile(r"(?:.*/)?(E\d{4})_(\d{4}-\d{2})\.csv$")


def read_member(archive: zipfile.ZipFile, name: str, freq: str) -> pd.DataFrame:
    """Sum and count of total kW per interval, so chunk and file boundaries combine exactly."""
    parts = []
    with archive.open(name) as handle:
        for chunk in pd.read_csv(handle, usecols=["_time", *POWER], chunksize=1_000_000):
            time = pd.to_datetime(chunk["_time"], utc=True, format="ISO8601").dt.tz_convert(TZ)
            time = time.dt.tz_localize(None)  # wall-clock time; the autumn repeated hour is out of range
            kw = chunk[POWER].sum(axis=1, min_count=len(POWER)) / 1000
            frame = pd.DataFrame({"kw": kw.to_numpy()}, index=time.dt.floor(freq)).dropna()
            parts.append(frame.groupby(level=0).kw.agg(["sum", "count"]))
    return pd.concat(parts).groupby(level=0).sum()


def stage(zip_path: Path, freq: str, prefix_end: str, out: Path, buildings=None, start=None,
          end=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    season = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    cutoff = pd.Timestamp(prefix_end)
    lo, hi = pd.Timestamp(start or "1900-01-01"), pd.Timestamp(end or "2100-01-01")
    suffix = "" if freq == "1h" and not (start or end) else f"-{freq}"
    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for name in archive.namelist():
            match = MEMBER.match(name)
            month = match and pd.Timestamp(match.group(2) + "-01")
            # Monthly files are split at local midnight; skip months outside the window.
            if match and (not buildings or match.group(1) in buildings) and \
                    month < hi and month + pd.offsets.MonthBegin(1) > lo:
                members.setdefault(match.group(1), []).append(name)
        for building in sorted(members):
            stats = pd.concat(read_member(archive, name, freq) for name in sorted(members[building]))
            stats = stats.groupby(level=0).sum()
            stats = stats[(stats.index >= lo) & (stats.index < hi)]
            index = pd.date_range(stats.index.min(), stats.index.max(), freq=freq)
            mean = (stats["sum"] / stats["count"]).reindex(index)
            missing = mean.isna().to_numpy()
            series = mean.interpolate(limit_direction="both").to_numpy(np.float64)
            prefix = int((index < cutoff).sum())
            if not 1 < prefix < len(series):
                raise ValueError(f"{building}: --prefix-end {prefix_end} leaves no prefix or no suffix")

            stem = f"derconnect-{building}{suffix}"
            pd.DataFrame({"time_local": index, "kw": series, "filled": missing}).to_parquet(
                out / f"{stem}-times.parquet", index=False)
            row = {"item_id": building, "series": [series.tolist()], "clean_prefix_length": prefix,
                   "season_length": season, "building_type": BUILDINGS.get(building, "unknown"),
                   "start_local": str(index[0]), "freq": freq, "filled_points": int(missing.sum())}
            path = out / f"{stem}.parquet"
            pd.DataFrame(row, index=[0]).to_parquet(path, index=False)
            manifest = {"name": stem, "source": zip_path.name, "members": sorted(members[building]),
                        "signal": "total active power (A+B+C), kW, interval mean", "freq": freq,
                        "time": f"{TZ} wall clock", "prefix_end_local": prefix_end,
                        "n_points": len(series), "clean_prefix_length": prefix,
                        "filled_points": int(missing.sum()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"{building} ({row['building_type']}): {len(series)} points from {index[0]} to "
                  f"{index[-1]}, prefix {prefix}, filled {int(missing.sum())}, "
                  f"kW {series.min():.1f}-{series.max():.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", type=Path, help="DERConnect export (Haojian_Data.zip)")
    parser.add_argument("--freq", default="1h", help="resampling interval (pandas alias)")
    parser.add_argument("--prefix-end", default="2026-03-01",
                        help="local date where the assumed-normal prefix ends")
    parser.add_argument("--buildings", nargs="*", help="device IDs to stage (default: all)")
    parser.add_argument("--start", help="first local date to keep (default: all)")
    parser.add_argument("--end", help="local date to stop before (default: all)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    stage(args.zip.expanduser(), args.freq, args.prefix_end, args.out, args.buildings, args.start,
          args.end)


if __name__ == "__main__":
    main()
