"""Disk-backed raw validation/sorting and causal UTC minute preparation."""
from __future__ import annotations

import itertools
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.export import VERSION, write_json
from sarsim_fm.anomaly.provenance import file_hash, object_hash
from sarsim_fm.data.annotations import utc

PREPARATION_VERSION = "utc-minute-v1"


def _minute(rows, start, prep):
    """Rows are already deduplicated, ordered, and conflict-quarantined."""
    values, times = [], []
    flags = set()
    for t, total, reason in rows:
        if reason:
            flags.add(reason)
        if total is not None:
            values.append(total)
            times.append(t)
    count = len(values)
    expected = 60 / prep["cadence_seconds"]
    coverage = min(count / expected, 1.0)
    edges = [start.value // 1000, *times, (start + pd.Timedelta(minutes=1)).value // 1000]
    max_gap = max(np.diff(edges)) / 1e6
    if count > expected:
        flags.add("oversampling")
    if not count:
        flags.add("missing_data")
    if max_gap > prep["max_gap_seconds"]:
        flags.add("gap")
    if coverage < prep["min_coverage"]:
        flags.add("low_coverage")
    valid = count > 0 and coverage >= prep["min_coverage"] and max_gap <= prep["max_gap_seconds"]
    # Conflicts are quarantined observations. Remaining values can only qualify via coverage/gaps.
    return {"timestamp_utc": start, "actual": float(np.mean(values)) if count else None,
            "minimum": min(values) if count else None, "maximum": max(values) if count else None,
            "observation_count": count, "coverage_fraction": coverage, "max_gap_seconds": max_gap,
            "usable": valid, "quality_flags": json.dumps(sorted(flags))}


def stage_derconnect(source, cfg, out):
    """Bound memory by input chunks, SQLite disk sort and one prepared day per meter.

    Original columns are retained in staged raw payloads. No demand filling or unit conversion.
    Output directories are never overwritten; incomplete staging must use a new directory.
    """
    source, out = Path(source), Path(out)
    files = sorted(source.rglob("*.csv")) if source.is_dir() else [source]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("no source CSV files")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    prep = cfg["preparation"]
    conn = sqlite3.connect(out / "validation.sqlite")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("CREATE TABLE raw (meter TEXT, t INTEGER, payload TEXT, total REAL, reason TEXT, "
                 "copies INTEGER DEFAULT 1, PRIMARY KEY(meter,t,payload)) WITHOUT ROWID")
    counts, sources, unexpected = Counter(), [], set()
    columns = None
    channel_stats = {}
    quarantine = (out / "quarantine.jsonl").open("w")
    try:
        for path in files:
            source_rows = 0
            digest = file_hash(path)
            for chunk in pd.read_csv(path, dtype=str, keep_default_na=False,
                                     chunksize=prep["chunk_size"]):
                required = {"_time", "DeviceID", *prep["phase_columns"]}
                if not required <= set(chunk.columns):
                    raise ValueError(f"missing source columns in {path.name}: {required - set(chunk)}")
                current = sorted(chunk.columns)
                if columns is None:
                    columns = current
                if columns != current:
                    raise ValueError(f"inconsistent schema in {path.name}")
                records = []
                for row in chunk.to_dict("records"):
                    counts["input_rows"] += 1
                    source_rows += 1
                    meter = row["DeviceID"]
                    match = re.match(r"(E\d+)_([0-9]{4}-[0-9]{2})", path.stem)
                    try:
                        t = utc(row["_time"])
                    except (ValueError, TypeError):
                        counts["invalid_timestamps"] += 1
                        quarantine.write(json.dumps({"file": path.name, "row": row,
                                                     "reason": "invalid_timestamp"}) + "\n")
                        continue
                    if match and meter != match[1]:
                        raise ValueError(f"meter/file disagreement: {path.name}: {meter}")
                    if meter not in cfg["meters"]:
                        unexpected.add(f"meter:{meter}")
                    # The filename month is a calendar/source convention, not necessarily UTC.
                    month = match[2] if match else t.strftime("%Y-%m")
                    if cfg.get("months") and month not in cfg["months"]:
                        unexpected.add(f"month:{month}")
                    if unexpected and cfg["unexpected"] == "error":
                        raise ValueError(f"unexpected inputs: {sorted(unexpected)}")
                    numeric = {}
                    for name in current:
                        if name in {"_time", "DeviceID"}:
                            continue
                        try:
                            value = float(row[name])
                            if not np.isfinite(value):
                                raise ValueError("nonfinite")
                        except ValueError:
                            value = None
                            counts[f"invalid_numeric:{name}"] += 1
                        numeric[name] = value
                        if value is not None:
                            key = f"{meter}:{name}"
                            lo, hi = channel_stats.get(key, (value, value))
                            channel_stats[key] = (min(lo, value), max(hi, value))
                    phases = [numeric[k] for k in prep["phase_columns"]]
                    total = float(sum(phases)) if all(v is not None for v in phases) else None
                    if total is not None and not np.isfinite(total):
                        total = None
                    reason = "invalid_phase" if total is None else ""
                    # Canonical payload makes identical duplicates independent of column order.
                    payload = json.dumps({**row, "_time": t.isoformat()}, sort_keys=True)
                    records.append((meter, t.value // 1000, payload, total, reason))
                conn.executemany("INSERT INTO raw(meter,t,payload,total,reason) VALUES(?,?,?,?,?) "
                                 "ON CONFLICT(meter,t,payload) DO UPDATE SET copies=copies+1", records)
                conn.commit()
            if file_hash(path) != digest:
                raise ValueError(f"source changed while staging: {path.name}")
            sources.append({"name": str(path.relative_to(source)) if source.is_dir() else path.name,
                            "sha256": digest, "rows": source_rows})
        quarantine.close()
        parts, raw_parts, ranges = [], [], {}
        for (meter,) in conn.execute("SELECT DISTINCT meter FROM raw ORDER BY meter"):
            query = conn.execute("SELECT t,payload,total,reason,copies FROM raw WHERE meter=? ORDER BY t,payload",
                                 (meter,))
            raw_buffer, day_buffer, minute_rows = [], [], []
            current_bin, current_day, raw_index = None, None, 0
            first, last = None, None

            def flush_raw(raw_buffer=raw_buffer, meter=meter):
                nonlocal raw_index
                if raw_buffer:
                    name = f"raw/{meter}/part-{raw_index:06d}.parquet"
                    target = out / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(raw_buffer).to_parquet(target, index=False)
                    raw_parts.append(name)
                    raw_index += 1
                    raw_buffer.clear()

            def flush_day(day_buffer=day_buffer, meter=meter):
                if day_buffer:
                    name = f"prepared/{meter}/{day_buffer[0]['timestamp_utc']:%Y-%m-%d}.parquet"
                    target = out / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(day_buffer).assign(meter_id=meter).to_parquet(target, index=False)
                    parts.append(name)
                    day_buffer.clear()

            def emit(start, values, day_buffer=day_buffer):
                nonlocal current_day
                if current_day != start.date():
                    flush_day()
                    current_day = start.date()
                day_buffer.append(_minute(values, start, prep))

            for time_us, group in itertools.groupby(query, key=lambda r: r[0]):
                variants = list(group)
                counts["identical_duplicates"] += sum(r[4] - 1 for r in variants)
                conflict = len(variants) > 1
                counts["conflicting_timestamps"] += int(conflict)
                t = pd.Timestamp(time_us, unit="us", tz="UTC")
                first = first or t
                last = t
                for _, payload, total, reason, copies in variants:
                    raw_buffer.append({"timestamp_utc": t, "meter_id": meter, "payload": payload,
                                       "copies": copies, "quality_reason": "conflict" if conflict else reason})
                    if len(raw_buffer) >= prep["chunk_size"]:
                        flush_raw()
                bin_start = t.floor("min")
                if current_bin is None:
                    current_bin = bin_start
                while current_bin < bin_start:
                    emit(current_bin, minute_rows)
                    minute_rows = []
                    current_bin += pd.Timedelta(minutes=1)
                minute_rows.append((time_us, None if conflict else variants[0][2],
                                    "conflict" if conflict else variants[0][3]))
                counts["unique_timestamps"] += 1
                counts["excluded_observations"] += int(conflict or variants[0][2] is None)
            if current_bin is not None:
                emit(current_bin, minute_rows)
            flush_day()
            flush_raw()
            ranges[meter] = {"start": first.isoformat(), "end_inclusive": last.isoformat()}
        manifest = {"schema_version": VERSION, "preparation_version": PREPARATION_VERSION,
                    "preparation": prep, "sources": sources, "counts": dict(counts),
                    "unexpected": sorted(unexpected), "columns": columns, "ranges": ranges,
                    "constant_channels": [k for k, (lo, hi) in sorted(channel_stats.items()) if lo == hi],
                    "channel_ranges": channel_stats, "units_status": "confirmed" if prep["units_confirmed"]
                    else "unconfirmed", "unit": prep["unit"], "prepared_parts": parts,
                    "raw_parts": raw_parts, "complete": True}
        manifest["prepared_hashes"] = {p: file_hash(out / p) for p in parts}
        manifest["raw_hashes"] = {p: file_hash(out / p) for p in raw_parts}
        manifest["preparation_id"] = object_hash({"version": PREPARATION_VERSION,
                                                  "sources": sorted(s["sha256"] for s in sources),
                                                  "preparation": {k: v for k, v in prep.items()
                                                                  if k != "chunk_size"},
                                                  "prepared_hashes": manifest["prepared_hashes"]})
        write_json(out / "stage-manifest.json", manifest)
        return manifest
    finally:
        quarantine.close()
        conn.close()
