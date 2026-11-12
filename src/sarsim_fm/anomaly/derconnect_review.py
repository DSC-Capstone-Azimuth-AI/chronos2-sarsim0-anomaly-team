"""Reproducible prepared-resolution injections and label-free review bundles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sarsim_fm.anomaly.derconnect import SIGNAL, stage_manifest
from sarsim_fm.anomaly.export import verify_bundle, write_json
from sarsim_fm.anomaly.provenance import file_hash, object_hash
from sarsim_fm.data.annotations import covers_normal, load_annotations, window
from sarsim_fm.data.anomaly_injection import inject_anomaly

REVIEW_CATEGORIES = ["expected_operation", "measurement_issue", "potential_operational_anomaly", "uncertain"]


def make_injections(cfg, out):
    stage = stage_manifest(cfg)
    spec = cfg["injections"]
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    annotations = load_annotations(cfg.get("annotations"))
    duration = spec["duration_seconds"]
    if type(duration) is not int or duration < 60 or duration % 60:
        raise ValueError("injection duration must be a positive multiple of 60 seconds")
    scenarios, backgrounds, labels = [], [], []
    for i, entry in enumerate(spec["windows"]):
        if set(entry) != {"meter_id", "start", "end", "split"} or entry["split"] not in {"development", "test"}:
            raise ValueError("injection windows require meter_id/start/end/split")
        start, end = window(entry)
        meter = entry["meter_id"]
        if meter not in cfg["meters"] or start != start.floor("min") or end != end.floor("min"):
            raise ValueError("invalid injection meter or alignment")
        if any(m == meter and start < e and end > s for m, s, e in backgrounds):
            raise ValueError("injection background windows must be disjoint")
        for key in ("scale", "calibration"):
            a, b = window(cfg["roles"][key])
            if start < b and end > a:
                raise ValueError("injections cannot alter calibration or scale windows")
        if not covers_normal(annotations, meter, SIGNAL, start, end):
            raise ValueError("injection backgrounds require approved reviewed-normal windows")
        if (end - start).total_seconds() > 86400:
            raise ValueError("v1 injection fixture backgrounds are capped at one day")
        frames = []
        for name in stage["prepared_parts"]:
            if name.startswith(f"prepared/{meter}/"):
                day = pd.Timestamp(Path(name).stem, tz="UTC")
                if day < end and day + pd.Timedelta(days=1) > start:
                    frame = pd.read_parquet(Path(cfg["staged"]) / name)
                    frames.append(frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc < end)])
        if not frames:
            raise ValueError("no injection background data")
        frame = pd.concat(frames).sort_values("timestamp_utc").reset_index(drop=True)
        if len(frame) != int((end - start).total_seconds() / 60) or not frame.usable.all():
            raise ValueError("injection background must be complete and usable")
        location = cfg["detection"]["context_length"]
        if location + duration // 60 > len(frame):
            raise ValueError("background too short for injection and causal context")
        before = object_hash(frame.actual.tolist())
        rng = np.random.default_rng(spec["seed"] + i)
        if spec["kind"] == "dropout":
            changed = frame.actual.to_numpy().copy()
            changed[location:location + duration // 60] = np.nan
            frame.loc[location:location + duration // 60 - 1, "usable"] = False
            frame.loc[location:location + duration // 60 - 1, "quality_flags"] = '["injected_missing_data"]'
        else:
            if spec["kind"] not in {"spike", "dip", "level_shift", "variance_burst", "stuck_sensor", "trend_change"}:
                raise ValueError("unsupported v1 injection kind")
            changed, _ = inject_anomaly(frame.actual.to_numpy(), rng, spec["kind"], spec["severity"],
                                        duration // 60, location, season_length=1)
        frame["actual"] = changed
        frame["injected_kind"] = spec["kind"]
        label_start = start + pd.Timedelta(minutes=location)
        label_end = label_start + pd.Timedelta(seconds=duration)
        scenario_id = object_hash([stage["preparation_id"], entry, spec["seed"] + i,
                                  spec["kind"], spec["severity"], duration])[:24]
        scenarios.append({"scenario_id": scenario_id, "split": entry["split"], "meter_id": meter,
                          "seed": spec["seed"] + i, "kind": spec["kind"], "severity": spec["severity"],
                          "background_start": start.isoformat(), "background_end": end.isoformat(),
                          "background_sha256": before, "resolution_seconds": 60,
                          "preparation_id": stage["preparation_id"]})
        labels.append({"scenario_id": scenario_id, "meter_id": meter, "signal": SIGNAL,
                       "start": label_start.isoformat(), "end": label_end.isoformat(),
                       "label": "unknown" if spec["kind"] == "dropout" else "anomaly",
                       "label_source": "injected_derconnect", "review_status": "approved",
                       "ground_truth_type": spec["kind"], "kind": "quality" if spec["kind"] == "dropout"
                       else "model_anomaly"})
        backgrounds.append((meter, start, end))
        if not out.exists():
            out.mkdir(parents=True)
        frame.to_parquet(out / f"{scenario_id}.parquet", index=False)
    if not scenarios:
        raise ValueError("at least one injection window required")
    write_json(out / "scenario-manifest.json", {"evaluation_kind": "injected_derconnect",
                                               "scenarios": scenarios, "annotations": labels,
                                               "source_prepared_hashes": stage["prepared_hashes"]})
    return scenarios


def review_bundle(run, out, compare_run=None):
    run, out = Path(run), Path(out)
    manifest = verify_bundle(run)
    if out.exists():
        raise FileExistsError(out)
    settings = manifest["resolved_config"].get("review", {})
    rng = np.random.default_rng(settings.get("seed", 42))
    width = settings.get("window_seconds", 1800)
    if type(width) is not int or not 60 <= width <= 86400:
        raise ValueError("review window_seconds must be between 60 and 86400")
    events = pd.read_parquet(run / "events.parquet")
    windows = [(e.meter_id, e.start_utc - pd.Timedelta(seconds=width // 2), "alert")
               for e in events.itertuples()]
    # Reservoir sampling yields reproducible random non-alert centers in bounded memory.
    reservoir, seen = [], 0
    count = settings.get("random_windows", 10)
    if type(count) is not int or not 0 <= count <= 1000:
        raise ValueError("review random_windows must be between 0 and 1000")
    other = verify_bundle(compare_run) if compare_run else None
    for name in manifest["prediction_parts"]:
        frame = pd.read_parquet(run / name)
        if other:
            matching = [n for n in other["prediction_parts"] if Path(n).parts[-2:] == Path(name).parts[-2:]]
            if matching:
                right = pd.read_parquet(Path(compare_run) / matching[0])
                joined = frame.merge(right[["meter_id", "timestamp_utc", "anomaly_flag"]],
                                     on=["meter_id", "timestamp_utc"], suffixes=("", "_other"))
                diff = joined.anomaly_flag.notna() & joined.anomaly_flag_other.notna() & (
                    joined.anomaly_flag != joined.anomaly_flag_other)
                for row in joined.loc[diff].head(count).itertuples():
                    windows.append((row.meter_id, row.timestamp_utc - pd.Timedelta(seconds=width // 2),
                                    "detector_disagreement"))
        for row in frame.loc[frame.anomaly_flag.eq(0) & ~frame.event_member].itertuples():
            start = row.timestamp_utc - pd.Timedelta(seconds=width // 2)
            end = start + pd.Timedelta(seconds=width)
            if len(events) and ((events.meter_id == row.meter_id) & (events.start_utc < end) &
                                (events.end_utc > start)).any():
                continue
            seen += 1
            candidate = (row.meter_id, start, "random_non_alert")
            if len(reservoir) < count:
                reservoir.append(candidate)
            elif count:
                choice = int(rng.integers(seen))
                if choice < count:
                    reservoir[choice] = candidate
    windows.extend(reservoir)
    out.mkdir(parents=True)
    templates = []
    for meter, start, reason in sorted(set(windows), key=str):
        end = start + pd.Timedelta(seconds=width)
        review_id = object_hash([manifest["run_id"], meter, start.isoformat(), reason])[:24]
        frames = []
        for name in manifest["prediction_parts"]:
            if Path(name).parent.name != meter:
                continue
            day = pd.Timestamp(Path(name).stem, tz="UTC")
            if day < end and day + pd.Timedelta(days=1) > start:
                frame = pd.read_parquet(run / name)
                frames.append(frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc < end)])
        if frames:
            pd.concat(frames).sort_values("timestamp_utc").to_parquet(out / f"{review_id}.parquet", index=False)
        templates.append({"review_id": review_id, "meter_id": meter, "signal": SIGNAL,
                          "start": start.isoformat(), "end": end.isoformat(), "selection_reason": reason,
                          "category": "uncertain", "review_status": "pending", "reviewer": "", "notes": ""})
    pd.DataFrame(templates, columns=["review_id", "meter_id", "signal", "start", "end",
                                   "selection_reason", "category", "review_status", "reviewer", "notes"]
                 ).to_csv(out / "annotation-template.csv", index=False)
    events.to_parquet(out / "events.parquet", index=False)
    pd.read_parquet(run / "quality-events.parquet").to_parquet(out / "quality-events.parquet", index=False)
    write_json(out / "review-manifest.json", {"run_id": manifest["run_id"], "seed": settings.get("seed", 42),
                                             "categories": REVIEW_CATEGORIES,
                                             "truth_mapping": None,
                                             "note": "Mapping categories to evaluation truth requires approved rules.",
                                             "comparison_run_id": other["run_id"] if other else None,
                                             "files": {p.name: file_hash(p) for p in sorted(out.glob("*"))
                                                       if p.is_file()}})
    return templates
