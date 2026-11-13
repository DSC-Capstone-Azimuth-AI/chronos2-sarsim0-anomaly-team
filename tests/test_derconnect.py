"""Offline DERCONNECT acceptance fixtures: no private data, weights, or network."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from sarsim_fm.anomaly.derconnect import (
    _forecast_day,
    detect_derconnect,
    dry_run,
)
from sarsim_fm.anomaly.derconnect_evaluation import _match, evaluate_files
from sarsim_fm.anomaly.derconnect_review import make_injections, review_bundle
from sarsim_fm.anomaly.export import (
    FIELDS,
    NULLABLE,
    export_bundle,
    hourly_summary,
    schema,
    verify_bundle,
    write_csv,
    write_table,
)
from sarsim_fm.anomaly.provenance import derconnect_model_identity, file_hash
from sarsim_fm.cli import main
from sarsim_fm.config import load_config, validate_config
from sarsim_fm.data.annotations import load_annotations, utc
from sarsim_fm.data.derconnect import _minute, stage_derconnect

FIXTURE = Path(__file__).parent / "fixtures/derconnect"
CONFIG = Path(__file__).resolve().parents[1] / "configs/derconnect/tiny-pilot.yaml"


class FakeForecaster:
    derconnect_identity: ClassVar[dict] = {"name": "offline-fixture", "version": 1}

    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        point = np.array([np.median(x) for x in contexts])
        # Deliberately crossed quantiles verify the shared rearrangement primitive.
        return point[:, None, None] + np.array([.2, -.2, 0., .1, -.1])[None, None, :]


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("derconnect")
    cfg = load_config(CONFIG)
    cfg["annotations"] = str((FIXTURE / "annotations.json").resolve())
    cfg["staged"] = str(root / "stage")
    stage_derconnect(FIXTURE, cfg, cfg["staged"])
    return cfg, root


@pytest.fixture(scope="module")
def completed(staged):
    cfg, root = staged
    run = root / "run"
    result = detect_derconnect(cfg, "chronos2:offline-fake", run, forecaster=FakeForecaster())
    return cfg, run, result


def points(run, manifest):
    return pd.concat([pd.read_parquet(run / p) for p in manifest["prediction_parts"]], ignore_index=True)


def test_staging_counts_and_chunk_invariance(staged, tmp_path):
    cfg, _ = staged
    original = json.loads((Path(cfg["staged"]) / "stage-manifest.json").read_text())
    changed = copy.deepcopy(cfg)
    changed["preparation"]["chunk_size"] = 31
    other = stage_derconnect(FIXTURE, changed, tmp_path / "small-chunks")
    assert original["counts"] == other["counts"]
    assert original["prepared_hashes"] == other["prepared_hashes"]
    assert original["preparation_id"] == other["preparation_id"]
    assert original["counts"]["identical_duplicates"] == 6
    assert original["counts"]["conflicting_timestamps"] == 2
    assert original["counts"]["excluded_observations"] == 4
    assert original["units_status"] == "unconfirmed"
    assert "E2451:Frequency" in original["constant_channels"]
    frame = pd.read_parquet(Path(cfg["staged"]) / "prepared/E2451/2026-01-02.parquet")
    zero = frame[frame.timestamp_utc.eq(utc("2026-01-02T00:10:00Z"))].iloc[0]
    assert zero.actual == 0 and zero.usable
    missing = frame[frame.timestamp_utc.eq(utc("2026-01-02T00:20:00Z"))].iloc[0]
    assert not missing.usable and pd.isna(missing.actual)
    assert frame.timestamp_utc.dt.tz is not None


def test_input_file_order_equivalence(staged, tmp_path):
    cfg, _ = staged
    src = tmp_path / "reordered"
    src.mkdir()
    files = list(FIXTURE.glob("*.csv"))
    for i, path in enumerate(reversed(files)):
        (src / f"source-{i}.csv").write_bytes(path.read_bytes())
    original = json.loads((Path(cfg["staged"]) / "stage-manifest.json").read_text())
    other = stage_derconnect(src, cfg, tmp_path / "stage")
    assert other["preparation_id"] == original["preparation_id"]


@pytest.mark.parametrize("case", ["missing_column", "meter", "timezone", "numeric", "unexpected"])
def test_input_validation(staged, tmp_path, case):
    cfg = copy.deepcopy(staged[0])
    cfg["preparation"]["chunk_size"] = 1
    src = tmp_path / "E2451_2026-01.csv"
    row = {"_time": "2026-01-01T00:00:00.123456Z", "DeviceID": "E2451",
           "Active_Power_A": "1", "Active_Power_B": "2", "Active_Power_C": "3"}
    if case == "missing_column":
        del row["Active_Power_C"]
    elif case == "meter":
        row["DeviceID"] = "E3020"
    elif case == "timezone":
        row["_time"] = "2026-01-01 00:00:00"
    elif case == "numeric":
        row["Active_Power_A"] = "inf"
    elif case == "unexpected":
        cfg["meters"] = ["E3020"]
    pd.DataFrame([row]).to_csv(src, index=False)
    if case in {"missing_column", "meter", "unexpected"}:
        with pytest.raises(ValueError):
            stage_derconnect(src, cfg, tmp_path / "out")
    else:
        result = stage_derconnect(src, cfg, tmp_path / "out")
        assert result["counts"].get("invalid_timestamps", 0) + result["counts"].get("excluded_observations", 0) == 1


def test_bin_edges_gaps_and_no_future_data(staged):
    prep = staged[0]["preparation"]
    t = utc("2026-01-01T00:00:00Z")
    readings = [(t.value // 1000 + i * 1_000_000 + 123456, -3., "") for i in range(0, 60, 2)]
    result = _minute(readings, t, prep)
    assert result["usable"] and result["actual"] == -3 and result["coverage_fraction"] == 1
    assert not _minute(readings[:24], t, prep)["usable"]  # edge gap despite 80% count
    assert _minute(readings + readings, t, prep)["coverage_fraction"] == 1
    assert "oversampling" in _minute(sorted(readings + readings), t, prep)["quality_flags"]
    assert utc("2026-11-01T01:30:00-07:00") != utc("2026-11-01T01:30:00-08:00")


def test_predictions_events_and_empty_schemas(completed, tmp_path):
    _, run, manifest = completed
    frame = points(run, manifest)
    assert len(frame) == 120
    assert not frame.duplicated(["run_id", "detector_id", "meter_id", "signal", "timestamp_utc"]).any()
    assert (frame.bin_end_utc - frame.timestamp_utc).dt.total_seconds().eq(60).all()
    assert frame.available_at_utc.equals(frame.bin_end_utc)
    assert frame.q50.equals(frame.expected_power)
    q = frame.loc[frame.q50.notna(), ["q01", "q10", "q50", "q90", "q99"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert set(frame.anomaly_type) == {"unknown"}
    unscored = frame[frame.prediction_status != "scored"]
    assert unscored.anomaly_flag.isna().all()
    assert unscored.anomaly_score.isna().all()
    missing = frame[frame.timestamp_utc.eq(utc("2026-01-02T00:20:00Z"))]
    assert missing.prediction_status.eq("insufficient_coverage").all()
    after = frame[frame.timestamp_utc.eq(utc("2026-01-02T00:21:00Z"))]
    assert after.prediction_status.eq("warmup").all()
    at_midnight = frame[frame.timestamp_utc.eq(utc("2026-01-02T00:00:00Z"))]
    assert at_midnight.prediction_status.eq("scored").all()
    events = pd.read_parquet(run / "events.parquet")
    crossing = events[(events.start_utc < utc("2026-01-02T00:00:00Z")) &
                      (events.end_utc > utc("2026-01-02T00:00:00Z"))]
    assert len(crossing) == 2
    for e in events.itertuples():
        member = frame[frame.event_id.eq(e.event_id)]
        assert member.event_member.all()
        assert member.anomaly_score.max() == e.peak_score
        assert member.timestamp_utc.min() == e.start_utc
        assert member.bin_end_utc.max() == e.end_utc
        assert e.detected_at_utc == e.start_utc + pd.Timedelta(minutes=2)
        assert e.flagged_points == member.anomaly_flag.sum()
    assert ((frame.event_member) & frame.anomaly_flag.eq(0)).any()
    write_table(tmp_path / "empty.parquet", [], "events")
    assert pq.read_schema(tmp_path / "empty.parquet") == schema("events")
    assert pd.read_parquet(tmp_path / "empty.parquet").empty


def test_future_perturbation_and_chunk_equivalence():
    from collections import deque
    index = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
    frame = pd.DataFrame({"actual": np.arange(30.), "usable": True}, index=index)
    fake = FakeForecaster()
    q = _forecast_day(frame, fake, deque(maxlen=4), 4, 3)
    changed = frame.copy()
    changed.loc[index[20]:, "actual"] = 999
    other = _forecast_day(changed, fake, deque(maxlen=4), 4, 3)
    np.testing.assert_equal(q[:20], other[:20])
    history = deque(maxlen=4)
    chunked = np.concatenate([_forecast_day(frame.iloc[:13], fake, history, 4, 3),
                              _forecast_day(frame.iloc[13:], fake, history, 4, 3)])
    np.testing.assert_equal(q, chunked)
    frame.loc[index[13], "usable"] = False
    broken = _forecast_day(frame, fake, deque(maxlen=4), 4, 3)
    assert np.isnan(broken[13:18]).all()


def test_approved_calibration_and_support(staged, tmp_path):
    cfg = copy.deepcopy(staged[0])
    cfg["annotations"] = None
    with pytest.raises(ValueError, match="approved reviewed-normal"):
        detect_derconnect(cfg, "chronos2:fake", tmp_path / "no-review", forecaster=FakeForecaster())
    cfg["calibration"]["exploratory"] = True
    cfg["roles"]["calibration"]["end"] = "2026-01-01T22:30:00Z"
    with pytest.raises(ValueError, match="insufficient calibration"):
        detect_derconnect(cfg, "chronos2:fake", tmp_path / "no-support", forecaster=FakeForecaster())


def test_unlabelled_partial_and_full_metrics(completed):
    cfg, run, manifest = completed
    no_labels = evaluate_files(run, manifest["prediction_parts"], [])
    for row in no_labels["results"]:
        assert row["point_fpr"] is None and row["event_f1"] is None
        assert row["point_counts_reviewed_only"] == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        assert row["unavailable_reason"] == "no_approved_labels"
    partial = evaluate_files(run, manifest["prediction_parts"], load_annotations(cfg["annotations"]))
    for row in partial["results"]:
        assert row["event_recall"] is None and row["point_recall"] is None
        assert row["reviewed_normal_seconds"] == 1200
    anns = []
    for meter in cfg["meters"]:
        for start, end, label in [("23:30", "23:58", "reviewed_normal"),
                                   ("23:58", "23:59", "anomaly")]:
            anns.append({"meter_id": meter, "signal": "total_active_power", "start": utc(f"2026-01-01T{start}:00Z"),
                         "end": utc(f"2026-01-01T{end}:00Z"), "label": label,
                         "label_source": "full", "review_status": "approved", "review_block": "block"})
    full = evaluate_files(run, manifest["prediction_parts"], anns)
    for row in full["results"]:
        assert row["event_recall"] == 0  # crossing the reviewed block boundary excludes this alert
        assert row["boundary_or_unknown_events_excluded"] >= 1
        assert row["event_precision"] is None


def test_event_matching_is_one_to_one():
    t = utc("2026-01-01T00:00:00Z")
    es = [SimpleNamespace(start_utc=t, end_utc=t + pd.Timedelta(minutes=4)),
          SimpleNamespace(start_utc=t, end_utc=t + pd.Timedelta(minutes=1))]
    truths = [{"start": t, "end": t + pd.Timedelta(minutes=1)},
              {"start": t + pd.Timedelta(minutes=2), "end": t + pd.Timedelta(minutes=3)}]
    assert len(_match(es, truths)) == 2


def test_csv_export_hourly_and_checksum(completed, tmp_path):
    _, run, manifest = completed
    out = tmp_path / "export"
    export_bundle(run, out)
    for name in manifest["prediction_parts"]:
        original = pd.read_parquet(run / name)
        exported = pd.read_parquet(out / name)
        pd.testing.assert_frame_equal(original, exported)
        csv_path = (out / name).with_suffix(".csv")
        raw = csv_path.read_text()
        assert "T" in raw and ".000000Z" in raw
        csv = pd.read_csv(csv_path, dtype={"anomaly_flag": "Int64"})
        assert csv.anomaly_flag.isna().equals(original.anomaly_flag.isna())
    hourly = pd.read_parquet(out / "hourly-summary.parquet")
    assert "q01" not in hourly and "q99" not in hourly
    for meter, group in hourly.groupby("meter_id"):
        assert group.iloc[0].has_anomaly and group.iloc[1].has_anomaly
        assert group.event_count.sum() == 1
    empty = points(run, manifest).copy()
    empty["anomaly_flag"] = np.nan
    empty["actual"] = np.nan
    empty["q50"] = np.nan
    empty["prediction_status"] = "insufficient_coverage"
    rows = hourly_summary(empty, pd.read_parquet(run / "events.parquet").iloc[:0])
    assert all(r["has_anomaly"] is None and r["actual_power_mean"] is None for r in rows)
    (out / "events.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(out)


def test_csv_microsecond_null_roundtrip(tmp_path):
    row = {key: None for key in FIELDS["predictions"]}
    row["timestamp_utc"] = utc("2026-01-01T00:00:00.123456Z")
    row["anomaly_flag"] = None
    frame = pd.DataFrame([row])
    write_csv(tmp_path / "roundtrip.csv", frame, "predictions")
    result = pd.read_csv(tmp_path / "roundtrip.csv")
    assert result.timestamp_utc[0] == "2026-01-01T00:00:00.123456Z"
    assert pd.isna(result.anomaly_flag[0])


def test_deterministic_resume_and_mutation(staged, tmp_path, monkeypatch):
    import sarsim_fm.anomaly.derconnect as module
    cfg = copy.deepcopy(staged[0])
    cfg["meters"] = ["E2451"]
    run = tmp_path / "interrupted"
    original = module._forecast_day
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_forecast_day", interrupt)
    with pytest.raises(RuntimeError):
        detect_derconnect(cfg, "chronos2:fake", run, forecaster=FakeForecaster())
    assert not (run / "run-manifest.json").exists()
    monkeypatch.setattr(module, "_forecast_day", original)
    resumed = detect_derconnect(cfg, "chronos2:fake", run, forecaster=FakeForecaster(), resume=True)
    direct = detect_derconnect(cfg, "chronos2:fake", tmp_path / "direct", forecaster=FakeForecaster())
    assert direct["run_id"] == resumed["run_id"]
    assert direct["outputs"] == resumed["outputs"]
    verify_bundle(run)
    changed = copy.deepcopy(cfg)
    changed["detection"]["context_length"] = 5
    with pytest.raises(ValueError, match="identity mismatch"):
        detect_derconnect(changed, "chronos2:fake", run, forecaster=FakeForecaster(), resume=True)
    with pytest.raises(FileExistsError):
        detect_derconnect(cfg, "chronos2:fake", run, forecaster=FakeForecaster())
    fake = FakeForecaster()
    fake.derconnect_identity = {"version": 2}
    with pytest.raises(ValueError, match="identity mismatch"):
        detect_derconnect(cfg, "chronos2:fake", run, forecaster=fake, resume=True)


@pytest.mark.parametrize("edit", ["typo", "overlap", "naive_date", "duration", "fpr", "unit", "locked"])
def test_config_rejections(staged, edit):
    cfg = copy.deepcopy(staged[0])
    if edit == "typo":
        cfg["detection"]["contex_length"] = 3
    elif edit == "overlap":
        cfg["roles"]["scale"]["end"] = cfg["roles"]["calibration"]["end"]
    elif edit == "naive_date":
        cfg["roles"]["scale"]["start"] = "2026-01-01T22:00:00"
    elif edit == "duration":
        cfg["detection"]["min_run_seconds"] = 61
    elif edit == "fpr":
        cfg["calibration"]["target_fpr"] = .1
    elif edit == "unit":
        cfg["preparation"]["phase_units"][1] = "kW"
    elif edit == "locked":
        cfg["locked"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_annotations_conflicts_and_unknown(tmp_path):
    a = {"meter_id": "E2451", "signal": "total_active_power", "start": "2026-01-01T00:00:00Z",
         "end": "2026-01-01T01:00:00Z", "label": "anomaly", "label_source": "expert", "review_status": "approved"}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps([a, {**a, "label": "reviewed_normal"}]))
    with pytest.raises(ValueError, match="conflicting"):
        load_annotations(path)
    assert load_annotations() == []


def test_daily_lag_and_dry_run(staged, capsys):
    cfg = copy.deepcopy(staged[0])
    with pytest.raises(ValueError, match="true seasonal lag"):
        dry_run(cfg, "seasonal_naive:1440")
    cfg["detection"]["context_length"] = 1440
    assert dry_run(cfg, "seasonal_naive:1440")["context_minutes"] == 1440
    from sarsim_fm.models.naive import SeasonalNaive
    values = np.arange(1440.)
    result = SeasonalNaive(1440).predict_quantiles([values], 1, [.1, .5, .9], 1, [1440])
    assert result[0, 0, 1] == 0
    main(["detect-derconnect", str(CONFIG), "--model", "seasonal_naive:1", "--dry-run"])
    assert '"weights_loaded": false' in capsys.readouterr().out
    assert derconnect_model_identity("seasonal_naive:1", SeasonalNaive(1))["origin"] == "baseline"


def test_review_and_injection_reproducibility(completed, tmp_path):
    cfg, run, manifest = completed
    before = {n: file_hash(Path(cfg["staged"]) / n) for n in
              json.loads((Path(cfg["staged"]) / "stage-manifest.json").read_text())["prepared_parts"]}
    one = make_injections(cfg, tmp_path / "injection1")
    two = make_injections(cfg, tmp_path / "injection2")
    assert one == two
    scenario = one[0]["scenario_id"]
    assert file_hash(tmp_path / "injection1" / f"{scenario}.parquet") == file_hash(tmp_path / "injection2" / f"{scenario}.parquet")
    for name, digest in before.items():
        assert file_hash(Path(cfg["staged"]) / name) == digest
    first = review_bundle(run, tmp_path / "review1")
    second = review_bundle(run, tmp_path / "review2")
    assert first == second
    assert any(r["selection_reason"] == "random_non_alert" for r in first)
    assert all(r["category"] == "uncertain" and r["review_status"] == "pending" for r in first)
    assert verify_bundle(run)["run_id"] == manifest["run_id"]


def test_contract_matches_machine_schema():
    contract = json.loads((CONFIG.parents[1].parent / "schemas/derconnect/v1/contract.json").read_text())
    for name, fields in FIELDS.items():
        assert contract["tables"][name]["columns"] == {
            k: {"type": v, "nullable": k in NULLABLE[name]} for k, v in fields.items()}


def test_fully_reviewed_delay_in_seconds(completed):
    cfg, run, manifest = completed
    anns = []
    for meter in cfg["meters"]:
        for start, end, label in [("2026-01-01T23:30:00Z", "2026-01-01T23:58:00Z", "reviewed_normal"),
                                   ("2026-01-01T23:58:00Z", "2026-01-02T00:05:00Z", "anomaly"),
                                   ("2026-01-02T00:05:00Z", "2026-01-02T00:15:00Z", "reviewed_normal")]:
            anns.append({"meter_id": meter, "signal": "total_active_power", "start": utc(start),
                         "end": utc(end), "label": label, "label_source": "complete",
                         "review_status": "approved", "review_block": "full"})
    for result in evaluate_files(run, manifest["prediction_parts"], anns)["results"]:
        assert result["event_precision"] == result["event_recall"] == result["event_f1"] == 1
        assert result["mean_delay_seconds"] == 120
        assert result["fully_reviewed_seconds"] == 45 * 60


def test_zero_threshold_and_no_events(staged, tmp_path):
    class WideFake(FakeForecaster):
        def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
            return np.tile([0., 10., 100., 200., 1000.], (len(contexts), 1, 1))
    cfg = copy.deepcopy(staged[0])
    cfg["meters"] = ["E2451"]
    cfg["annotations"] = None
    cfg["calibration"]["exploratory"] = True
    result = detect_derconnect(cfg, "chronos2:wide-fake", tmp_path / "run", forecaster=WideFake())
    assert result["calibration"]["E2451"]["threshold"] == 0
    assert result["calibration"]["E2451"]["baseline_review_status"] == "unverified"
    assert pd.read_parquet(tmp_path / "run/events.parquet").empty
    frame = points(tmp_path / "run", result)
    assert frame.loc[frame.prediction_status == "scored", "anomaly_flag"].eq(0).all()
    metrics = json.loads((tmp_path / "run/metrics.json").read_text())
    assert metrics["results"][0]["point_fpr"] is None


def test_model_identity_shards_and_local_changes(tmp_path, monkeypatch):
    import huggingface_hub
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "shard1.safetensors", "b": "shard2.safetensors"}}))
    for name in ("shard1.safetensors", "shard2.safetensors"):
        (tmp_path / name).write_bytes(name.encode())
    def cached(repo, name, revision):
        assert revision == "resolved-commit"
        path = tmp_path / name
        return str(path) if path.exists() else None
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-commit"))
    for fake in (SimpleNamespace(model=model), SimpleNamespace(pipeline=SimpleNamespace(model=model))):
        result = derconnect_model_identity("patchtst_fm:owner/model", fake)
        assert result["resolved_revision"] == "resolved-commit"
        assert set(result["files"]) == {"model.safetensors.index.json", "shard1.safetensors", "shard2.safetensors"}
    old = derconnect_model_identity(f"patchtst_fm:{tmp_path}", None)
    (tmp_path / "shard1.safetensors").write_bytes(b"changed")
    new = derconnect_model_identity(f"patchtst_fm:{tmp_path}", None)
    assert old["sha256"] != new["sha256"]


def test_staged_and_code_mutations_invalidate(staged, tmp_path, monkeypatch):
    import shutil

    import sarsim_fm.anomaly.derconnect as module
    cfg = copy.deepcopy(staged[0])
    cfg["meters"] = ["E2451"]
    run = tmp_path / "run"
    detect_derconnect(cfg, "chronos2:fake", run, forecaster=FakeForecaster())
    original = module.environment
    def changed_environment():
        value = original()
        value["source_sha256"] = "different-code"
        return value
    monkeypatch.setattr(module, "environment", changed_environment)
    with pytest.raises(ValueError, match="identity mismatch"):
        detect_derconnect(cfg, "chronos2:fake", run, forecaster=FakeForecaster(), resume=True)
    monkeypatch.setattr(module, "environment", original)
    shutil.copytree(cfg["staged"], tmp_path / "mutated-stage")
    cfg["staged"] = str(tmp_path / "mutated-stage")
    manifest = json.loads((Path(cfg["staged"]) / "stage-manifest.json").read_text())
    (Path(cfg["staged"]) / manifest["prepared_parts"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="staged hash mismatch"):
        detect_derconnect(cfg, "chronos2:fake", tmp_path / "other", forecaster=FakeForecaster())


def test_streaming_events_never_merge_across_invalid_bins():
    import sqlite3

    from sarsim_fm.anomaly.derconnect import _events
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE points (meter TEXT,t TEXT,score REAL,threshold REAL,flag INTEGER,event_id TEXT)")
    start = utc("2026-01-01T00:00:00Z")
    flags = [1, 1, 0, 1, 1, None, 1, 1, 0, 0, 1]  # last isolated flag is discarded
    for i, flag in enumerate(flags):
        conn.execute("INSERT INTO points VALUES(?,?,?,?,?,NULL)",
                     ("E2451", (start + pd.Timedelta(minutes=i)).isoformat(),
                      float(flag or 0) * 2 if flag is not None else None,
                      0. if flag is not None else None, flag))
    identity = {"schema_version": "derconnect/v1", "run_id": "run", "detector_id": "d",
                "model_name": "fake", "meter_id": "E2451", "signal": "total_active_power"}
    events = _events(conn, identity, 2, 6)
    assert len(events) == 2
    assert events[0]["start_utc"] == start and events[0]["end_utc"] == start + pd.Timedelta(minutes=5)
    assert events[0]["detected_at_utc"] == start + pd.Timedelta(minutes=2)
    assert events[0]["peak_timestamp_utc"] == start
    assert events[1]["start_utc"] == start + pd.Timedelta(minutes=6)
    assert conn.execute("SELECT event_id FROM points WHERE flag IS NULL").fetchone()[0] is None
    conn.close()
