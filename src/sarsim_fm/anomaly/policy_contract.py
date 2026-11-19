"""Independent alert-policy overlay schemas; detector v1 files remain unchanged."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "derconnect-alert-policy/v1"
KEYS = dict.fromkeys(("schema_version", "source_run_id", "detector_id", "meter_id", "signal",
                      "policy_id", "policy_application_id"), "string")
EVENT = {**KEYS, **dict.fromkeys(("event_id", "rule_id", "candidate_rule_ids", "decision",
                                "eligibility_status", "reason_codes", "evidence_hashes"), "string"),
         **dict.fromkeys(("start_utc", "end_utc", "decision_data_cutoff_utc"), "timestamp"),
         **dict.fromkeys(("reference_actual", "relative_change_min", "relative_change_max"), "float64"),
         **dict.fromkeys(("suppressed", "would_suppress", "time_overlap", "fully_contained",
                         "direction_matches", "magnitude_matches", "qa_passed", "approved"), "bool")}
POINT = {**KEYS, "timestamp_utc": "timestamp", "event_id": "string", "raw_anomaly_flag": "int64",
         "actionable_flag": "int64", "actionable_event_member": "bool"}
HOUR = {**KEYS, "timestamp_utc": "timestamp",
        **dict.fromkeys(("scored_minutes", "raw_flagged_minutes", "retained_flagged_minutes",
                        "actionable_flagged_minutes", "raw_event_starts", "suppressed_event_starts",
                        "actionable_event_starts"), "int64"),
        "has_raw_event_overlap": "bool", "has_actionable_event_overlap": "bool"}
FIELDS = {"event-decisions": EVENT, "point-decisions": POINT, "hourly-policy-summary": HOUR}
NULLS = {"event-decisions": {"rule_id", "would_suppress", "reference_actual", "relative_change_min",
                              "relative_change_max", "direction_matches", "magnitude_matches"},
         "point-decisions": {"event_id", "raw_anomaly_flag", "actionable_flag"},
         "hourly-policy-summary": {"has_raw_event_overlap", "has_actionable_event_overlap"}}


def schema(name):
    types = {"string": pa.string(), "timestamp": pa.timestamp("us", tz="UTC"),
             "float64": pa.float64(), "bool": pa.bool_(), "int64": pa.int64()}
    return pa.schema([pa.field(k, types[v], nullable=k in NULLS[name]) for k, v in FIELDS[name].items()],
                     metadata={b"schema_version": VERSION.encode()})


def write_table(path, rows, name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    for key, kind in FIELDS[name].items():
        if key not in f:
            f[key] = pd.Series(index=f.index, dtype="object")
        if kind == "timestamp":
            f[key] = pd.to_datetime(f[key], utc=True)
        elif kind == "int64":
            f[key] = f[key].astype("Int64")
    table = pa.Table.from_pandas(f[list(FIELDS[name])], schema=schema(name), preserve_index=False)
    table.validate(full=True)
    if any(table[field.name].null_count for field in table.schema if not field.nullable):
        raise ValueError(f"null in required {name} field")
    tmp = path.with_suffix(".partial")
    pq.write_table(table, tmp)
    tmp.replace(path)
    csv = f[list(FIELDS[name])].copy()
    for k, v in FIELDS[name].items():
        if v == "timestamp":
            csv[k] = csv[k].map(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if pd.notna(x) else None)
    tmp_csv = path.with_suffix(".csv.partial")
    csv.to_csv(tmp_csv, index=False, na_rep="")
    tmp_csv.replace(path.with_suffix(".csv"))


def contract():
    keys = {"event-decisions": ["source_run_id", "detector_id", "meter_id", "signal", "event_id",
                                "policy_application_id"],
            "point-decisions": ["source_run_id", "detector_id", "meter_id", "signal", "timestamp_utc",
                                "policy_application_id"],
            "hourly-policy-summary": ["source_run_id", "detector_id", "meter_id", "signal", "timestamp_utc",
                                       "policy_application_id"]}
    return {"schema_version": VERSION, "intervals": "half-open", "tables": {
        name: {"primary_key": keys[name], "columns": {
            k: {"type": v, "nullable": k in NULLS[name]} for k, v in fields.items()}}
        for name, fields in FIELDS.items()}}
