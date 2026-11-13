"""Explicit approved mapping from consolidated review worksheets to annotations."""
import json
from pathlib import Path

import pandas as pd

from sarsim_fm.anomaly.export import write_json
from sarsim_fm.anomaly.provenance import file_hash
from sarsim_fm.data.annotations import load_annotations, utc


def normalize_reviews(reviews, mapping_path, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    mapping = json.loads(Path(mapping_path).read_text())
    expected = {"status", "reviewer", "reviewed_at", "label_source", "categories"}
    if set(mapping) != expected or mapping["status"] != "approved" or not mapping["reviewer"] or not mapping["label_source"]:
        raise ValueError("truth mapping requires an explicit approved reviewer/source record")
    utc(mapping["reviewed_at"])
    allowed = {"expected_operation", "measurement_issue", "potential_operational_anomaly", "uncertain"}
    if set(mapping["categories"]) != allowed or any(v not in {"reviewed_normal", "anomaly", "unknown"}
                                                   for v in mapping["categories"].values()):
        raise ValueError("mapping requires the four supported review categories")
    if mapping["categories"]["measurement_issue"] != "unknown" or mapping["categories"]["uncertain"] != "unknown":
        raise ValueError("measurement issues and uncertainty must remain unknown demand truth")
    frame = pd.read_csv(reviews, keep_default_na=False)
    rows = []
    for row in frame.to_dict("records"):
        if row["category"] not in allowed or row["review_status"] not in {"pending", "approved"}:
            raise ValueError("unsupported review category/status")
        approved = row["review_status"] == "approved"
        if approved and not row.get("reviewer"):
            raise ValueError("approved review row requires a reviewer")
        label = mapping["categories"][row["category"]] if approved else "unknown"
        if label == "anomaly" and str(row.get("truth_confirmed", "")).lower() not in {"true", "1"}:
            label = "unknown"
        record = {"meter_id": row["meter_id"], "signal": row["signal"], "start": utc(row["start"]).isoformat(),
                  "end": utc(row["end"]).isoformat(), "label": label,
                  "label_source": mapping["label_source"], "review_status": "approved" if approved else "pending"}
        if row.get("review_block") and approved and label != "unknown":
            record["review_block"] = row["review_block"]
        rows.append(record)
    # Require human consolidation of overlapping windows rather than choosing one label silently.
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".validation.json")
    try:
        temporary.write_text(json.dumps(rows))
        load_annotations(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(out, rows)
    write_json(out.with_suffix(".provenance.json"), {"reviews_sha256": file_hash(reviews),
                                                   "mapping_sha256": file_hash(mapping_path),
                                                   "annotations_sha256": file_hash(out),
                                                   "mapping": mapping, "rows": len(rows)})
    return rows
