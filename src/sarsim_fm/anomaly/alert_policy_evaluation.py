"""Compare raw, grouped and policy alerts without changing reviewed coverage."""
from pathlib import Path

import pandas as pd

from sarsim_fm.anomaly.derconnect_evaluation import evaluate_files
from sarsim_fm.anomaly.provenance import file_hash, object_hash
from sarsim_fm.data.annotations import load_annotations


def evaluate_policy(reader, suppressed, annotation_path):
    annotations = load_annotations(annotation_path)
    source, parts = reader.root, reader.parts
    events = pd.read_parquet(source / "events.parquet")

    def grouped(frame):
        frame.loc[frame.anomaly_flag.notna() & ~frame.event_member, "anomaly_flag"] = 0
        return frame

    def adjusted(frame):
        frame = grouped(frame)
        frame.loc[frame.anomaly_flag.notna() & frame.event_id.isin(suppressed), "anomaly_flag"] = 0
        return frame

    results = {
        "raw_detector": evaluate_files(source, parts, annotations),
        "retained_events_before_policy": evaluate_files(source, parts, annotations, point_transform=grouped),
        "retained_events_after_policy": evaluate_files(
            source, parts, annotations, events_override=events[~events.event_id.isin(suppressed)],
            point_transform=adjusted),
    }
    # Denominators and truths are supplied independently of suppression. Keep this an invariant.
    before = results["retained_events_before_policy"]["results"]
    after = results["retained_events_after_policy"]["results"]
    for a, b in zip(before, after, strict=True):
        for key in ("eligible_seconds", "reviewed_normal_seconds", "reviewed_anomaly_seconds",
                    "fully_reviewed_seconds", "evaluated_true_events"):
            if a[key] != b[key]:
                raise ValueError(f"policy changed evaluation denominator: {key}")
    identity = {"source_run_id": reader.manifest["run_id"], "suppressed_event_ids": sorted(suppressed),
                "annotations_sha256": file_hash(Path(annotation_path))}
    return {"evaluation_id": object_hash(identity), **identity, "layers": results,
            "note": "Suppressed true anomalies remain in recall denominators; unknown regions stay unknown."}
