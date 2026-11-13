"""Verified, bounded reads of completed prediction artifacts."""
import json
from collections import OrderedDict
from itertools import islice
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sarsim_fm.anomaly.export import verify_bundle
from sarsim_fm.anomaly.provenance import file_hash, object_hash
from sarsim_fm.data.annotations import utc

GROUP = ["run_id", "detector_id", "meter_id", "signal"]


class BundleReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = verify_bundle(self.root)
        self.parts = self.manifest["prediction_parts"]
        self.cache = OrderedDict()
        self.ranges = []
        for name in self.parts:
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("partition outside source bundle")
            pf = pq.ParquetFile(path)
            if not pf.metadata.num_rows:
                continue
            # Catalog and cache are bounded even if a consumer provides large partitions.
            for i, batch in enumerate(pf.iter_batches(columns=["timestamp_utc"], batch_size=4096)):
                times = batch.to_pandas().timestamp_utc
                if times.isna().any():
                    raise ValueError("null source timestamp")
                self.ranges.append((name, i, times.min(), times.max() + pd.Timedelta(minutes=1)))
        if not self.ranges:
            raise ValueError("source bundle has no prediction grid")
        self.start = min(r[2] for r in self.ranges)
        self.end = max(r[3] for r in self.ranges)

    def frames(self):
        for name in self.parts:
            for i, batch in enumerate(pq.ParquetFile(self.root / name).iter_batches(batch_size=4096)):
                frame = batch.to_pandas()
                if frame.duplicated(GROUP + ["timestamp_utc"]).any():
                    raise ValueError("duplicate source prediction keys")
                yield name, i, frame

    def window(self, meter, signal, detector, start, end):
        start, end = utc(start), utc(end)
        frames = []
        for name, index, lo, hi in self.ranges:
            if lo >= end or hi <= start:
                continue
            key = (name, index)
            if key not in self.cache:
                batches = pq.ParquetFile(self.root / name).iter_batches(batch_size=4096)
                self.cache[key] = next(islice(batches, index, index + 1)).to_pandas()
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            frame = self.cache[key]
            mask = ((frame.meter_id == meter) & (frame.signal == signal) &
                    (frame.detector_id == detector) & (frame.timestamp_utc >= start) &
                    (frame.timestamp_utc < end))
            frames.append(frame.loc[mask])
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames).sort_values("timestamp_utc").reset_index(drop=True)
        if frame.duplicated(GROUP + ["timestamp_utc"]).any():
            raise ValueError("duplicate source keys across partitions")
        return frame

    def source_identity(self):
        return {"source_run_id": self.manifest["run_id"],
                "source_manifest_sha256": file_hash(self.root / "run-manifest.json"),
                "source_output_hashes": self.manifest["outputs"]}


def read_evidence(ref, base, *, required=False):
    if ref is None:
        if required:
            raise ValueError("missing required evidence")
        return None
    path = Path(ref["path"])
    path = path if path.is_absolute() else Path(base) / path
    if not path.is_file() or file_hash(path) != ref["sha256"]:
        raise ValueError(f"evidence checksum mismatch: {path.name}")
    return json.loads(path.read_text())


def portable_config(value):
    """Evidence hashes define identity; machine-specific file paths do not."""
    if isinstance(value, dict):
        return {k: portable_config(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [portable_config(v) for v in value]
    return value


def ensure_new_output(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    return out


def evidence_id(value):
    return object_hash(portable_config(value))
