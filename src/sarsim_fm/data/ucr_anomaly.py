"""Read-only UCR/HexagonML staging with explicit source annotation conventions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sarsim_fm.anomaly.provenance import file_hash

PATTERN = re.compile(r"^(\d+)_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


@dataclass(frozen=True)
class UCRSeriesMetadata:
    dataset_id: int
    name: str
    train_end: int
    anomaly_start: int
    anomaly_end: int


def parse_filename(path: str | Path) -> UCRSeriesMetadata:
    match = PATTERN.fullmatch(Path(path).name)
    if not match:
        raise ValueError(f"invalid UCR filename: {Path(path).name}")
    ident, name, train, start, end = match.groups()
    return UCRSeriesMetadata(int(ident), name, int(train), int(start), int(end))


def detector_interval(meta: UCRSeriesMetadata) -> tuple[int, int]:
    """The single source-to-detector conversion: [a,b] one-based -> [a-1,b)."""
    return meta.anomaly_start - 1, meta.anomaly_end


def load_values(path: Path, meta: UCRSeriesMetadata | None = None) -> np.ndarray:
    path = Path(path)
    # split handles arbitrary whitespace; float() rejects unparsed trailing tokens.
    # A row-vector stored on one tab-separated line is still a univariate sequence.
    try:
        values = np.asarray([float(token) for token in path.read_text().split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path.name}: expected whitespace-separated numeric values") from exc
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: expected finite nonempty one-dimensional values")
    meta = meta or parse_filename(path)
    if not 0 < meta.train_end < meta.anomaly_start <= meta.anomaly_end <= len(values):
        raise ValueError(f"{path.name}: invalid index ordering for length {len(values)}")
    return values


def mnemonic_family(name: str) -> str:
    """Conservative mnemonic grouping, preserving digits internal to source names.

    Remove distortion/noise variants and terminal replicate numbers; keep an
    auditable family string in the manifest (not a domain classification).
    """
    family = re.sub(r"^(?:(?:DISTORTED\d*|NOISE\d*))", "", name, flags=re.IGNORECASE)
    family = re.sub(r"\d+$", "", family)
    return family.lower()


def stage_ucr(source: Path, out: Path) -> dict:
    """Stage metadata only; never copy or edit source data. Require the complete archive."""
    source, out = Path(source).resolve(), Path(out)
    files = sorted(source.glob("*.txt"))
    if len(files) != 250:
        raise ValueError(f"expected exactly 250 UCR files, found {len(files)} in {source}")
    parsed = [(parse_filename(path), path) for path in files]
    ids = [m.dataset_id for m, _ in parsed]
    if sorted(ids) != list(range(1, 251)):
        raise ValueError("UCR files must have unique IDs 1 through 250")
    entries = []
    for meta, path in sorted(parsed, key=lambda pair: pair[0].dataset_id):
        values = load_values(path, meta)
        entries.append({**asdict(meta), "source_path": str(path), "file_size": path.stat().st_size,
                        "sha256": file_hash(path), "series_length": len(values),
                        "clean_prefix_length": meta.train_end, "family": mnemonic_family(meta.name),
                        "annotation_convention": "one-based inclusive"})
    manifest = {"name": "ucr-anomaly", "format": "ucr", "source": str(source),
                "n_series": 250, "annotation_convention": "one-based inclusive", "series": entries}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Staged {len(entries)} UCR file references in {out}")
    return manifest


def iter_staged(manifest, subset_ids=None):
    requested = None if subset_ids is None else set(subset_ids)
    available = {entry["dataset_id"] for entry in manifest["series"]}
    if requested is not None and (not requested or requested - available):
        raise ValueError("subset_ids must be a nonempty subset of staged UCR IDs")
    for entry in manifest["series"]:
        if requested is not None and entry["dataset_id"] not in requested:
            continue
        path = Path(entry["source_path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"{path.name}: staged source hash mismatch")
        meta = UCRSeriesMetadata(**{k: entry[k] for k in UCRSeriesMetadata.__dataclass_fields__})
        values = load_values(path, meta)
        a, b = detector_interval(meta)
        yield {"item_id": f"ucr-{meta.dataset_id:03d}", "series": values,
               "clean_prefix_length": meta.train_end, "season_length": 1,
               "anomaly_start": a, "anomaly_end": b, "family": entry["family"]}
