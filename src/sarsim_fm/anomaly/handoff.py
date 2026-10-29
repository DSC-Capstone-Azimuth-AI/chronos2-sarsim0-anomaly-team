"""Verified local-to-remote handoff; never starts model training or inference."""

from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from sarsim_fm.anomaly.pipeline import _resolved, _write
from sarsim_fm.anomaly.provenance import ROOT, environment, file_hash, object_hash

COMPLETED_RUNS = (
    "development-0", "development-1", "development-2",
    "locked-synthetic-0", "locked-synthetic-1", "locked-synthetic-2",
    "ucr-smoke-0", "ucr-smoke-1", "ucr-full-0", "ucr-full-1",
)


def verify_handoff(root: Path) -> dict:
    """Verify every packaged byte before setup/restaging; extra generated files are allowed."""
    root = Path(root).resolve()
    manifest = json.loads((root / "BUNDLE-MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"unsafe bundle member: {relative}")
        if not path.is_file() or path.stat().st_size != expected["size"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"handoff file missing or modified: {relative}")
    return {"verified_files": len(manifest["files"]), "source_sha256": manifest["source_sha256"]}


def run_local_checks(root: Path) -> dict:
    """Required code checks only; deliberately excludes sfm check/train and large inference."""
    results = []
    for argv in [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        ["git", "diff", "--check"],
        ["bash", "-n", "scripts/run-anomaly-remote.sh"],
    ]:
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=root, text=True, capture_output=True,
                                   timeout=300, check=False)
        results.append({"argv": argv, "returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr,
                        "seconds": time.perf_counter() - started})
        if completed.returncode:
            raise RuntimeError(f"local verification failed: {argv}\n{completed.stdout}\n{completed.stderr}")
    return {"status": "passed", "checks": results, "training_executed": False}


def prepare_handoff(out: Path, *, include_ucr=True, overwrite=False) -> dict:
    """Package an allowlist of source, frozen settings, data, and completed evidence.

    Full UCR raw forecast caches remain local (several GB). Compact UCR evidence
    is included separately; the remote experiment can regenerate complete caches.
    No credentials, virtual environment, Git internals, or model weights are included.
    """
    out = Path(out)
    archive_path = out / "anomaly-handoff.tar.gz"
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"{archive_path} exists; choose a new output or pass --overwrite")
    selection_path = ROOT / "outputs/anomaly/synthetic-selection.json"
    selection = json.loads(selection_path.read_text())
    if (selection.get("selection_split") != "sim-anomaly-dev"
            or selection["settings_sha256"] != object_hash(selection["detector"])):
        raise ValueError("a valid frozen synthetic development selection is required")
    from sarsim_fm.config import load_config

    _resolved(load_config(ROOT / "configs/anomaly/ucr-chronos2.yaml"))
    validation = run_local_checks(ROOT)
    provenance = environment()
    summaries = {}
    for name in COMPLETED_RUNS:
        path = ROOT / "outputs/anomaly" / name / "summary.json"
        if not path.exists():
            raise ValueError(f"required local run is incomplete: {name}")
        summaries[name] = json.loads(path.read_text())
    completion = {
        "scope": "local", "status": "completed", "training_executed": False,
        "verification": validation, "environment": provenance,
        "selection_sha256": file_hash(selection_path), "selected_detector": selection["detector"],
        "runs": summaries,
        "diagnostic_plot_counts": {
            name: len(list((ROOT / "outputs/anomaly" / name / "diagnostics").glob("*.png")))
            for name in COMPLETED_RUNS},
        "remaining_remote": ["pretrained Chronos UCR smoke and full 250-file evaluation",
                             "fine-tuned checkpoint evaluation when its weights are available",
                             "full M4 forecasting regression guardrails",
                             "final cross-model UCR diagnostics and comparison",
                             "model retraining only if locked results justify it"],
        "limitations": ["Synthetic splits contain 30 series each; they are pilot-scale benchmarks",
                        "UCR seasonal-naive uses period 1 because archive files declare no season length",
                        "Clean-prefix false alarms are calibration in-sample; held-out normal metrics are separate"],
    }
    _write(ROOT / "outputs/anomaly/local-completion.json", completion)
    _write(out / "local-verification.json", validation)
    files = {}

    def add(path, relative=None):
        path = Path(path)
        if path.is_file():
            files[relative or str(path.relative_to(ROOT))] = path

    for directory, pattern in [("src", "*.py"), ("tests", "*.py"), ("configs", "*.yaml"),
                               ("docs", "*.md"), ("scripts", "*.sh")]:
        for path in sorted((ROOT / directory).rglob(pattern)):
            add(path)
    for name in ["README.md", "pyproject.toml", "uv.lock", ".gitignore", ".python-version"]:
        add(ROOT / name)
    for pattern in ["sim-*.parquet", "sim-*.manifest.json"]:
        for path in (ROOT / "data/anomaly").glob(pattern):
            add(path)
    for path in (ROOT / "data/eval").glob("sim*"):
        add(path)
    add(selection_path)
    for run in COMPLETED_RUNS:
        directory = ROOT / "outputs/anomaly" / run
        if not (directory / "summary.json").exists():
            raise ValueError(f"required local run is incomplete: {run}")
        is_full_ucr = run.startswith("ucr-full-")
        for path in directory.rglob("*"):
            if path.suffix in {".json", ".parquet", ".zip", ".png"}:
                if is_full_ucr and path.parent.name in {"forecasts", "scores"}:
                    continue
                relative = (f"reports/local-ucr/{run}/{path.relative_to(directory)}"
                            if is_full_ucr else None)
                add(path, relative)
    for directory in ["outputs/anomaly-baseline", "outputs/anomaly-pilot", "outputs/baselines"]:
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in {".json", ".parquet"}:
                add(path)
    add(ROOT / "outputs/anomaly/comparison.json", "reports/local-comparison.json")
    add(ROOT / "outputs/anomaly/local-completion.json")
    if include_ucr:
        staged = json.loads((ROOT / "data/anomaly/ucr.manifest.json").read_text())
        if staged["n_series"] != 250 or len(staged["series"]) != 250:
            raise ValueError("handoff requires the complete 250-file UCR archive")
        for entry in staged["series"]:
            path = Path(entry["source_path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError(f"UCR source changed: {path.name}")
            add(path, f"external/ucr-anomaly/{path.name}")
    # Do not transfer the absolute-path UCR manifest. Restage from verified data remotely.
    generated = {
        "SOURCE-PROVENANCE.json": json.dumps(provenance, indent=2).encode(),
        "LOCAL-VALIDATION.json": json.dumps(validation, indent=2).encode(),
    }
    manifest = {"format_version": 1, "source_sha256": provenance["source_sha256"],
                "ucr_included": include_ucr, "files": {
                    name: {"size": path.stat().st_size, "sha256": file_hash(path)}
                    for name, path in sorted(files.items())}}
    import hashlib

    for name, content in generated.items():
        manifest["files"][name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    generated["BUNDLE-MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    out.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name, path in sorted(files.items()):
            info = tarfile.TarInfo(f"anomaly-handoff/{name}")
            info.size, info.mode = path.stat().st_size, 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        for name, content in generated.items():
            info = tarfile.TarInfo(f"anomaly-handoff/{name}")
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
    temporary.replace(archive_path)
    result = {"archive": str(archive_path.resolve()), "sha256": file_hash(archive_path),
              "bytes": archive_path.stat().st_size, "files": len(manifest["files"]),
              "source_sha256": provenance["source_sha256"], "ucr_included": include_ucr,
              "training_executed": False}
    _write(out / "handoff.json", result)
    return result
