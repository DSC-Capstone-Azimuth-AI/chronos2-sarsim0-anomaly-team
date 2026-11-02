"""Portable run provenance and deterministic file/directory digests."""

import hashlib
import json
import platform
import subprocess
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    provenance_file = ROOT / "SOURCE-PROVENANCE.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else {}

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            if args == ("rev-parse", "HEAD"):
                return provenance.get("git_sha")
            return "\n".join(provenance.get("uncommitted_files", []))

    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "torch", "chronos-forecasting", "transformers", "sarsim0"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    source_files = sorted((ROOT / "src").rglob("*.py"))
    source_files += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files}
    return {"timestamp": datetime.now(UTC).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages, "git_sha": git("rev-parse", "HEAD"),
            "uncommitted_files": git("status", "--porcelain").splitlines(),
            "handoff_provenance": provenance or None,
            "source_sha256": object_hash(hashes), "source_files": hashes}


def save_source_snapshot(path):
    """Keep dirty and untracked implementation files reproducible without Git commits."""
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").rglob("*.yaml"))
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    if (ROOT / "SOURCE-PROVENANCE.json").exists():
        files.append(ROOT / "SOURCE-PROVENANCE.json")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         strict_timestamps=False) as archive:
        for source in files:
            archive.write(source, str(source.relative_to(ROOT)))


def checkpoint_identity(spec, forecaster=None):
    _, _, location = spec.partition(":")
    record = {"spec": spec, "path": location or None, "sha256": None,
              "origin": "baseline" if not location else "unverified_checkpoint"}
    if location == "amazon/chronos-2":
        record["origin"] = "pretrained"
    if location and Path(location).is_dir():
        files = sorted(p for p in Path(location).rglob("*")
                       if p.is_file() and ".cache" not in p.relative_to(location).parts)
        record["files"] = {str(p.relative_to(location)): file_hash(p) for p in files}
        record["sha256"] = object_hash(record["files"])
        record["path"] = str(Path(location).resolve())
        training = Path(location).parent / "run-config.json"
        if training.exists():
            init = json.loads(training.read_text()).get("config", {}).get("model", {}).get("init")
            record["origin"] = "random_initialized" if init == "scratch" else "fine_tuned"
        pinned = Path(location) / "pretrained-provenance.json"
        if pinned.exists():
            source = json.loads(pinned.read_text())
            weights_hash = file_hash(Path(location) / "model.safetensors")
            if source["weights_sha256"] != weights_hash:
                raise ValueError("pinned pretrained checkpoint hash mismatch")
            record.update(origin="pretrained", source_repository=source["repository"],
                          resolved_revision=source["revision"], weights_sha256=weights_hash)
    elif forecaster is not None and hasattr(forecaster, "pipeline"):
        # HF stores the resolved immutable repository commit in the loaded config.
        config = forecaster.pipeline.model.config
        revision = getattr(config, "_commit_hash", None)
        from huggingface_hub import try_to_load_from_cache

        weights = try_to_load_from_cache(location, "model.safetensors", revision=revision)
        if isinstance(weights, str):
            record["resolved_revision"] = Path(weights).parent.name
            record["sha256"] = file_hash(weights)
        else:
            raise ValueError("could not identify cached checkpoint weights for reproducible inference")
    return record
