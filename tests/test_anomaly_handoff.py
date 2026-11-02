import hashlib
import json
import os
import zipfile

import pytest

from sarsim_fm.anomaly.handoff import verify_handoff
from sarsim_fm.anomaly.provenance import checkpoint_identity


def test_verify_handoff_detects_modified_files_and_traversal(tmp_path):
    content = b"frozen settings"
    path = tmp_path / "selection.json"
    path.write_bytes(content)
    manifest = {"source_sha256": "source", "files": {
        "selection.json": {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}
    (tmp_path / "BUNDLE-MANIFEST.json").write_text(json.dumps(manifest))
    assert verify_handoff(tmp_path)["verified_files"] == 1
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        verify_handoff(tmp_path)
    manifest["files"] = {"../outside": {}}
    (tmp_path / "BUNDLE-MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        verify_handoff(tmp_path)


def test_pinned_checkpoint_provenance_and_download_cache(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"test fixture; not model weights")
    (tmp_path / "pretrained-provenance.json").write_text(json.dumps({
        "repository": "amazon/chronos-2", "revision": "fixed-revision",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}))
    spec = f"chronos2:{tmp_path}"
    identity = checkpoint_identity(spec)
    assert identity["origin"] == "pretrained"
    assert identity["resolved_revision"] == "fixed-revision"
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download-metadata").write_text("timestamp changed")
    assert checkpoint_identity(spec)["sha256"] == identity["sha256"]
    weights.write_bytes(b"different")
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint_identity(spec)


def test_environment_supports_source_bundle_without_git(tmp_path, monkeypatch):
    from sarsim_fm.anomaly import provenance

    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "SOURCE-PROVENANCE.json").write_text(json.dumps({
        "git_sha": "original-commit", "uncommitted_files": ["?? src/new.py"]}))
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    result = provenance.environment()
    assert result["git_sha"] == "original-commit"
    assert result["uncommitted_files"] == ["?? src/new.py"]


def test_source_snapshot_handles_reproducible_archive_timestamps(tmp_path, monkeypatch):
    from sarsim_fm.anomaly import provenance

    source = tmp_path / "src"
    source.mkdir()
    files = [source / "module.py", tmp_path / "pyproject.toml", tmp_path / "uv.lock"]
    for path in files:
        path.write_text("# fixture\n")
        os.utime(path, (0, 0))
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    snapshot = tmp_path / "snapshot.zip"
    provenance.save_source_snapshot(snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        assert len(archive.infolist()) == 3
        assert all(entry.date_time[0] >= 1980 for entry in archive.infolist())
