"""Handoff generation only: no kubectl calls, model loads, or real data."""
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from sarsim_fm.anomaly.provenance import file_hash

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/derconnect/prepare_nrp_pilot.py"
spec = importlib.util.spec_from_file_location("prepare_nrp_pilot", SCRIPT)
nrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nrp)


def test_generated_pilot_and_cluster_resources(tmp_path):
    out = tmp_path / "ali-pilot"
    nrp.initialize(out, "/data/ali/pilot", "team", "ali", "chronos2:amazon/chronos-2", "E2451")
    config = yaml.safe_load((out / "pilot.yaml").read_text())
    assert config["staged"] == "/data/ali/pilot/staged"
    assert config["calibration"]["exploratory"] and config["annotations"] is None
    assert config["roles"]["evaluation"]["start"] == "2026-03-01T00:00:00Z"
    assert "development" not in config["roles"] and "injections" not in config
    local = yaml.safe_load((out / "local-config.yaml").read_text())
    assert local["staged"] == str(out / "staged")
    pod = yaml.safe_load((out / "transfer-pod.yaml").read_text())
    assert "nvidia.com/gpu" not in pod["spec"]["containers"][0]["resources"]["requests"]
    job = yaml.safe_load((out / "inference-job.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    script = container["command"][-1]
    assert "sfm check" not in script and "sfm train" not in script and "sleep" not in script
    assert "export-derconnect" in script and "checksum mismatch" in script
    with pytest.raises(FileExistsError):
        nrp.initialize(out, "/data/ali/pilot", "team", "ali", "chronos2:amazon/chronos-2", "E2451")


def test_pack_limits_uploads_to_prepared_data_and_detects_mutations(tmp_path):
    out = tmp_path / "ali-pilot"
    nrp.initialize(out, "/data/ali/pilot", "team", "ali", "chronos2:amazon/chronos-2", "E2451")
    staged = out / "staged"
    part = staged / "prepared/E2451/2026-01-01.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"prepared-fixture")
    (staged / "raw-private.csv").write_text("never upload")
    manifest = {"complete": True, "ranges": {"E2451": {}},
                "prepared_parts": ["prepared/E2451/2026-01-01.parquet"],
                "prepared_hashes": {"prepared/E2451/2026-01-01.parquet": file_hash(part)}}
    (staged / "stage-manifest.json").write_text(json.dumps(manifest))
    part.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="partition"):
        nrp.pack(out)
    part.write_bytes(b"prepared-fixture")
    nrp.pack(out)
    with tarfile.open(out / "prepared.tar.gz") as archive:
        assert set(archive.getnames()) == {"stage-manifest.json", "prepared/E2451/2026-01-01.parquet"}
    with tarfile.open(out / "code.tar.gz") as archive:
        assert "src/sarsim_fm/anomaly/derconnect.py" in archive.getnames()
        assert not any("__pycache__" in name or name.startswith("outputs/") for name in archive.getnames())
    hashes = json.loads((out / "upload-sha256.json").read_text())
    assert all(file_hash(out / name) == value for name, value in hashes.items())
    with pytest.raises(FileExistsError):
        nrp.pack(out)


def test_retry_reuses_prepared_data_without_raw_files(tmp_path):
    previous = tmp_path / "ali-original"
    nrp.initialize(previous, "/data/ali/original", "team", "ali", "chronos2:amazon/chronos-2", "E2451")
    config = yaml.safe_load((previous / "local-config.yaml").read_text())
    staged = previous / "staged"
    part = staged / "prepared/E2451/2026-01-01.parquet"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"prepared-fixture")
    manifest = {"complete": True, "ranges": {"E2451": {}},
                "preparation": config["preparation"],
                "prepared_parts": ["prepared/E2451/2026-01-01.parquet"],
                "prepared_hashes": {"prepared/E2451/2026-01-01.parquet": file_hash(part)}}
    original = json.dumps(manifest)
    (staged / "stage-manifest.json").write_text(original)
    out = tmp_path / "ali-retry"
    nrp.retry(previous, out)
    assert (out / "upload-sha256.json").exists()
    assert (out / "staged/prepared/E2451/2026-01-01.parquet").read_bytes() == part.read_bytes()
    assert (staged / "stage-manifest.json").read_text() == original
    config = yaml.safe_load((out / "pilot.yaml").read_text())
    assert config["staged"] == "/data/ali/ali-retry/staged"
    job = yaml.safe_load((out / "inference-job.yaml").read_text())
    assert job["metadata"]["name"] == "ali-retry"
