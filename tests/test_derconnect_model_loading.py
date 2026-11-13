"""Pin actual loaded snapshots without relying on Transformers private config fields."""
from types import SimpleNamespace

import pytest

from sarsim_fm.anomaly.provenance import derconnect_model_identity
from sarsim_fm.models import LOADERS, load_derconnect_forecaster


@pytest.mark.parametrize("kind,repo", [("chronos2", "amazon/chronos-2"),
                                      ("patchtst_fm", "ibm-granite/granite-timeseries-patchtst-fm-r2")])
def test_pin_snapshot_before_loading_without_config_commit_hash(tmp_path, monkeypatch, kind, repo):
    import huggingface_hub
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"architecture": "test"}')
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls = []

    def config_download(repository, filename):
        assert repository == repo and filename == "config.json"
        calls.append("resolve")
        return str(snapshot / filename)

    def download(repository, *, revision, allow_patterns):
        assert repository == repo and revision == snapshot.name
        assert "*.safetensors" in allow_patterns and "*.json" in allow_patterns
        calls.append("download_exact_revision")
        return str(snapshot)

    fake_model = SimpleNamespace(config=SimpleNamespace())  # Same missing field as NRP.
    def loader(path, runtime, context_length):
        assert path == str(snapshot) and context_length == 256
        calls.append("load_local_snapshot")
        return (SimpleNamespace(pipeline=SimpleNamespace(model=fake_model)) if kind == "chronos2"
                else SimpleNamespace(model=fake_model))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", config_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setitem(LOADERS, kind, loader)
    forecaster = load_derconnect_forecaster(kind, None, 256)
    identity = derconnect_model_identity(kind, forecaster)
    assert calls == ["resolve", "download_exact_revision", "load_local_snapshot"]
    assert identity["source_repository"] == repo and identity["resolved_revision"] == revision
    assert set(identity["files"]) == {"config.json", "model.safetensors"}
    assert "path" not in identity  # Scratch directory names cannot affect identity.
    (snapshot / "model.safetensors").write_bytes(b"changed weights")
    assert derconnect_model_identity(kind, forecaster)["sha256"] != identity["sha256"]


def test_local_model_bypasses_hub_resolution(tmp_path, monkeypatch):
    import huggingface_hub
    def forbidden(*args, **kwargs):
        raise AssertionError("local checkpoints must not resolve Hub branches")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    expected = SimpleNamespace()
    monkeypatch.setitem(LOADERS, "chronos2", lambda path, runtime, context: expected)
    assert load_derconnect_forecaster(f"chronos2:{tmp_path}", None, 256) is expected


def test_snapshot_mismatch_rejected_before_model_load(tmp_path, monkeypatch):
    import huggingface_hub
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args: str(snapshot / "config.json"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(tmp_path / "other"))
    def forbidden(*args):
        raise AssertionError("a mismatched snapshot must never be loaded")
    monkeypatch.setitem(LOADERS, "chronos2", forbidden)
    with pytest.raises(ValueError, match="differs"):
        load_derconnect_forecaster("chronos2", None, 256)
