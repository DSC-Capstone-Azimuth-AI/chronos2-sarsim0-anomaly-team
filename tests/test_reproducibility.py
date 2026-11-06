"""Targeted regression tests for configuration validation and reproducible output manifests."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from sarsim_fm import cli
from sarsim_fm.config import load_config, set_dotted


@pytest.fixture
def train_config(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("""model: {type: chronos2, base: amazon/chronos-2, init: pretrained}
name: demo
data: {context_length: 512, prediction_length: 18, batch_size: 8, seed: 42, sarsim: {}}
train: {max_steps: 100, learning_rate: 0.00001}
runtime: {device: cpu, precision: fp32}
eval: {datasets: [sim_iid], batch_size: 4}
""")
    child = tmp_path / "child.yaml"
    child.write_text("inherit: base.yaml\ntrain: {max_steps: 500}\n")
    return child


def test_valid_inheritance_and_overrides(train_config):
    cfg = load_config(train_config, ["train.max_steps=501", "model.revision=abcdef", "data.sarsim.s_range=[10,30]"])
    assert cfg["train"]["learning_rate"] == 0.00001
    assert cfg["train"]["max_steps"] == 501
    assert cfg["model"]["revision"] == "abcdef"
    assert cfg["data"]["sarsim"]["s_range"] == [10, 30]


@pytest.mark.parametrize("override", [
    "train.max_stpes=500", "eval.datsets=[sim_iid]", "inheirt=other.yaml",
    "train.max_steps.oops=1", "unknown.branch=value", "runtime.preciison=fp32",
])
def test_unknown_override_is_rejected_and_atomic(train_config, override):
    cfg = load_config(train_config)
    snapshot = json.dumps(cfg, sort_keys=True)
    with pytest.raises(ValueError):
        set_dotted(cfg, override)
    assert json.dumps(cfg, sort_keys=True) == snapshot


def test_unknown_key_in_child_config(train_config):
    train_config.write_text("inherit: base.yaml\ntrain: {max_stpes: 500}\n")
    with pytest.raises(ValueError, match="max_stpes"):
        load_config(train_config)


def test_simulation_and_detector_config_are_supported(tmp_path):
    sim = tmp_path / "sim.yaml"
    sim.write_text("defaults: {n_series: 2, sarsim: {}}\nsplits: {test: {seed: 100, sarsim: {s_range: [5, 9]}}}\n")
    assert load_config(sim)["splits"]["test"]["seed"] == 100
    detect = tmp_path / "detector.yaml"
    detect.write_text("datasets: [example.parquet]\ndetector: {events: {min_run: 2}}\n")
    assert load_config(detect)["detector"]["events"]["min_run"] == 2
    detect.write_text("datasets: [example.parquet]\ndetector: {events: {min_ruun: 2}}\n")
    with pytest.raises(ValueError, match="min_ruun"):
        load_config(detect)


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    (tmp_path / "uv.lock").write_bytes(b"locked fixture")
    data_dir = tmp_path / "data" / "eval"
    data_dir.mkdir(parents=True)
    (data_dir / "sim_iid.parquet").write_bytes(b"synthetic dataset fixture")
    package = types.ModuleType("sarsim_fm.data")
    package.__path__ = []
    eval_sets = types.ModuleType("sarsim_fm.data.eval_sets")
    eval_sets.EVAL_DIR = data_dir
    eval_sets.eval_set_path = lambda name: data_dir / f"{name}.parquet"
    monkeypatch.setitem(sys.modules, "sarsim_fm.data", package)
    monkeypatch.setitem(sys.modules, "sarsim_fm.data.eval_sets", eval_sets)
    import sarsim_fm.runtime as runtime_module
    monkeypatch.setattr(runtime_module, "environment_info", lambda runtime: {"python": "test", "device": "cpu"})
    runtime = SimpleNamespace(device="cpu", bf16=False, tf32=False)
    monkeypatch.setattr(cli, "_runtime_from", lambda args: runtime)
    paths = []
    def fake_eval(spec, out_dir, datasets, args, runtime, meta):
        paths.append(out_dir)
        output = out_dir / "eval"
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"sim_iid": {"model": spec}}))
    monkeypatch.setattr(cli, "_run_eval", fake_eval)
    return tmp_path, paths, runtime


def _eval_args(out=None, datasets=None):
    return SimpleNamespace(
        model=["chronos2:amazon/chronos-2"], out=out, datasets=datasets or [],
        device=None, precision=None, batch_size=64, context_length=512, wandb=False,
    )


def test_baselines_never_reuse_directories_and_include_manifest(isolated_cli):
    root, paths, _ = isolated_cli
    cli.cmd_eval(_eval_args())
    cli.cmd_eval(_eval_args())
    assert len(paths) == 2 and paths[0] != paths[1]
    for path in paths:
        assert path.exists()
        assert path.parent.name == "chronos2_amazon_chronos-2"
        manifest = json.loads((path / "run-config.json").read_text())
        assert manifest["job_type"] == "baseline"
        info = manifest["reproducibility"]
        assert info["uv_lock_sha256"] == hashlib.sha256(b"locked fixture").hexdigest()
        assert info["dataset_sha256"]["sim_iid"] == hashlib.sha256(b"synthetic dataset fixture").hexdigest()
        assert info["model_revision"]["loaded_commit"] is None
        assert info["model_revision"]["verified"] is False
        assert info["settings"]["bf16"] is False


def test_explicit_output_cannot_overwrite(isolated_cli):
    root, paths, _ = isolated_cli
    output = root / "custom"
    cli.cmd_eval(_eval_args(out=str(output)))
    with pytest.raises(FileExistsError, match="already exists"):
        cli.cmd_eval(_eval_args(out=str(output)))
    assert len(paths) == 1


def test_manifest_records_missing_datasets_and_revision(isolated_cli):
    root, _, runtime = isolated_cli
    cfg = {
        "model": {"type": "chronos2", "base": "amazon/chronos-2", "revision": "a" * 40},
        "data": {"seed": 42, "num_workers": 2},
    }
    manifest = cli._run_manifest(cfg, "config.yaml", runtime, ["sim_iid", "missing"], "train")
    repro = manifest["reproducibility"]
    assert repro["dataset_sha256"]["missing"] is None
    assert repro["settings"]["seed"] == 42
    assert repro["model_revision"]["immutable_requested"] is True
    assert repro["model_revision"]["loaded_commit"] is None


def test_training_writes_expanded_manifest(isolated_cli, monkeypatch):
    root, _, runtime = isolated_cli
    models = types.ModuleType("sarsim_fm.models")
    models.get_trainer = lambda kind: lambda cfg, runtime, out_dir: out_dir / "checkpoint-final"
    monkeypatch.setitem(sys.modules, "sarsim_fm.models", models)
    cfg = {
        "name": "train-smoke",
        "model": {"type": "chronos2", "base": "amazon/chronos-2", "init": "scratch"},
        "data": {"seed": 42, "context_length": 512, "num_workers": 0},
        "eval": {"datasets": ["sim_iid"]},
    }
    args = SimpleNamespace(config="train.yaml", overwrite=False, no_eval=True)
    cli._train_and_eval(args, cfg, runtime)
    manifest = json.loads((root / "outputs" / "train-smoke" / "run-config.json").read_text())
    assert manifest["job_type"] == "train"
    assert manifest["config_file"] == "train.yaml"
    assert manifest["reproducibility"]["dataset_sha256"]["sim_iid"]
    assert manifest["reproducibility"]["model_revision"]["applicable"] is False
    assert isinstance(manifest["timing"]["train_seconds"], (int, float))
