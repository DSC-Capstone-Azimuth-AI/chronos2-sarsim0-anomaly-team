import json
from typing import ClassVar

import huggingface_hub
import pytest

from sarsim_fm import hub
from sarsim_fm.config import load_config


class FakeApi:
    # Class-level so the test can inspect what instances created inside hub.py recorded.
    role = "write"
    existing: ClassVar[set] = set()
    commits: ClassVar[list] = []

    def whoami(self):
        return {"name": "someone", "auth": {"accessToken": {"role": self.role}}}

    def repo_exists(self, repo_id):
        return repo_id in self.existing

    def create_repo(self, repo_id, **kwargs):
        self.existing.add(repo_id)

    def create_commit(self, repo_id, operations, commit_message):
        self.commits.append((repo_id, operations))


@pytest.fixture
def api(monkeypatch):
    FakeApi.role, FakeApi.existing, FakeApi.commits = "write", set(), []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    return FakeApi


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "chronos2-ft-test"
    (run / "checkpoint-final").mkdir(parents=True)
    (run / "checkpoint-final" / "model.safetensors").write_bytes(b"weights")
    (run / "checkpoint-final" / "config.json").write_text("{}")
    (run / "eval").mkdir()
    (run / "eval" / "summary.json").write_text(json.dumps({"sim_iid": {"mase": 0.61, "sql": 0.5}}))
    (run / "run-config.json").write_text(json.dumps({
        "config": {"name": "chronos2-ft-test", "model": {"type": "chronos2", "base": "amazon/chronos-2"},
                   "train": {"max_steps": 10, "learning_rate": 1e-5}},
        "config_file": "configs/train/x.yaml", "environment": {"git_sha": "abc123"},
    }))
    return run


def test_hub_off_by_default_and_repo_naming():
    cfg = load_config("configs/train/chronos2-ft-5k.yaml")
    settings = hub.settings(cfg)
    assert settings["enabled"] is False and settings["private"] is True
    assert hub.repo_id_for(cfg["name"], settings) == "azimuth-anomaly-detection/chronos2-ft-5k"
    assert hub.repo_id_for("x", {**settings, "repo": "me/other"}) == "me/other"


def test_push_uploads_checkpoint_at_repo_root(api, run_dir):
    url = hub.push_run(run_dir, "org/chronos2-ft-test")
    assert url == "https://huggingface.co/org/chronos2-ft-test"
    [(repo_id, ops)] = api.commits
    assert repo_id == "org/chronos2-ft-test"
    paths = {op.path_in_repo for op in ops}
    assert paths == {"model.safetensors", "config.json", "run-config.json", "eval/summary.json", "README.md"}
    card = next(op for op in ops if op.path_in_repo == "README.md").path_or_fileobj.decode()
    assert "base_model: amazon/chronos-2" in card and "abc123" in card and "0.6100" in card
    assert "sfm eval --model chronos2:org/chronos2-ft-test" in card


def test_push_refuses_existing_repo_unless_overwrite(api, run_dir):
    api.existing = {"org/chronos2-ft-test"}
    with pytest.raises(hub.HubError, match="already exists"):
        hub.push_run(run_dir, "org/chronos2-ft-test")
    assert api.commits == []
    hub.push_run(run_dir, "org/chronos2-ft-test", overwrite=True)
    assert len(api.commits) == 1


def test_push_refuses_read_only_token(api, run_dir):
    api.role = "read"
    with pytest.raises(hub.HubError, match="read-only"):
        hub.push_run(run_dir, "org/chronos2-ft-test")


def test_push_requires_finished_run(api, tmp_path):
    with pytest.raises(hub.HubError, match="model.safetensors"):
        hub.push_run(tmp_path, "org/empty")
