"""Hugging Face Hub uploads of finished runs, so teammates load trained models by repo id.

`sfm train ... --set hub.enabled=true` uploads after training and eval finish; `sfm push
outputs/<run>` uploads a finished run by hand. Both make one commit to `<org>/<run name>`:

    config.json, model.safetensors, ...   from checkpoint-final/ (loadable with from_pretrained)
    run-config.json, train-log.csv        how the weights were made (git sha, config, loss curve)
    eval/summary.json                     scores at upload time
    README.md                             model card generated from the above

Teammates then use `chronos2:<org>/<run name>` anywhere a local checkpoint path works.
An existing repo is never overwritten unless asked (`hub.overwrite` / `sfm push --force`).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"enabled": False, "org": "azimuth-anomaly-detection", "repo": None, "private": True, "overwrite": False}
RUN_FILES = ["run-config.json", "train-log.csv", "eval/summary.json"]


class HubError(RuntimeError):
    pass


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("hub") or {})}


def repo_id_for(name: str, hub: dict) -> str:
    return hub["repo"] or f"{hub['org']}/{name}"


def check_access(repo_id: str, overwrite: bool) -> None:
    """Fail before a long run, not after: logged in with write access, and no repo to clobber."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

    api = HfApi()
    try:
        role = api.whoami()["auth"].get("accessToken", {}).get("role")
    except (LocalTokenNotFoundError, HfHubHTTPError, KeyError) as exc:
        raise HubError("Not logged in to Hugging Face: run `uv run hf auth login` with a write token") from exc
    if role == "read":
        raise HubError(
            "Your Hugging Face token is read-only. Make a write token at "
            "https://huggingface.co/settings/tokens, then `uv run hf auth login` "
            "(an HF_TOKEN environment variable takes precedence over the login)."
        )
    if not overwrite and api.repo_exists(repo_id):
        raise HubError(f"{repo_id} already exists on the Hub; rename the run or pass --force / hub.overwrite=true")


def _model_card(run_dir: Path, repo_id: str) -> str:
    record = json.loads((run_dir / "run-config.json").read_text())
    cfg = record.get("config", {})
    model, train = cfg.get("model", {}), cfg.get("train", {})
    env = record.get("environment", {})
    init, base = model.get("init", "pretrained"), model.get("base")
    if not base:
        origin = ""
    elif init == "pretrained":
        origin = f", fine-tuned from `{base}`"
    else:
        origin = f", `{base}` architecture from random init"
    lines = [
        "---",
        "library_name: transformers",
        "pipeline_tag: time-series-forecasting",
        *([f"base_model: {base}"] if init == "pretrained" and base else []),
        "---",
        "",
        f"# {cfg.get('name', run_dir.name)}",
        "",
        (
            f"`{model.get('type')}` model{origin}, trained on SarSim0 for {train.get('max_steps')} steps"
            f" at learning rate {train.get('learning_rate')}."
        ),
        f"Config `{record.get('config_file')}` at git commit `{env.get('git_sha', 'unknown')}`; full record in `run-config.json`.",
        "",
        "```bash",
        f"uv run sfm eval --model {model.get('type')}:{repo_id}",
        "```",
    ]
    summary_path = run_dir / "eval" / "summary.json"
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text())
        metrics = [k for k in ("mase", "sql", "smape", "coverage_98") if any(k in s for s in summaries.values())]
        if metrics:
            lines += ["", "| eval set | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
            for name, s in summaries.items():
                cells = [f"{s[m]:.4f}" if isinstance(s.get(m), (int, float)) else "" for m in metrics]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def push_run(run_dir: Path, repo_id: str, private: bool = True, overwrite: bool = False) -> str:
    """Upload `run_dir` (checkpoint-final + run metadata) to `repo_id` in one commit. Returns the repo URL."""
    from huggingface_hub import CommitOperationAdd, HfApi

    run_dir = Path(run_dir)
    checkpoint = run_dir / "checkpoint-final"
    if not (checkpoint / "model.safetensors").exists():
        raise HubError(f"{checkpoint}/model.safetensors not found; is {run_dir} a finished training run?")
    if not (run_dir / "run-config.json").exists():
        raise HubError(f"{run_dir}/run-config.json not found")
    check_access(repo_id, overwrite)

    ops = [CommitOperationAdd(p.name, str(p)) for p in sorted(checkpoint.iterdir()) if p.is_file()]
    ops += [CommitOperationAdd(rel, str(run_dir / rel)) for rel in RUN_FILES if (run_dir / rel).exists()]
    ops.append(CommitOperationAdd("README.md", _model_card(run_dir, repo_id).encode()))

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.create_commit(repo_id, ops, commit_message=f"Upload {run_dir.name}")
    return f"https://huggingface.co/{repo_id}"
