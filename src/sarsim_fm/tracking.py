"""Optional Weights & Biases tracking. Every function is a no-op unless a run was started.

One W&B project holds every run. A run's `config` is the fully resolved YAML, so runs can be
filtered by any field (`model.type`, `data.sarsim.s_range`, `train.learning_rate`, ...).
`group` names the experiment question a run belongs to, `job_type` is train / baseline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULTS = {"project": "sarsim-fm", "entity": "chewycrunch-ucsd", "group": None, "tags": []}


def in_sweep() -> bool:
    """True when launched by `wandb agent`."""
    return "WANDB_SWEEP_ID" in os.environ


def _target(settings: dict) -> tuple[str, str]:
    return (
        os.environ.get("WANDB_ENTITY", settings["entity"]),
        os.environ.get("WANDB_PROJECT", settings["project"]),
    )


def start_run(cfg: dict, job_type: str, name: str, extra_tags: list[str] | None = None, **init_kwargs):
    """Start a W&B run if `cfg["wandb"]["enabled"]` (or under a sweep). Returns the run or None."""
    settings = {**DEFAULTS, **cfg.get("wandb", {})}
    if not (settings.get("enabled") or in_sweep()):
        return None

    import wandb

    entity, project = _target(settings)
    tags = [*settings["tags"], *(extra_tags or [])]
    return wandb.init(
        entity=entity,
        project=project,
        group=settings["group"],
        job_type=job_type,
        name=name,
        tags=sorted(set(tags)),
        config=cfg,
        **init_kwargs,
    )


def active() -> bool:
    """True if a run is open in this process (the HF Trainer then reports to it)."""
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def log_eval(dataset: str, summary: dict) -> None:
    """Put an eval set's numeric metrics in the run summary as `eval/<dataset>/<metric>`."""
    if not active():
        return
    import wandb

    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            wandb.run.summary[f"eval/{dataset}/{key}"] = value


def log_summary(**fields) -> None:
    if active():
        import wandb

        wandb.run.summary.update(fields)


def finish(exit_code: int = 0) -> None:
    if active():
        import wandb

        wandb.finish(exit_code=exit_code)


def import_run(run_dir: Path, rel_name: str, group: str | None) -> str:
    """Upload a finished `outputs/` run (config, train-log.csv curve, eval summary) as a W&B run.

    The run id is derived from the folder name, so importing twice is skipped rather than
    duplicated. Runs that were logged live (`wandb.enabled` in their config) are skipped too.
    Returns what happened, for printing.
    """
    import pandas as pd

    record_path = run_dir / "run-config.json"
    summaries = json.loads((run_dir / "eval" / "summary.json").read_text())
    if record_path.exists():
        record = json.loads(record_path.read_text())
        cfg = record["config"]
        if cfg.get("wandb", {}).get("enabled"):
            return "skipped (already logged live)"
        cfg = {**cfg, "environment": record.get("environment", {}), "config_file": record.get("config_file")}
        job_type = "train"
        tags = [cfg["model"]["type"], cfg["model"].get("init", "pretrained"), "imported"]
        timing = record.get("timing", {})
    else:
        first = next(iter(summaries.values()))
        cfg = {
            "model": {"type": first["model"].split(":", 1)[0], "spec": first["model"]},
            "data": {"context_length": first.get("context_length")},
            "runtime": {"device": first.get("device")},
        }
        job_type, tags, timing = "baseline", ["baseline", "imported"], {}
        group = "baselines"
    cfg["wandb"] = {**cfg.get("wandb", {}), "enabled": True, "group": group}

    run_id = "import-" + re.sub(r"[^A-Za-z0-9_-]+", "-", rel_name).strip("-")
    entity, project = _target({**DEFAULTS, **cfg["wandb"]})
    if os.environ.get("WANDB_MODE") != "offline":
        import wandb

        try:
            # The `name` filter matches the run id; an empty result means not imported yet.
            exists = len(wandb.Api().runs(f"{entity}/{project}", filters={"name": run_id})) > 0
        except ValueError:  # project does not exist yet
            exists = False
        if exists:
            return f"skipped (exists as {run_id})"

    run = start_run(cfg, job_type, rel_name, extra_tags=tags, id=run_id)
    try:
        log_path = run_dir / "train-log.csv"
        if log_path.exists():
            # Same keys and x-axis the HF Trainer uses for live runs.
            run.define_metric("train/global_step")
            run.define_metric("train/*", step_metric="train/global_step")
            log = pd.read_csv(log_path)
            for row in log.to_dict("records"):
                values = {
                    f"train/{k}": v for k, v in row.items()
                    if k != "step" and pd.notna(v)
                }
                if values:
                    run.log({**values, "train/global_step": int(row["step"])})
        for dataset, summary in summaries.items():
            log_eval(dataset, summary)
        log_summary(**{f"timing/{k}": v for k, v in timing.items()})
    except BaseException:
        finish(exit_code=1)
        raise
    finish()
    return f"imported as {run_id}"
