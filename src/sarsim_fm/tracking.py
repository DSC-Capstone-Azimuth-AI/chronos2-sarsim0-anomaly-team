"""Optional Weights & Biases tracking. Every function is a no-op unless a run was started.

One W&B project holds every run. A run's `config` is the fully resolved YAML, so runs can be
filtered by any field (`model.type`, `data.sarsim.s_range`, `train.learning_rate`, ...).
`group` names the experiment question a run belongs to, `job_type` is train / baseline.
"""

from __future__ import annotations

import os

DEFAULTS = {"project": "sarsim-fm", "entity": "chewycrunch-ucsd", "group": None, "tags": []}


def in_sweep() -> bool:
    """True when launched by `wandb agent`."""
    return "WANDB_SWEEP_ID" in os.environ


def start_run(cfg: dict, job_type: str, name: str, extra_tags: list[str] | None = None):
    """Start a W&B run if `cfg["wandb"]["enabled"]` (or under a sweep). Returns the run or None."""
    settings = {**DEFAULTS, **cfg.get("wandb", {})}
    if not (settings.get("enabled") or in_sweep()):
        return None

    import wandb

    tags = [*settings["tags"], *(extra_tags or [])]
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", settings["entity"]),
        project=os.environ.get("WANDB_PROJECT", settings["project"]),
        group=settings["group"],
        job_type=job_type,
        name=name,
        tags=sorted(set(tags)),
        config=cfg,
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
