"""YAML run configs with `inherit:` and validated `--set a.b=value` overrides."""

from __future__ import annotations

import copy
from difflib import get_close_matches
from pathlib import Path

import yaml


# Explicitly extensible fields: their keys are defined by SarSim0 / PatchTST-FM, not by
# the shared experiment config. All other configuration keys are checked recursively.
_VALUE = object()
_OPEN_MAP = object()
_MISSING = object()

_SIM_FIELDS = {
    "seed": _VALUE,
    "normal": _VALUE,
    "n_series": _VALUE,
    "series_length": _VALUE,
    "clean_prefix_length": _VALUE,
    "context_length": _VALUE,
    "prediction_length": _VALUE,
    "season_length": _VALUE,
    "duration_periods": _VALUE,
    "severity_range": _VALUE,
    "max_retries": _VALUE,
    "sarsim": _OPEN_MAP,
}
_TRAIN_SCHEMA = {
    "name": _VALUE,
    "model": {
        "type": _VALUE, "base": _VALUE, "init": _VALUE,
        "architecture": _OPEN_MAP, "revision": _VALUE,
    },
    "data": {
        "context_length": _VALUE, "prediction_length": _VALUE,
        "batch_size": _VALUE, "seed": _VALUE, "num_workers": _VALUE,
        "sarsim": _OPEN_MAP,
    },
    "train": {
        "max_steps": _VALUE, "learning_rate": _VALUE, "lr_scheduler": _VALUE,
        "warmup_steps": _VALUE, "grad_accum": _VALUE,
        "weight_decay": _VALUE, "max_grad_norm": _VALUE,
        "logging_steps": _VALUE, "save_steps": _VALUE,
    },
    "runtime": {"device": _VALUE, "precision": _VALUE},
    "eval": {"datasets": _VALUE, "batch_size": _VALUE},
    "wandb": {
        "enabled": _VALUE, "group": _VALUE, "tags": _VALUE,
        "project": _VALUE, "entity": _VALUE,
    },
    "hub": {
        "enabled": _VALUE, "org": _VALUE, "repo": _VALUE,
        "private": _VALUE, "overwrite": _VALUE,
    },
}
_SIM_SCHEMA = {"defaults": _SIM_FIELDS, "splits": {"*": _SIM_FIELDS}}
_ANOMALY_SCHEMA = {
    "name": _VALUE, "datasets": _VALUE, "locked": _VALUE,
    "selection_file": _VALUE, "subset_items": _VALUE,
    "detector": {
        "mode": _VALUE, "context_length": _VALUE, "prediction_length": _VALUE,
        "batch_size": _VALUE, "score": _VALUE, "target_fpr": _VALUE,
        "events": {
            "min_run": _VALUE, "merge_gap": _VALUE,
            "release_threshold_ratio": _VALUE,
        },
    },
}


def _schema_for(config: dict) -> dict:
    if "model" in config or "train" in config:
        return _TRAIN_SCHEMA
    if "splits" in config or "defaults" in config:
        return _SIM_SCHEMA
    if "datasets" in config or "detector" in config or "selection_file" in config:
        return _ANOMALY_SCHEMA
    raise ValueError("Unrecognized config type (expected training, simulation, or anomaly config)")


def _check_keys(node: dict, schema: dict | object, prefix: str = "") -> None:
    if schema is _VALUE:
        return
    if schema is _OPEN_MAP:
        if not isinstance(node, dict):
            raise ValueError(f"Configuration field {prefix!r} must be a mapping")
        return
    if not isinstance(node, dict):
        raise ValueError(f"Configuration field {prefix or '<root>'!r} must be a mapping")
    for key, value in node.items():
        if not isinstance(key, str):
            raise ValueError(f"Non-string config key at {prefix or '<root>'}: {key!r}")
        spec = schema.get(key, schema.get("*", _MISSING))
        dotted = f"{prefix}.{key}" if prefix else key
        if spec is _MISSING:
            suggestion = get_close_matches(key, [k for k in schema if k != "*"], n=1)
            hint = f"; did you mean {(prefix + '.' if prefix else '') + suggestion[0]!r}?" if suggestion else ""
            raise ValueError(f"Unknown configuration key {dotted!r}{hint}")
        _check_keys(value, spec, dotted)


def validate_config(config: dict) -> None:
    """Reject misspelled keys in the three supported config families."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    _check_keys(config, _schema_for(config))


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. `inherit: other.yaml` (relative to this file) is merged underneath."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {path} must be a YAML mapping")

    parent = config.pop("inherit", None)
    if parent is not None:
        config = _deep_merge(load_config(path.parent / parent), config)

    validate_config(config)
    for item in overrides or []:
        set_dotted(config, item)
    return config


def set_dotted(config: dict, item: str) -> None:
    """Apply `train.max_steps=500`; reject unknown paths before modifying the config."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key.path=value, got {item!r}")
    parts = key.strip().split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid override key {key!r}")

    # Work on a copy so an invalid override never leaves the original partly changed.
    candidate = copy.deepcopy(config)
    node = candidate
    for part in parts[:-1]:
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {key!r}: {part!r} is not a mapping")
        if part not in node:
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"Cannot set {key!r}: parent is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)
    validate_config(candidate)
    config.clear()
    config.update(candidate)
