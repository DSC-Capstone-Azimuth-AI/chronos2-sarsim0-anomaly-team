"""YAML run configs with `inherit:` and `--set a.b=value` overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


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

    parent = config.pop("inherit", None)
    if parent is not None:
        config = _deep_merge(load_config(path.parent / parent), config)

    for item in overrides or []:
        set_dotted(config, item)
    return config


def set_dotted(config: dict, item: str) -> None:
    """Apply `train.max_steps=500`. The value is parsed as YAML, so numbers and lists work."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key.path=value, got {item!r}")

    *parents, leaf = key.strip().split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)
