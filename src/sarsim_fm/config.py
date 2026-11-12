"""YAML run configs with `inherit:` and validated `--set a.b=value` overrides."""

from __future__ import annotations

import copy
from difflib import get_close_matches
from itertools import pairwise
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
    if config.get("kind") == "derconnect":
        return _DERCONNECT_SCHEMA
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
        raise ValueError(f"Configuration field {prefix or '<root>'!r} must be a mapping")  # noqa: TRY004 -- config validation uses ValueError
    for key, value in node.items():
        if not isinstance(key, str):
            raise ValueError(f"Non-string config key at {prefix or '<root>'}: {key!r}")  # noqa: TRY004 -- config validation uses ValueError
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
        raise ValueError("Configuration must be a YAML mapping")  # noqa: TRY004 -- config validation uses ValueError
    _check_keys(config, _schema_for(config))
    if config.get("kind") == "derconnect":
        validate_derconnect(config)


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
        raise ValueError(f"Configuration {path} must be a YAML mapping")  # noqa: TRY004 -- config validation uses ValueError

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
            raise ValueError(f"Cannot set {key!r}: {part!r} is not a mapping")  # noqa: TRY004 -- config validation uses ValueError
        if part not in node:
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        raise ValueError(f"Cannot set {key!r}: parent is not a mapping")  # noqa: TRY004 -- config validation uses ValueError
    node[parts[-1]] = yaml.safe_load(raw)
    validate_config(candidate)
    config.clear()
    config.update(candidate)

# DERCONNECT has a separate, closed namespace; legacy benchmark schemas stay intact.
_WINDOW_SCHEMA = {"start": _VALUE, "end": _VALUE}
_DERCONNECT_SCHEMA = {
    "kind": _VALUE, "data_kind": _VALUE, "name": _VALUE, "staged": _VALUE, "annotations": _VALUE,
    "meters": _VALUE, "months": _VALUE, "unexpected": _VALUE, "locked": _VALUE,
    "selection_record": _VALUE, "calibration_record": _VALUE,
    "preparation": {"chunk_size": _VALUE, "cadence_seconds": _VALUE,
                    "sampling_seconds": _VALUE, "min_coverage": _VALUE,
                    "max_gap_seconds": _VALUE, "phase_columns": _VALUE, "phase_units": _VALUE,
                    "units_confirmed": _VALUE, "unit": _VALUE},
    "roles": {k: _WINDOW_SCHEMA for k in
              ("scale", "calibration", "heldout_normal", "development", "evaluation")},
    "calibration": {"exploratory": _VALUE, "target_fpr": _VALUE, "max_points": _VALUE},
    "detection": {"context_length": _VALUE, "batch_size": _VALUE,
                  "min_run_seconds": _VALUE, "merge_gap_seconds": _VALUE},
    "injections": {"seed": _VALUE, "kind": _VALUE, "severity": _VALUE,
                   "duration_seconds": _VALUE, "windows": _VALUE},
    "review": {"seed": _VALUE, "random_windows": _VALUE, "window_seconds": _VALUE},
}


def validate_derconnect(config):
    from sarsim_fm.data.annotations import window

    if config["kind"] != "derconnect":
        raise ValueError("expected kind: derconnect")
    if config.get("data_kind") not in {"artificial_fixture", "real_derconnect", "injected_derconnect"}:
        raise ValueError("data_kind must distinguish artificial_fixture, real_derconnect or injected_derconnect")
    required = {"preparation", "roles", "calibration", "detection", "meters", "unexpected", "staged"}
    if not required <= config.keys():
        raise ValueError(f"missing DERCONNECT fields: {sorted(required - config.keys())}")
    for section in ("preparation", "calibration", "detection"):
        expected = set(_DERCONNECT_SCHEMA[section])
        if not expected <= config[section].keys():
            raise ValueError(f"missing {section} fields: {sorted(expected - config[section].keys())}")
    if type(config.get("locked", False)) is not bool:
        raise ValueError("locked must be Boolean")
    prep = config["preparation"]
    if type(prep["units_confirmed"]) is not bool:
        raise ValueError("units_confirmed must be Boolean")
    for field in ("chunk_size", "sampling_seconds", "cadence_seconds", "max_gap_seconds"):
        if type(prep[field]) is not int or prep[field] < 1:
            raise ValueError(f"preparation.{field} must be a positive integer")
    if prep["sampling_seconds"] != 60:
        raise ValueError("v1 supports a one-minute detection grid")
    if 60 % prep["cadence_seconds"] or not 0 < prep["min_coverage"] <= 1:
        raise ValueError("invalid cadence or min_coverage")
    if len(prep["phase_columns"]) != 3 or len(set(prep["phase_columns"])) != 3:
        raise ValueError("exactly three distinct phase columns required")
    if len(prep["phase_units"]) != 3 or len(set(prep["phase_units"])) != 1:
        raise ValueError("phase units must be compatible and explicitly equal")
    if prep["units_confirmed"] and prep["unit"] == "source_unconfirmed":
        raise ValueError("confirmed units require an explicit unit")
    if not prep["units_confirmed"] and prep["unit"] != "source_unconfirmed":
        raise ValueError("unconfirmed units must be source_unconfirmed")
    if config["unexpected"] not in {"report", "error"}:
        raise ValueError("unexpected must be report or error")
    if not config["meters"] or len(set(config["meters"])) != len(config["meters"]):
        raise ValueError("meters must be nonempty and unique")
    d = config["detection"]
    for key in ("context_length", "batch_size", "min_run_seconds", "merge_gap_seconds"):
        if type(d[key]) is not int or d[key] < (0 if key == "merge_gap_seconds" else 1):
            raise ValueError(f"invalid detection.{key}")
    if d["min_run_seconds"] % 60 or d["merge_gap_seconds"] % 60:
        raise ValueError("event durations must be exact multiples of sampling_seconds")
    cal = config["calibration"]
    if cal["target_fpr"] not in (.005, .01, .02) or type(cal["exploratory"]) is not bool:
        raise ValueError("invalid target_fpr or exploratory override")
    if type(cal["max_points"]) is not int or cal["max_points"] < 199:
        raise ValueError("calibration.max_points must be an integer >=199")
    roles = config["roles"]
    parsed = {key: window(value) for key, value in roles.items()}
    for key, (start, end) in parsed.items():
        if start.second or start.microsecond or end.second or end.microsecond:
            raise ValueError(f"{key} boundaries must align to minute bins")
    ordered = [parsed[k] for k in ("scale", "calibration", "heldout_normal", "development", "evaluation")
               if k in parsed]
    if any(a[1] > b[0] for a, b in pairwise(ordered)):
        raise ValueError("temporal roles must be disjoint and ordered before evaluation")
    if not {"scale", "calibration", "evaluation"} <= parsed.keys():
        raise ValueError("scale, calibration and evaluation windows required")
    if config.get("locked") and (not config.get("selection_record") or
                                 not config.get("calibration_record")):
        raise ValueError("locked evaluation requires selection and calibration records")
    for key in ("selection_record", "calibration_record"):
        if config.get("locked") and not Path(config[key]).is_file():
            raise ValueError(f"missing {key}")

    if "injections" in config:
        inj = config["injections"]
        if set(inj) != set(_DERCONNECT_SCHEMA["injections"]):
            raise ValueError("injections requires seed/kind/severity/duration_seconds/windows")
        if type(inj["seed"]) is not int or inj["seed"] < 0:
            raise ValueError("injection seed must be a nonnegative integer")
        if inj["kind"] not in {"spike", "dip", "level_shift", "variance_burst", "stuck_sensor",
                               "trend_change", "dropout"}:
            raise ValueError("unsupported injection kind")
        if not isinstance(inj["severity"], (int, float)) or not 0 < inj["severity"] < float("inf"):
            raise ValueError("injection severity must be finite and positive")
        if type(inj["duration_seconds"]) is not int or inj["duration_seconds"] < 60 or inj["duration_seconds"] % 60:
            raise ValueError("injection duration must be a positive multiple of 60")
        if not isinstance(inj["windows"], list):
            raise ValueError("injection windows must be a list")
        for entry in inj["windows"]:
            if not isinstance(entry, dict) or set(entry) != {"meter_id", "start", "end", "split"}:
                raise ValueError("injection windows require meter_id/start/end/split")
            window(entry)
            if entry["split"] not in {"development", "test"}:
                raise ValueError("injection split must be development or test")
    if "review" in config:
        review = config["review"]
        if set(review) != set(_DERCONNECT_SCHEMA["review"]):
            raise ValueError("review requires seed/random_windows/window_seconds")
        if type(review["seed"]) is not int or review["seed"] < 0:
            raise ValueError("review seed must be a nonnegative integer")
        if type(review["random_windows"]) is not int or not 0 <= review["random_windows"] <= 1000:
            raise ValueError("review random_windows must be in 0..1000")
        if type(review["window_seconds"]) is not int or not 60 <= review["window_seconds"] <= 86400:
            raise ValueError("review window_seconds must be in 60..86400")
