"""`sfm` command line. Run `sfm <command> --help` for options."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

from sarsim_fm import tracking
from sarsim_fm.runtime import ROOT

OUTPUTS = ROOT / "outputs"
DEFAULT_SIM_CONFIG = ROOT / "configs" / "data" / "sim-heldout.yaml"
DEFAULT_CHECK_CONFIG = ROOT / "configs" / "train" / "chronos2-ft-smoke.yaml"


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:N | mps (overrides runtime.device)")
    parser.add_argument("--precision", help="auto | bf16 | fp32 (overrides runtime.precision)")


def _runtime_from(args, cfg: dict | None = None):
    from sarsim_fm.runtime import resolve_runtime

    runtime_cfg = (cfg or {}).get("runtime", {})
    return resolve_runtime(
        args.device or runtime_cfg.get("device", "auto"),
        args.precision or runtime_cfg.get("precision", "auto"),
    )


def cmd_check(args) -> None:
    """Environment report, SarSim0 sanity checks, and timed training steps."""
    import torch

    from sarsim_fm.config import load_config
    from sarsim_fm.data.sim import training_stream
    from sarsim_fm.models import chronos2
    from sarsim_fm.runtime import environment_info

    cfg = load_config(args.config, args.set)
    runtime = _runtime_from(args, cfg)
    print(json.dumps(environment_info(runtime), indent=2))

    data_cfg = cfg["data"]
    batch = next(iter(training_stream(data_cfg)))
    context, future = batch["context"], batch["future_target"]
    assert context.shape[-1] == data_cfg["context_length"], context.shape
    assert torch.isfinite(context).all(), "non-finite context"
    assert torch.isfinite(future[..., : data_cfg["prediction_length"]]).all(), "non-finite target"
    print(f"SarSim0 stream OK: context {tuple(context.shape)}, future_target {tuple(future.shape)}")

    if cfg["model"]["type"] != "chronos2":
        sys.exit(f"`sfm check` timing only supports chronos2 so far, not {cfg['model']['type']}")
    print(f"Running {args.steps} training steps on {runtime.device} (bf16={runtime.bf16}) ...")
    result = chronos2.timed_steps(cfg, runtime, args.steps)
    print(json.dumps(result, indent=2))
    accum = cfg["train"].get("grad_accum", 1)
    per_opt_step = result["sec_per_step"] * accum
    print(
        f"≈ {per_opt_step:.2f} s per optimizer step (grad_accum={accum}) → "
        f"{3600 / per_opt_step:.0f} optimizer steps/hour"
    )


def cmd_stage_m4(args) -> None:
    from sarsim_fm.data.m4 import stage_m4_monthly

    stage_m4_monthly(Path(args.m4_root)) if args.m4_root else stage_m4_monthly()


def cmd_make_sim(args) -> None:
    from sarsim_fm.config import load_config
    from sarsim_fm.data.sim import make_heldout_split

    cfg = load_config(args.config, args.set)
    for name, split in cfg["splits"].items():
        if not args.only or name in args.only:
            make_heldout_split(name, split, cfg.get("defaults", {}))


def _safe_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec).strip("_")


def _run_eval(model_spec: str, out_dir: Path, datasets: list[str], args, runtime, meta: dict) -> None:
    from sarsim_fm.data.eval_sets import EVAL_DIR, eval_set_path, load_eval_set
    from sarsim_fm.evaluate import evaluate, save_eval
    from sarsim_fm.models import load_forecaster

    if not datasets:
        datasets = sorted(p.stem for p in EVAL_DIR.glob("*.parquet"))
    forecaster = load_forecaster(model_spec, runtime, args.context_length)
    for name in datasets:
        if not eval_set_path(name).exists():
            print(f"[skip] {name}: {eval_set_path(name)} not found")
            continue
        eval_set = load_eval_set(name)
        per_series, summary = evaluate(forecaster, eval_set, args.batch_size)
        save_eval(
            out_dir, name, per_series, summary,
            {"model": model_spec, "context_length": args.context_length, "device": str(runtime.device), **meta},
        )
        tracking.log_eval(name, summary)
        print(
            f"{name:>16}: sMAPE {100 * summary['smape']:.3f}%  MASE {summary['mase']:.4f}  "
            f"SQL {summary['sql']:.4f}  WQL {summary['wql']:.4f}  cov80 {summary['coverage_80']:.3f}  "
            f"cov98 {summary['coverage_98']:.3f}  ({summary['predict_seconds']} s)"
        )


def cmd_eval(args) -> None:
    if args.out and len(args.model) > 1:
        sys.exit("--out only works with a single --model")
    runtime = _runtime_from(args)
    for spec in args.model:
        out_dir = Path(args.out) if args.out else OUTPUTS / "baselines" / _safe_name(spec)
        print(f"== {spec} → {out_dir}")
        run_cfg = {
            "wandb": {"enabled": args.wandb, "group": "baselines"},
            "model": {"type": spec.split(":", 1)[0], "spec": spec},
            "data": {"context_length": args.context_length},
            "runtime": {"device": str(runtime.device)},
        }
        tracking.start_run(run_cfg, "baseline", tracking.baseline_name(spec), extra_tags=["baseline"])
        try:
            _run_eval(spec, out_dir, args.datasets, args, runtime, {})
        except BaseException:
            tracking.finish(exit_code=1)
            raise
        tracking.finish()


def _format_duration(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.2f} h"


def _record_timing(run_dir: Path, **fields) -> None:
    """Merge timing fields into run-config.json, so a crash during eval still keeps train time."""
    from datetime import UTC, datetime

    path = run_dir / "run-config.json"
    record = json.loads(path.read_text())
    record.setdefault("timing", {}).update(
        fields, finished=datetime.now(UTC).isoformat(timespec="seconds")
    )
    path.write_text(json.dumps(record, indent=2))
    tracking.log_summary(**{f"timing/{k}": v for k, v in fields.items()})


def _train_seconds(run_dir: Path) -> float | None:
    """From run-config.json timing; falls back to the HF Trainer's train_runtime for older runs."""
    config_path, log_path = run_dir / "run-config.json", run_dir / "train-log.csv"
    if config_path.exists():
        seconds = json.loads(config_path.read_text()).get("timing", {}).get("train_seconds")
        if seconds is not None:
            return seconds
    if log_path.exists():
        import pandas as pd

        runtime = pd.read_csv(log_path).get("train_runtime")
        if runtime is not None and runtime.notna().any():
            return float(runtime.dropna().iloc[-1])
    return None


def cmd_train(args) -> None:
    from sarsim_fm.config import load_config

    cfg = load_config(args.config, [*args.set, *args.overrides])
    runtime = _runtime_from(args, cfg)
    cfg.setdefault("runtime", {}).update(device=str(runtime.device), bf16=runtime.bf16)

    tags = [cfg["model"]["type"], cfg["model"].get("init", "pretrained"), runtime.device.type]
    run = tracking.start_run(cfg, "train", cfg["name"], extra_tags=tags)
    if run is not None and tracking.in_sweep():
        # Sweep runs share one config file, so make each run directory unique.
        cfg["name"] = run.name = f"{cfg['name']}-{run.id}"
        run.config.update({"name": cfg["name"]}, allow_val_change=True)
    try:
        _train_and_eval(args, cfg, runtime)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()


def _train_and_eval(args, cfg: dict, runtime) -> None:
    import time

    import torch

    from sarsim_fm.models import get_trainer
    from sarsim_fm.runtime import environment_info

    run_dir = OUTPUTS / cfg["name"]
    if (run_dir / "run-config.json").exists():
        if not args.overwrite:
            sys.exit(f"{run_dir} already has a run; pass --overwrite or change `name`")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    (run_dir / "run-config.json").write_text(
        json.dumps(
            {"config": cfg, "config_file": str(args.config), "environment": environment_info(runtime)},
            indent=2,
        )
    )
    print(f"Run dir: {run_dir}\nDevice: {runtime.device}  bf16={runtime.bf16}  tf32={runtime.tf32}")

    kind = cfg["model"]["type"]
    start = time.perf_counter()
    checkpoint = get_trainer(kind)(cfg, runtime, run_dir)
    train_seconds = time.perf_counter() - start
    _record_timing(run_dir, train_seconds=round(train_seconds, 1))
    print(f"Final checkpoint: {checkpoint}\nTraining took {_format_duration(train_seconds)}")

    eval_cfg = cfg.get("eval", {})
    if args.no_eval or eval_cfg.get("datasets") == []:
        return
    gc.collect()
    if runtime.is_cuda:
        torch.cuda.empty_cache()
    args.batch_size = eval_cfg.get("batch_size", 64)
    args.context_length = cfg["data"]["context_length"]
    eval_start = time.perf_counter()
    _run_eval(
        f"{kind}:{checkpoint}", run_dir, eval_cfg.get("datasets"), args, runtime, {"run": cfg["name"]}
    )
    eval_seconds = time.perf_counter() - eval_start
    _record_timing(
        run_dir,
        eval_seconds=round(eval_seconds, 1),
        total_seconds=round(time.perf_counter() - start, 1),
    )
    print(
        f"Done: train {_format_duration(train_seconds)}, eval {_format_duration(eval_seconds)}"
    )


def cmd_wandb_import(args) -> None:
    run_dirs = sorted(p.parent.parent for p in Path(args.root).glob("**/eval/summary.json"))
    if not run_dirs:
        sys.exit(f"No eval summaries under {args.root}")
    for run_dir in run_dirs:
        rel = str(run_dir.relative_to(args.root))
        if args.only and rel not in args.only:
            continue
        print(f"{rel}: {tracking.import_run(run_dir, rel, args.group)}")


def cmd_results(args) -> None:
    rows = []
    for path in sorted(Path(args.root).glob("**/eval/summary.json")):
        run = str(path.parent.parent.relative_to(args.root))
        for dataset, s in json.loads(path.read_text()).items():
            rows.append((dataset, run, s))
    if not rows:
        sys.exit(f"No eval summaries under {args.root}")

    train_times = {run: _train_seconds(Path(args.root) / run) for _, run, _ in rows}

    print("| Dataset | Run | Model | sMAPE % | MASE | SQL | WQL | Cov 80% | Cov 98% | N | Train time |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, run, s in sorted(rows, key=lambda r: (r[0], r[1])):
        seconds = train_times[run]
        print(
            f"| {dataset} | {run} | `{s['model'].replace(f'{ROOT}/', '')}` | {100 * s['smape']:.2f} | {s['mase']:.4f} | "
            f"{s['sql']:.4f} | {s['wql']:.4f} | {s['coverage_80']:.3f} | {s['coverage_98']:.3f} | {s['n_series']} | "
            f"{_format_duration(seconds) if seconds is not None else '—'} |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sfm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="environment report + SarSim0 checks + timed training steps")
    p.add_argument("--config", default=DEFAULT_CHECK_CONFIG)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    _add_runtime_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stage-m4", help="stage M4 Monthly into data/eval/m4_monthly.parquet")
    p.add_argument("--m4-root", help="checkout of Mcompetitions/M4-methods (default external/M4-data)")
    p.set_defaults(func=cmd_stage_m4)

    p = sub.add_parser("make-sim", help="generate fixed held-out SarSim0 eval sets")
    p.add_argument("--config", default=DEFAULT_SIM_CONFIG)
    p.add_argument("--only", nargs="*", help="split names to build (default: all)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_make_sim)

    p = sub.add_parser("train", help="train from a YAML config, then evaluate the final checkpoint")
    p.add_argument("config")
    p.add_argument(
        "overrides", nargs="*", metavar="KEY=VALUE", help="same as --set (lets `wandb agent` pass sweep values)"
    )
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--overwrite", action="store_true", help="delete an existing run with the same name")
    p.add_argument("--no-eval", action="store_true")
    _add_runtime_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="zero-shot evaluation of one or more models")
    p.add_argument("--model", action="append", required=True, help="e.g. chronos2:amazon/chronos-2 (repeatable)")
    p.add_argument("--datasets", nargs="*", default=[], help="eval set names (default: all in data/eval)")
    p.add_argument("--out", help="output dir (default outputs/baselines/<model>)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--wandb", action="store_true", help="log each model as a W&B `baseline` run")
    _add_runtime_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("wandb-import", help="upload finished runs under outputs/ to W&B (backfill)")
    p.add_argument("--root", default=str(OUTPUTS))
    p.add_argument("--only", nargs="*", help="run folders relative to --root (default: all)")
    p.add_argument("--group", default="imported", help="W&B group for imported train runs")
    p.set_defaults(func=cmd_wandb_import)

    p = sub.add_parser("results", help="markdown table of every eval summary under outputs/")
    p.add_argument("--root", default=str(OUTPUTS))
    p.set_defaults(func=cmd_results)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
