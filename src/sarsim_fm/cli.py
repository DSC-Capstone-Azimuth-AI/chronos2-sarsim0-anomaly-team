"""`sfm` command line. Run `sfm <command> --help` for options."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sarsim_fm import hub, tracking
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
    from sarsim_fm.models import get_timer
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

    timer = get_timer(cfg["model"]["type"])
    print(f"Running {args.steps} training steps on {runtime.device} (bf16={runtime.bf16}) ...")
    result = timer(cfg, runtime, args.steps)
    print(json.dumps(result, indent=2))
    accum = cfg["train"].get("grad_accum", 1)
    per_opt_step = result["sec_per_step"] * accum
    print(
        f"≈ {per_opt_step:.2f} s per optimizer step (grad_accum={accum}) → "
        f"{3600 / per_opt_step:.0f} optimizer steps/hour"
    )


def cmd_stage_m4(args) -> None:
    from sarsim_fm.data.m4 import M4_ROOT, stage_m4

    for freq in args.freq:
        stage_m4(freq, Path(args.m4_root) if args.m4_root else M4_ROOT)


def cmd_make_sim(args) -> None:
    from sarsim_fm.config import load_config
    from sarsim_fm.data.sim import make_heldout_split

    cfg = load_config(args.config, args.set)
    for name, split in cfg["splits"].items():
        if not args.only or name in args.only:
            make_heldout_split(name, split, cfg.get("defaults", {}))


def cmd_make_anomaly_sim(args) -> None:
    from sarsim_fm.config import load_config
    from sarsim_fm.data.sim import make_anomaly_split

    cfg = load_config(args.config, args.set)
    defaults = cfg.get("defaults", {})
    ranges = []
    for name, split in cfg["splits"].items():
        spec = {**defaults, **split, "sarsim": {**defaults.get("sarsim", {}), **split.get("sarsim", {})}}
        seeds = set(range(spec["seed"], spec["seed"] + spec["n_series"]))
        if any(seeds & previous for previous in ranges):
            raise ValueError("anomaly split seed ranges overlap")
        ranges.append(seeds)
        make_anomaly_split(name, spec, Path(args.out))


def cmd_detect(args) -> None:
    from sarsim_fm.anomaly.pipeline import run_detection
    from sarsim_fm.config import load_config

    cfg = load_config(args.config, args.set)
    run_detection(cfg, args.model, Path(args.out) if args.out else None,
                  args.device, args.overwrite, config_file=str(args.config))


def cmd_anomaly_results(args) -> None:
    from sarsim_fm.anomaly.pipeline import collect_results

    print(json.dumps(collect_results(Path(args.root)), indent=2))


def cmd_select_anomaly(args) -> None:
    from sarsim_fm.anomaly.selection import select_settings

    select_settings(Path(args.run), Path(args.out))


def cmd_stage_ucr(args) -> None:
    from sarsim_fm.data.ucr_anomaly import stage_ucr

    stage_ucr(Path(args.source), Path(args.out))


def cmd_anomaly_pilot(args) -> None:
    from sarsim_fm.anomaly.experiment import compare_pilot
    from sarsim_fm.config import load_config

    compare_pilot(load_config(args.config), args.model, Path(args.out), args.device)


def cmd_anomaly_experiment(args) -> None:
    from sarsim_fm.anomaly.experiment import run_experiment

    run_experiment(args.device, include_ucr=not args.synthetic_only,
                   external_only=args.external_only, checkpoint=args.checkpoint,
                   pretrained_checkpoint=args.pretrained_checkpoint)


def cmd_anomaly_diagnostics(args) -> None:
    from sarsim_fm.anomaly.diagnostics import write_diagnostics

    print(json.dumps(write_diagnostics(Path(args.run), args.max_plots), indent=2))


def cmd_prepare_anomaly_handoff(args) -> None:
    from sarsim_fm.anomaly.handoff import prepare_handoff

    print(json.dumps(prepare_handoff(Path(args.out), include_ucr=not args.without_ucr,
                                    overwrite=args.overwrite), indent=2))


def cmd_verify_anomaly_handoff(args) -> None:
    from sarsim_fm.anomaly.handoff import verify_handoff

    print(json.dumps(verify_handoff(Path(args.root)), indent=2))


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


def _file_sha256(path: Path) -> str | None:
    """Return a streaming checksum; a missing optional dataset is recorded as null."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hashes(names: list[str] | None) -> dict[str, str | None]:
    from sarsim_fm.data.eval_sets import EVAL_DIR, eval_set_path

    names = names if names else sorted(p.stem for p in EVAL_DIR.glob("*.parquet"))
    return {name: _file_sha256(eval_set_path(name)) for name in names}


def _reproducibility_info(cfg: dict, runtime, datasets: list[str] | None) -> dict:
    """Snapshot what is verifiable without network calls or assuming a HF loader revision."""
    import torch

    model = cfg.get("model", {})
    pretrained = model.get("init", "pretrained") != "scratch" and model.get("type") != "seasonal_naive"
    revision = model.get("revision")
    return {
        "uv_lock_sha256": _file_sha256(ROOT / "uv.lock"),
        "dataset_sha256": _dataset_hashes(datasets),
        "model_revision": {
            "source": model.get("base", model.get("spec")),
            "requested": revision if pretrained else None,
            "immutable_requested": bool(pretrained and isinstance(revision, str)
                                        and re.fullmatch(r"[a-fA-F0-9]{40}", revision)),
            # The pretrained loader is responsible for reporting its actual commit.
            # A revision configured here does not prove the loader used it.
            "loaded_commit": None,
            "verified": False,
            "applicable": pretrained,
        },
        "settings": {
            "seed": cfg.get("data", {}).get("seed"),
            "num_workers": cfg.get("data", {}).get("num_workers"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "device": str(runtime.device),
            "bf16": runtime.bf16,
            "tf32": runtime.tf32,
        },
    }


def _run_manifest(cfg: dict, config_file: str | None, runtime,
                  datasets: list[str] | None, job_type: str) -> dict:
    from sarsim_fm.runtime import environment_info

    return {
        "schema_version": 2,
        "job_type": job_type,
        "config": cfg,
        "config_file": config_file,
        "environment": environment_info(runtime),
        "reproducibility": _reproducibility_info(cfg, runtime, datasets),
    }


def _baseline_directory(spec: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose a new --out to avoid mixing results")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = OUTPUTS / "baselines" / _safe_name(spec) / f"{stamp}-{uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def cmd_eval(args) -> None:
    if args.out and len(args.model) > 1:
        sys.exit("--out only works with a single --model")
    runtime = _runtime_from(args)
    for spec in args.model:
        out_dir = _baseline_directory(spec, args.out)
        print(f"== {spec} → {out_dir}")
        run_cfg = {
            "wandb": {"enabled": args.wandb, "group": "baselines"},
            "model": {"type": spec.split(":", 1)[0], "spec": spec},
            "data": {"context_length": args.context_length},
            "runtime": {"device": str(runtime.device), "bf16": runtime.bf16},
        }
        (out_dir / "run-config.json").write_text(
            json.dumps(_run_manifest(run_cfg, None, runtime, args.datasets, "baseline"), indent=2)
        )
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
    hub_cfg = hub.settings(cfg)
    repo_id = hub.repo_id_for(cfg["name"], hub_cfg) if hub_cfg["enabled"] else None
    try:
        if repo_id:
            # Check login and repo before hours of training, not after.
            hub.check_access(repo_id, hub_cfg["overwrite"])
        _train_and_eval(args, cfg, runtime)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
    if repo_id:
        _push(OUTPUTS / cfg["name"], repo_id, hub_cfg["private"], hub_cfg["overwrite"])


def _push(run_dir: Path, repo_id: str, private: bool, overwrite: bool) -> None:
    print(f"Uploading {run_dir.name} to {repo_id} ...")
    try:
        url = hub.push_run(run_dir, repo_id, private=private, overwrite=overwrite)
    except hub.HubError as exc:
        sys.exit(f"Upload failed (the run is kept in {run_dir}): {exc}")
    print(f"Uploaded: {url}")


def cmd_push(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    hub_cfg = hub.settings({"hub": {k: v for k, v in (("org", args.org), ("repo", args.repo)) if v}})
    _push(run_dir, hub.repo_id_for(run_dir.name, hub_cfg), not args.public, args.force)


def _train_and_eval(args, cfg: dict, runtime) -> None:
    import time

    import torch

    from sarsim_fm.models import get_trainer
    run_dir = OUTPUTS / cfg["name"]
    if (run_dir / "run-config.json").exists():
        if not args.overwrite:
            sys.exit(f"{run_dir} already has a run; pass --overwrite or change `name`")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    (run_dir / "run-config.json").write_text(
        json.dumps(
            _run_manifest(cfg, str(args.config), runtime, cfg.get("eval", {}).get("datasets"), "train"),
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

    p = sub.add_parser("stage-m4", help="stage M4 subsets into data/eval/m4_<freq>.parquet")
    p.add_argument("--m4-root", help="checkout of Mcompetitions/M4-methods (default external/M4-data)")
    p.add_argument(
        "--freq", nargs="+", default=["monthly", "quarterly"], choices=["monthly", "quarterly"],
        help="M4 subsets to stage (default: both; monthly = test, quarterly = validation)",
    )
    p.set_defaults(func=cmd_stage_m4)

    p = sub.add_parser("make-sim", help="generate fixed held-out SarSim0 eval sets")
    p.add_argument("--config", default=DEFAULT_SIM_CONFIG)
    p.add_argument("--only", nargs="*", help="split names to build (default: all)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_make_sim)

    p = sub.add_parser("make-anomaly-sim", help="generate fixed synthetic anomaly and normal splits")
    p.add_argument("config")
    p.add_argument("--out", default=str(ROOT / "data" / "anomaly"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_make_anomaly_sim)

    p = sub.add_parser("detect", help="sequential forecasts, clean calibration, events, and metrics")
    p.add_argument("config")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("anomaly-results", help="compare saved anomaly summaries and paired UCR outcomes")
    p.add_argument("--root", default=str(OUTPUTS / "anomaly"))
    p.set_defaults(func=cmd_anomaly_results)

    p = sub.add_parser("select-anomaly", help="freeze score/FPR/event settings from a completed dev run")
    p.add_argument("run")
    p.add_argument("--out", default=str(OUTPUTS / "anomaly" / "synthetic-selection.json"))
    p.set_defaults(func=cmd_select_anomaly)

    p = sub.add_parser("stage-ucr-anomaly", help="validate all 250 archive files and stage a manifest")
    p.add_argument("--source", required=True)
    p.add_argument("--out", default=str(ROOT / "data" / "anomaly" / "ucr.manifest.json"))
    p.set_defaults(func=cmd_stage_ucr)

    p = sub.add_parser("anomaly-pilot", help="compare rolling and block forecasts on a fixed dev pilot")
    p.add_argument("config")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=str(OUTPUTS / "anomaly-pilot"))
    p.add_argument("--device")
    p.set_defaults(func=cmd_anomaly_pilot)

    p = sub.add_parser("anomaly-experiment", help="resume ordered dev selection, locked tests, UCR, comparison")
    p.add_argument("--device")
    p.add_argument("--synthetic-only", action="store_true", help="stop after locked synthetic evaluation")
    p.add_argument("--external-only", action="store_true", help="use transferred frozen settings; run UCR only")
    p.add_argument("--checkpoint", help="optional existing fine-tuned checkpoint directory")
    p.add_argument("--pretrained-checkpoint", help="local directory of revision-pinned pretrained weights")
    p.set_defaults(func=cmd_anomaly_experiment)

    p = sub.add_parser("anomaly-diagnostics", help="post-lock failure plots and family metrics")
    p.add_argument("run")
    p.add_argument("--max-plots", type=int, default=20)
    p.set_defaults(func=cmd_anomaly_diagnostics)

    p = sub.add_parser("prepare-anomaly-handoff", help="verify and package code/data/frozen settings for a remote GPU")
    p.add_argument("--out", default=str(OUTPUTS / "anomaly-handoff"))
    p.add_argument("--without-ucr", action="store_true", help="transfer UCR separately and set SARSIM_UCR_SOURCE remotely")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare_anomaly_handoff)

    p = sub.add_parser("verify-anomaly-handoff", help="verify all packaged file hashes after transfer")
    p.add_argument("--root", default=str(ROOT))
    p.set_defaults(func=cmd_verify_anomaly_handoff)

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
    p.add_argument("--model", action="append", required=True, help="e.g. chronos2:amazon/chronos-2, patchtst_fm:outputs/<run>/checkpoint-final (repeatable)")
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

    p = sub.add_parser("push", help="upload a finished run (checkpoint-final + metadata) to the Hugging Face Hub")
    p.add_argument("run_dir", help="e.g. outputs/chronos2-ft-5k; the folder name becomes the repo name")
    p.add_argument("--org", help=f"Hub organization (default {hub.DEFAULTS['org']})")
    p.add_argument("--repo", help="full repo id, overriding <org>/<folder name>")
    p.add_argument("--public", action="store_true", help="create a public repo (default private)")
    p.add_argument("--force", action="store_true", help="upload into an existing repo (new commit on top)")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("results", help="markdown table of every eval summary under outputs/")
    p.add_argument("--root", default=str(OUTPUTS))
    p.set_defaults(func=cmd_results)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
