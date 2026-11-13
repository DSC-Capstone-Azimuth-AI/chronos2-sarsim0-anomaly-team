"""Lazy CLI routes for DERCONNECT; dry-run/export never import model loaders."""
from __future__ import annotations

import json

from sarsim_fm.config import load_config


def stage(args):
    from sarsim_fm.data.derconnect import stage_derconnect
    result = stage_derconnect(args.source, load_config(args.config), args.out)
    print(json.dumps({"preparation_id": result["preparation_id"], "counts": result["counts"]}, indent=2))


def detect(args):
    from sarsim_fm.anomaly.derconnect import detect_derconnect, dry_run
    cfg = load_config(args.config)
    if args.meter:
        if not set(args.meter) <= set(cfg["meters"]):
            raise ValueError("meter selector outside configured scope")
        cfg["meters"] = args.meter
    if args.dry_run:
        result = dry_run(cfg, args.model)
    else:
        if not args.out:
            raise ValueError("--out required for detection")
        result = detect_derconnect(cfg, args.model, args.out, device=args.device, resume=args.resume)
        result = {"run_id": result["run_id"], "complete": result["complete"]}
    print(json.dumps(result, indent=2))


def evaluate(args):
    from sarsim_fm.anomaly.derconnect_evaluation import evaluate_run
    evaluate_run(args.run, args.annotations)


def export(args):
    from sarsim_fm.anomaly.export import export_bundle
    export_bundle(args.run, args.out, csv=args.format == "csv", hourly=args.hourly)


def injections(args):
    from sarsim_fm.anomaly.derconnect_review import make_injections
    make_injections(load_config(args.config), args.out)


def review(args):
    from sarsim_fm.anomaly.derconnect_review import review_bundle
    review_bundle(args.run, args.out, args.compare_run, isolated_count=args.isolated_flags)


def apply_alert_policy(args):
    from pathlib import Path

    from sarsim_fm.anomaly.alert_policy import apply_policy
    cfg = load_config(args.config)
    result = apply_policy(args.run, cfg, args.out, mode=args.mode,
                          evidence_base=Path(args.config).resolve().parent, annotations=args.annotations)
    print(json.dumps(result, indent=2))


def normalize(args):
    from sarsim_fm.anomaly.review_annotations import normalize_reviews
    rows = normalize_reviews(args.reviews, args.mapping, args.out)
    print(json.dumps({"annotations": len(rows), "out": args.out}))


def register(sub):
    p = sub.add_parser("apply-derconnect-policy", help="verified schedule-policy overlay; no models")
    p.add_argument("run")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["preview", "active"], default="preview")
    p.add_argument("--out", required=True)
    p.add_argument("--annotations")
    p.set_defaults(func=apply_alert_policy)
    p = sub.add_parser("normalize-derconnect-reviews", help="approved review mapping to explicit truth")
    p.add_argument("--reviews", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=normalize)

    p = sub.add_parser("stage-derconnect", help="validate and prepare DERCONNECT CSVs without models")
    p.add_argument("--source", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=stage)
    p = sub.add_parser("detect-derconnect", help="frozen-weight causal DERCONNECT detection")
    p.add_argument("config")
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.add_argument("--device", default="cpu")
    p.add_argument("--meter", action="append")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=detect)
    p = sub.add_parser("evaluate-derconnect", help="evaluate only approved reviewed coverage")
    p.add_argument("run")
    p.add_argument("--annotations", required=True)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("export-derconnect", help="dashboard files from completed artifacts only")
    p.add_argument("run")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    p.add_argument("--hourly", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=export)
    p = sub.add_parser("make-derconnect-injections", help="copy reviewed backgrounds with reproducible injections")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.set_defaults(func=injections)
    p = sub.add_parser("review-derconnect", help="alert and random non-alert review windows")
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.add_argument("--compare-run")
    p.add_argument("--isolated-flags", type=int, default=0)
    p.set_defaults(func=review)
