"""Inspect saved forecasts or explicitly reproduce a bounded pinned-model case set."""
import argparse
import json

from sarsim_fm.anomaly.forecast_diagnostics import inspect_bundle, reproduce, reproduction_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "reproduce", "scope"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True)
        p.add_argument("--target", action="append")
        p.add_argument("--dry-run", action="store_true")
        if name != "scope":
            p.add_argument("--out", required=True)
        if name == "reproduce":
            p.add_argument("--staged", required=True)
            p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "scope" or args.dry_run:
        result = reproduction_scope(args.bundle, args.target)
    elif args.command == "inspect":
        result = inspect_bundle(args.bundle, args.out, args.target)
        result = {"diagnostic_id": result["diagnostic_id"], "detectors": result["detectors"]}
    else:
        result = reproduce(args.bundle, args.staged, args.out, device=args.device, targets=args.target)
        result = {"target_predictions": result["target_predictions"], "unavailable": result["unavailable"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
