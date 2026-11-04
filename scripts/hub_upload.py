"""Upload a finished run to the Hugging Face Hub (scripts/nrp-train.sh calls this after training).

    python scripts/hub_upload.py outputs/<run>      # -> <repo>/<run>/checkpoint-final, logs, eval
    python scripts/hub_upload.py --check            # token works and the repo exists (before training)

`checkpoint-final/`, `train-log.csv`, `run-config.json`, `timing.json` and `eval/` go up;
intermediate `checkpoint-<step>/` folders stay behind. The token comes from `HF_TOKEN`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO = "teyquila/sarsim0-anomaly-models"
SKIP = ["checkpoint-[0-9]*"]  # fnmatch: also matches every file inside those folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--check", action="store_true", help="only verify the token and repo")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set (k8s: the hf-token key of sarsim-secrets)")
    api = HfApi(token=os.environ["HF_TOKEN"])
    if args.check:
        api.repo_info(args.repo, repo_type="model")  # raises if missing or not visible to the token
        print(f"Hugging Face: {api.whoami()['name']} can see {args.repo}")
        return
    if args.run_dir is None or not (args.run_dir / "checkpoint-final").is_dir():
        raise SystemExit(f"{args.run_dir} has no checkpoint-final to upload")

    run = args.run_dir.resolve().name
    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.run_dir,
        path_in_repo=run,
        ignore_patterns=SKIP,
        commit_message=f"Add {run}",
    )
    print(f"Uploaded {run}: {commit.commit_url}")


if __name__ == "__main__":
    main()
