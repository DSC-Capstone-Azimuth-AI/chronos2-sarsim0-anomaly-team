#!/usr/bin/env bash
# Train one config inside an NRP GPU pod (k8s/*-job.yaml; see docs/nrp.md), log to W&B, then upload
# the run to the Hugging Face Hub so it outlives the pod.
# Usage: scripts/nrp-train.sh <config> [extra `sfm train` arguments]
# Env: WANDB_API_KEY, HF_TOKEN (from the sarsim-secrets Secret), WANDB_GROUP (default: none).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config=$1
shift

uv sync --frozen --extra cu130 --extra wandb
uv run python -c 'import torch; assert torch.cuda.is_available(), "no CUDA GPU"; print(torch.cuda.get_device_name(0))'
# Fail now, not after hours of training, if the upload would be refused.
uv run python scripts/hub_upload.py --check

# Eval sets are not in git: rebuild them as on any new machine (README "Eval sets").
uv run sfm make-sim
git clone --depth 1 --filter=blob:none --sparse https://github.com/Mcompetitions/M4-methods.git external/M4-data
git -C external/M4-data sparse-checkout set Dataset
uv run sfm stage-m4

# Peak GPU memory and seconds per step at the config's micro-batch, before committing to the run.
uv run sfm check --config "$config" --steps 3

group=()
[[ -n "${WANDB_GROUP:-}" ]] && group=(--set "wandb.group=$WANDB_GROUP")
status=0
uv run sfm train "$config" --set wandb.enabled=true "${group[@]}" "$@" || status=$?

# The pod's disk is deleted with it. Upload even if evaluation failed after training finished.
for run in outputs/*/; do
  if [[ -d "$run/checkpoint-final" ]]; then
    uv run python scripts/hub_upload.py "$run"
  fi
done
exit "$status"
