#!/usr/bin/env bash
# Run 2-second DERConnect detection on NRP: one GPU Job per building, then export dashboard files.
#
#   scripts/nrp-derconnect-2s.sh [BUILDING ...]       # default: all five
#
# Before the first run (once):
#   1. Stage locally:   uv run python scripts/stage_derconnect.py ~/Downloads/Haojian_Data.zip \
#                           --freq 2s --out data/anomaly/derconnect-2s
#   2. Upload data:     scripts/nrp-transfer.sh up data/anomaly/derconnect-2s /data/data/anomaly
#   3. Upload weights:  scripts/nrp-transfer.sh up outputs/chronos2-ft-5k/checkpoint_final /data/outputs/chronos2-ft-5k
#      (skip if the checkpoint is already on the volume: scripts/nrp-transfer.sh ls /data/outputs)
#   4. Commit and push this script, the config and scripts/export_dashboard.py: the Job clones GitHub.
#
# Env overrides: CHECKPOINT (path on the volume), MODEL_NAME (dashboard model name), plus everything
# scripts/nrp-run.sh accepts (NRP_USER, CPU, MEM, DRY_RUN=1, ...).
# Results: /data/outputs/anomaly/derconnect-2s-<MODEL_NAME>-<BUILDING>/ (raw detector run) and
#          /data/outputs/dashboard/<MODEL_NAME>-2s/<BUILDING>/ (Parquet for the dashboard).
# Fetch:   scripts/nrp-transfer.sh down /data/outputs/dashboard/<MODEL_NAME>-2s outputs/dashboard
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/data/outputs/chronos2-ft-5k/checkpoint_final}"
MODEL_NAME="${MODEL_NAME:-chronos2-ft-5k}"
if [[ $# -eq 0 ]]; then set -- E2451 E3020 E3510 E7080 E7085; fi

cd "$(dirname "$0")/.."
for B in "$@"; do
  RUN="outputs/anomaly/derconnect-2s-${MODEL_NAME}-${B}"
  # In the Job, `sfm` already means `uv run --locked --extra cu130 --extra wandb sfm`.
  CMD="test -f data/anomaly/derconnect-2s/derconnect-${B}-2s.parquet || { echo 'data not uploaded'; exit 1; }
test -f ${CHECKPOINT}/config.json || { echo 'checkpoint not found: ${CHECKPOINT}'; exit 1; }
sfm detect configs/anomaly/derconnect-2s.yaml --model chronos2:${CHECKPOINT} --device cuda \\
  --set 'datasets=[data/anomaly/derconnect-2s/derconnect-${B}-2s.parquet]' --out ${RUN}
uv run --locked --extra cu130 --extra wandb python scripts/export_dashboard.py ${RUN} \\
  --model-name ${MODEL_NAME} --out outputs/dashboard/${MODEL_NAME}-2s/${B} --formats parquet"
  scripts/nrp-run.sh "der2s-$(echo "$B" | tr '[:upper:]' '[:lower:]')" "$CMD"
done
