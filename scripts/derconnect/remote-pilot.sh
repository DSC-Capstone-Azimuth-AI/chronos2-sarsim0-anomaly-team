#!/usr/bin/env bash
# Run on an existing remote machine only, after the real-data pilot gate is approved.
# Does not provision infrastructure, submit jobs, install packages, or train models.
set -euo pipefail
: "${DERCONNECT_CONFIG:?set DERCONNECT_CONFIG to the reviewed remote YAML}"
: "${DERCONNECT_MODEL:?set DERCONNECT_MODEL to a model spec, preferably pinned local weights}"
: "${DERCONNECT_OUT:?set DERCONNECT_OUT to a new persistent result directory}"
: "${DERCONNECT_SCRATCH:?set DERCONNECT_SCRATCH to remote scratch storage}"
export HF_HOME="$DERCONNECT_SCRATCH/huggingface"
export UV_CACHE_DIR="$DERCONNECT_SCRATCH/uv-cache"
sfm detect-derconnect "$DERCONNECT_CONFIG" --model "$DERCONNECT_MODEL" --dry-run
sfm detect-derconnect "$DERCONNECT_CONFIG" --model "$DERCONNECT_MODEL" \
    --device "${DERCONNECT_DEVICE:-cuda}" --out "$DERCONNECT_OUT"
sfm export-derconnect "$DERCONNECT_OUT" --format csv --hourly --out "$DERCONNECT_OUT-dashboard"
