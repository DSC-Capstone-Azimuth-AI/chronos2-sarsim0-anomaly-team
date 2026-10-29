#!/usr/bin/env bash
# Execute only on the remote GPU machine. This script performs evaluation, never training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv on the remote machine first; see README.md.' >&2
  exit 1
fi

uv sync --frozen --extra cu130 --extra dev --extra notebooks
uv run sfm verify-anomaly-handoff --root .
uv run python -c 'import torch; assert torch.cuda.is_available(), "This launcher requires a remote CUDA GPU"; print(torch.cuda.get_device_name(0))'
uv run pytest -q
uv run ruff check src tests

# Use the exact pretrained revision evaluated during local synthetic development.
uv run python - <<'PY'
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from sarsim_fm.anomaly.provenance import file_hash

record = json.loads(Path("outputs/anomaly/development-2/run-config.json").read_text())["checkpoint"]
target = Path("external/chronos2-pretrained")
snapshot_download("amazon/chronos-2", revision=record["resolved_revision"], local_dir=target)
if file_hash(target / "model.safetensors") != record["sha256"]:
    raise ValueError("Downloaded pretrained weights differ from the locally evaluated checkpoint")
(target / "pretrained-provenance.json").write_text(json.dumps({
    "repository": "amazon/chronos-2", "revision": record["resolved_revision"],
    "weights_sha256": record["sha256"],
}, indent=2))
PY

ucr_source="${SARSIM_UCR_SOURCE:-external/ucr-anomaly}"
uv run sfm stage-ucr-anomaly --source "$ucr_source"
uv run sfm anomaly-experiment --external-only --device cuda \
  --pretrained-checkpoint external/chronos2-pretrained "$@"
uv run sfm anomaly-diagnostics outputs/anomaly/ucr-full-2 --max-plots 20
uv run sfm anomaly-results
