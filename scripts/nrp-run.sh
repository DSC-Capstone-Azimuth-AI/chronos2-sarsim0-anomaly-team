#!/usr/bin/env bash
# Launch any `sfm` command as an NRP GPU Job. See docs/nrp.md.
#
#   scripts/nrp-run.sh <label> "<command>" [git-ref]
#
# Examples:
#   scripts/nrp-run.sh check "sfm check --device cuda --steps 100"
#   scripts/nrp-run.sh ft-5k "sfm train configs/train/chronos2-ft-5k.yaml --set wandb.enabled=true"
#
# Inside <command>, `sfm` means `uv run --locked --extra cu130 --extra wandb sfm`.
# The Job clones the *pushed* commit (default: your current HEAD) from GitHub.
# Env overrides: NRP_USER, REPO, CPU, MEM, GPUS, IMAGE. DRY_RUN=1 prints the manifest instead.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> \"<command>\" [git-ref]" >&2
  exit 1
fi

LABEL="$1"
RUN_CMD="$2"
GIT_REF="${3:-$(git rev-parse HEAD)}"

NRP_USER="${NRP_USER:-$(id -un)}"
REPO="${REPO:-DSC-Capstone-Azimuth-AI/chronos2-sarsim0-anomaly-team}"
CPU="${CPU:-12}"
MEM="${MEM:-48Gi}"
GPUS="${GPUS:-1}"
IMAGE="${IMAGE:-ghcr.io/astral-sh/uv:python3.11-bookworm}"

# k8s names: lowercase alphanumerics and dashes, <= 63 chars, alphanumeric at both ends.
k8s_name() { tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-' | cut -c1-63 | sed 's/^-*//; s/-*$//'; }
NRP_USER="$(echo "$NRP_USER" | k8s_name)"
JOB_NAME="$(echo "${NRP_USER}-${LABEL}-$(date +%m%d-%H%M)" | k8s_name)"

# The pod clones from GitHub, so the ref must be on a pushed branch.
git fetch -q origin 2>/dev/null || true
if [[ -z "$(git branch -r --contains "$GIT_REF" 2>/dev/null)" ]]; then
  echo "warning: $GIT_REF isn't on any pushed branch; the Job clones from GitHub. Push first." >&2
fi

# Runs inside the pod. Quoted heredoc: nothing here is expanded on your laptop; the pod gets
# JOB_NAME, REPO, GIT_REF and RUN_CMD as environment variables.
read -r -d '' POD_SCRIPT <<'SH' || true
set -euo pipefail
mkdir -p /data/logs
{
  echo "== ${JOB_NAME}: ${REPO} @ ${GIT_REF}"
  git clone -q "https://github.com/${REPO}.git" /work
  cd /work
  git checkout -q "${GIT_REF}"
  echo "commit: $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  # outputs/, data/ and external/ live on the shared volume (all gitignored in the repo).
  for d in outputs data external; do
    mkdir -p "/data/$d"
    ln -sfn "/data/$d" "$d"
  done

  uv sync --locked --extra cu130 --extra wandb
  sfm() { uv run --locked --extra cu130 --extra wandb sfm "$@"; }

  # Build the shared eval sets once; the lock stops concurrent first jobs from racing.
  (
    flock 9
    if [[ ! -f data/eval/sim_iid.parquet || ! -f data/eval/sim_ood_season.parquet ]]; then
      sfm make-sim
    fi
    if [[ ! -f data/eval/m4_monthly.parquet || ! -f data/eval/m4_quarterly.parquet ]]; then
      if [[ ! -d external/M4-data/Dataset ]]; then
        rm -rf external/M4-data
        git clone -q --depth 1 --filter=blob:none --sparse \
          https://github.com/Mcompetitions/M4-methods.git external/M4-data
        git -C external/M4-data sparse-checkout set Dataset
      fi
      sfm stage-m4
    fi
  ) 9>/data/.eval-sets.lock

  echo "== running: ${RUN_CMD}"
  eval "${RUN_CMD}"
} 2>&1 | tee -a "/data/logs/${JOB_NAME}.log"
SH

indent() { sed "s/^/$(printf '%*s' "$1" '')/"; }

MANIFEST="$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels: { owner: "${NRP_USER}", run: "$(echo "$LABEL" | k8s_name)" }
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: { owner: "${NRP_USER}" }
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.compute.major   # Ampere or newer -> bf16 support
                operator: Gt
                values: ["7"]
      containers:
      - name: run
        image: ${IMAGE}
        command: ["bash", "-c"]
        args:
        - |
$(printf '%s\n' "$POD_SCRIPT" | indent 10)
        env:
        - { name: JOB_NAME, value: "${JOB_NAME}" }
        - { name: REPO, value: "${REPO}" }
        - { name: GIT_REF, value: "${GIT_REF}" }
        - name: RUN_CMD
          value: |
$(printf '%s\n' "$RUN_CMD" | indent 12)
        - name: WANDB_API_KEY
          valueFrom: { secretKeyRef: { name: wandb-api-key, key: api-key } }
        - { name: UV_CACHE_DIR, value: /data/.cache/uv }
        - { name: UV_LINK_MODE, value: copy }
        - { name: HF_HOME, value: /data/.cache/huggingface }
        resources:
          requests: { cpu: "${CPU}", memory: "${MEM}", nvidia.com/gpu: "${GPUS}" }
          limits:   { cpu: "${CPU}", memory: "${MEM}", nvidia.com/gpu: "${GPUS}" }
        volumeMounts:
        - { name: data, mountPath: /data }
        - { name: shm, mountPath: /dev/shm }
      volumes:
      - name: data
        persistentVolumeClaim: { claimName: d05-data }
      - name: shm
        emptyDir: { medium: Memory, sizeLimit: 8Gi }
EOF
)"

if [[ -n "${DRY_RUN:-}" ]]; then
  printf '%s\n' "$MANIFEST"
  exit 0
fi

printf '%s\n' "$MANIFEST" | kubectl apply -f -

echo
echo "Launched ${JOB_NAME} at ${GIT_REF:0:8}"
echo "  logs:   kubectl logs -f job/${JOB_NAME}   (also /data/logs/${JOB_NAME}.log)"
echo "  status: kubectl get pods -l job-name=${JOB_NAME}"
echo "  stop:   kubectl delete job ${JOB_NAME}"
