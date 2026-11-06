#!/usr/bin/env bash
# Launch any repo CLI command as an NRP GPU Job.
#
#   scripts/nrp-run.sh <run-name> "<command>" [git-ref]
#
# Examples:
#   scripts/nrp-run.sh timing "python -m d05.train --config configs/nrp_a100.yaml --max-steps 100"
#   scripts/nrp-run.sh chronos-small-01 "python -m d05.train --config configs/chronos_small.yaml" main
#
# Env overrides: REPO (owner/name), CPU, MEM, GPUS, IMAGE, INSTALL_CMD
# Watch:  kubectl logs -f job/<job-name>
# Outputs land in /data/runs/<job-name>/ on the d05-data volume.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run-name> \"<command>\" [git-ref]" >&2
  exit 1
fi

RUN_NAME="$1"
RUN_CMD="$2"
GIT_REF="${3:-$(git rev-parse HEAD)}"

REPO="${REPO:-OWNER/REPO}"   # <-- set your GitHub owner/repo
CPU="${CPU:-12}"
MEM="${MEM:-48Gi}"
GPUS="${GPUS:-1}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.04-py3}"
INSTALL_CMD="${INSTALL_CMD:-pip install -q -e .}"

# k8s names: lowercase, digits, dashes, <= 63 chars
JOB_NAME="$(echo "david-${RUN_NAME}-$(date +%m%d-%H%M)" | tr '[:upper:]_' '[:lower:]-' | cut -c1-63)"

# Warn if the ref you're launching hasn't been pushed (the pod clones from GitHub)
if [[ -z "$(git branch -r --contains "$GIT_REF" 2>/dev/null)" ]]; then
  echo "warning: $GIT_REF may not be pushed; the Job clones from GitHub." >&2
fi

indent() { sed 's/^/            /'; }

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels: { owner: david, run: "${RUN_NAME}" }
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
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
          set -euo pipefail
          git clone -q "https://\${GIT_USER}:\${GITHUB_TOKEN}@github.com/${REPO}.git" /work
          cd /work && git checkout -q "${GIT_REF}"
          ${INSTALL_CMD}
          export RUN_DIR="/data/runs/${JOB_NAME}"
          mkdir -p "\$RUN_DIR"
          git rev-parse HEAD > "\$RUN_DIR/git-sha.txt"
          nvidia-smi --query-gpu=name,memory.total --format=csv | tee "\$RUN_DIR/gpu.txt"
          eval "\$RUN_CMD" 2>&1 | tee "\$RUN_DIR/train.log"
        env:
        - name: GIT_USER
          valueFrom: { secretKeyRef: { name: chronos2-github, key: username } }
        - name: GITHUB_TOKEN
          valueFrom: { secretKeyRef: { name: chronos2-github, key: token } }
        - name: WANDB_API_KEY
          valueFrom: { secretKeyRef: { name: wandb-api-key, key: api-key } }
        - name: RUN_CMD
          value: |
$(printf '%s\n' "$RUN_CMD" | indent)
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

echo
echo "Launched ${JOB_NAME} at ${GIT_REF:0:8}"
echo "  logs:   kubectl logs -f job/${JOB_NAME}"
echo "  status: kubectl get pods -l job-name=${JOB_NAME}"
echo "  stop:   kubectl delete job ${JOB_NAME}"