#!/usr/bin/env bash
# Copy files between your laptop and the shared d05-data volume on NRP, through a small CPU-only pod
# (no GPU held). The pod is deleted when the script exits. See docs/nrp.md.
#
#   scripts/nrp-transfer.sh up   <local-path>  <volume-dir>   # e.g. data/anomaly/derconnect-2s /data/data/anomaly
#   scripts/nrp-transfer.sh down <volume-path> <local-dir>    # e.g. /data/outputs/dashboard/x outputs/dashboard
#   scripts/nrp-transfer.sh ls   <volume-path>
#
# Paths on the volume start with /data (the repo's data/ is /data/data, outputs/ is /data/outputs).
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,9p' "$0" >&2
  exit 1
fi
ACTION="$1"

NRP_USER="$(echo "${NRP_USER:-$(id -un)}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-\n' '-')"
POD="${NRP_USER}-transfer"
IMAGE="${IMAGE:-ghcr.io/astral-sh/uv:python3.11-bookworm}"

# An interactive pod (not a Job), so idling is allowed; it removes itself after 2 hours regardless.
kubectl apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  labels: { owner: "${NRP_USER}" }
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 7200
  containers:
  - name: shell
    image: ${IMAGE}
    command: ["sleep", "7200"]
    resources:
      requests: { cpu: "1", memory: 2Gi }
      limits:   { cpu: "1", memory: 2Gi }
    volumeMounts:
    - { name: data, mountPath: /data }
  volumes:
  - name: data
    persistentVolumeClaim: { claimName: d05-data }
EOF
trap 'kubectl delete pod "$POD" --wait=false >/dev/null 2>&1 || true' EXIT
kubectl wait --for=condition=Ready "pod/$POD" --timeout=10m >/dev/null

case "$ACTION" in
  up)
    SRC="${2%/}"; DEST="$3"
    echo "uploading $SRC -> $POD:$DEST/$(basename "$SRC") ($(du -sh "$SRC" | cut -f1))"
    kubectl exec "$POD" -- mkdir -p "$DEST"
    # tar over exec rather than `kubectl cp`: same layout whether or not the target already exists.
    COPYFILE_DISABLE=1 tar -C "$(dirname "$SRC")" -cf - "$(basename "$SRC")" \
      | kubectl exec -i "$POD" -- tar -C "$DEST" -xf -
    kubectl exec "$POD" -- du -sh "$DEST/$(basename "$SRC")"
    ;;
  down)
    SRC="${2%/}"; DEST="$3"
    mkdir -p "$DEST"
    echo "downloading $POD:$SRC -> $DEST/$(basename "$SRC")"
    kubectl exec "$POD" -- tar -C "$(dirname "$SRC")" -cf - "$(basename "$SRC")" | tar -C "$DEST" -xf -
    du -sh "$DEST/$(basename "$SRC")"
    ;;
  ls)
    kubectl exec "$POD" -- ls -la "$2"
    ;;
  *)
    echo "unknown action: $ACTION (up, down or ls)" >&2
    exit 1
    ;;
esac
