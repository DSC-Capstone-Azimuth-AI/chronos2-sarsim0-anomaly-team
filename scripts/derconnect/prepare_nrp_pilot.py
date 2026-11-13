"""Prepare an inference-only NRP handoff; never contacts or submits to the cluster."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import shutil
import tarfile
from pathlib import Path

import yaml

from sarsim_fm.anomaly.export import write_json
from sarsim_fm.anomaly.provenance import ROOT, file_hash
from sarsim_fm.config import load_config, validate_config

IMAGE = "ghcr.io/astral-sh/uv:python3.11-bookworm"
JOB_SCRIPT = r'''set -euo pipefail
mkdir -p /work
python3 - <<'CHECK'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
for name, expected in json.loads((root / "upload-sha256.json").read_text()).items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"upload checksum mismatch: {name}")
print("Upload checksums verified")
CHECK
tar -xzf "$RUN_ROOT/code.tar.gz" -C /work
cd /work
export UV_CACHE_DIR=/scratch/uv-cache
export UV_PROJECT_ENVIRONMENT=/scratch/venv
export UV_PYTHON_INSTALL_DIR=/scratch/python
export HF_HOME=/scratch/huggingface
uv sync --locked --extra cu130
nvidia-smi
command -v gcc >/dev/null || { echo 'Image needs a C compiler for model JIT'; exit 1; }
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
uv run --no-sync sfm detect-derconnect "$RUN_ROOT/pilot.yaml" \
  --model "$MODEL_SPEC" --device cuda --out "$RUN_ROOT/run"
uv run --no-sync sfm export-derconnect "$RUN_ROOT/run" \
  --format csv --hourly --out "$RUN_ROOT/dashboard"
uv run --no-sync sfm review-derconnect "$RUN_ROOT/run" --out "$RUN_ROOT/review"
'''


def initialize(out, remote_root, namespace, owner, model, meter, config=None):
    out = Path(out).resolve()
    name = out.name
    for value in (name, namespace, owner):
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
            raise ValueError("run directory name, namespace and owner must be Kubernetes names")
    remote = Path(remote_root)
    if not remote.is_absolute() or remote.parts[:2] != ("/", "data") or ".." in remote.parts:
        raise ValueError("remote root must be an absolute directory below /data")
    cfg = copy.deepcopy(config) if config is not None else load_config(
        ROOT / "configs/derconnect/remote-pilot.yaml",
    )
    cfg.update(name=name, meters=[meter], staged=str(remote / "staged"))
    if config is None:
        cfg["roles"].pop("development", None)
        cfg["roles"].pop("heldout_normal", None)  # No reviewed-normal holdout is claimed.
        cfg["roles"]["evaluation"] = {"start": "2026-03-01T00:00:00Z", "end": "2026-03-08T00:00:00Z"}
        cfg.pop("injections", None)
    validate_config(cfg)
    local_cfg = {**cfg, "staged": str(out / "staged")}
    volume = {"name": "data", "persistentVolumeClaim": {"claimName": "d05-data"}}
    transfer_name = f"{owner}-derconnect-transfer"
    if len(transfer_name) > 63:
        raise ValueError("owner is too long for transfer pod name")
    transfer = {
        "apiVersion": "v1", "kind": "Pod", "metadata": {"name": transfer_name, "namespace": namespace},
        "spec": {"restartPolicy": "Never", "containers": [{
            "name": "transfer", "image": IMAGE, "command": ["sleep", "21600"],
            "resources": {"requests": {"cpu": "1", "memory": "2Gi"},
                          "limits": {"cpu": "1", "memory": "2Gi"}},
            "volumeMounts": [{"name": "data", "mountPath": "/data"}],
        }], "volumes": [volume]},
    }
    resources = {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1", "ephemeral-storage": "30Gi"}
    job = {
        "apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": name, "namespace": namespace},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 86400, "template": {"spec": {
            "restartPolicy": "Never",
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": [{"key": "nvidia.com/gpu.compute.major",
                                                             "operator": "Gt", "values": ["7"]}]}]}}},
            "containers": [{"name": "inference", "image": IMAGE,
                            "command": ["bash", "-c", JOB_SCRIPT],
                            "env": [{"name": "RUN_ROOT", "value": str(remote)},
                                    {"name": "MODEL_SPEC", "value": model}],
                            "resources": {"requests": resources.copy(), "limits": resources.copy()},
                            "volumeMounts": [{"name": "data", "mountPath": "/data"},
                                             {"name": "scratch", "mountPath": "/scratch"}]}],
            "volumes": [volume, {"name": "scratch", "emptyDir": {}}],
        }}},
    }
    out.mkdir(parents=True, exist_ok=False)
    for filename, value in (("local-config.yaml", local_cfg), ("pilot.yaml", cfg),
                            ("transfer-pod.yaml", transfer), ("inference-job.yaml", job)):
        (out / filename).write_text(yaml.safe_dump(value, sort_keys=False))
    settings = {"LOCAL_RUN": str(out), "REMOTE_RUN": str(remote), "RUN_TAG": name,
                "NRP_NAMESPACE": namespace, "TRANSFER_POD": transfer_name}
    (out / "session.env").write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in settings.items()))
    print(f"Created {out}; no cluster commands executed")


def pack(out):
    out = Path(out).resolve()
    if (out / "upload-sha256.json").exists():
        raise FileExistsError("handoff already packed; use a new run directory")
    staged = out / "staged"
    manifest = json.loads((staged / "stage-manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("staging incomplete")
    config = load_config(out / "local-config.yaml")
    if any(meter not in manifest["ranges"] for meter in config["meters"]):
        raise ValueError("configured meter missing from staging")
    for name, expected in manifest["prepared_hashes"].items():
        path = (staged / name).resolve()
        if not path.is_relative_to(staged) or file_hash(path) != expected:
            raise ValueError(f"invalid prepared partition: {name}")
    with tarfile.open(out / "prepared.tar.gz", "w:gz") as archive:
        archive.add(staged / "stage-manifest.json", arcname="stage-manifest.json")
        for name in sorted(manifest["prepared_parts"]):
            archive.add(staged / name, arcname=name)
    with tarfile.open(out / "code.tar.gz", "w:gz") as archive:
        for name in ("src", "configs", "schemas", "scripts", "pyproject.toml", "uv.lock"):
            archive.add(ROOT / name, arcname=name, filter=lambda info: None
                        if "__pycache__" in Path(info.name).parts or info.name.endswith(".pyc") else info)
    write_json(out / "upload-sha256.json", {name: file_hash(out / name) for name in
                                           ("code.tar.gz", "prepared.tar.gz", "pilot.yaml")})
    print(f"Packed verified prepared data and current code in {out}")


def retry(previous, out):
    """Reuse verified minute partitions in a fresh handoff without restaging raw CSVs."""
    from sarsim_fm.anomaly.derconnect import stage_manifest

    previous, out = Path(previous).resolve(), Path(out).resolve()
    config = load_config(previous / "local-config.yaml")
    if len(config["meters"]) != 1 or config.get("annotations"):
        raise ValueError("retry helper currently supports single-meter exploratory handoffs without labels")
    manifest = stage_manifest(config)
    old_job = yaml.safe_load((previous / "inference-job.yaml").read_text())
    container = old_job["spec"]["template"]["spec"]["containers"][0]
    env = {v["name"]: v["value"] for v in container["env"]}
    old_pod = yaml.safe_load((previous / "transfer-pod.yaml").read_text())
    owner = old_pod["metadata"]["name"].removesuffix("-derconnect-transfer")
    remote = Path(env["RUN_ROOT"]).parent / out.name
    initialize(out, str(remote), old_job["metadata"]["namespace"], owner,
               env["MODEL_SPEC"], config["meters"][0], config=config)
    staged = out / "staged"
    staged.mkdir()
    shutil.copyfile(Path(config["staged"]) / "stage-manifest.json", staged / "stage-manifest.json")
    for name in manifest["prepared_parts"]:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(config["staged"]) / name, target)
    pack(out)
    print("Retry handoff ready; raw CSVs were not read and no cluster commands executed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--out", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--namespace", default="ucsd-ubellur-ugcapstone-d05")
    p.add_argument("--owner", default="ali")
    p.add_argument("--meter", default="E2451")
    p.add_argument("--model", default="chronos2:amazon/chronos-2")
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True)
    p = sub.add_parser("retry")
    p.add_argument("--from", dest="previous", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.out, args.remote_root, args.namespace, args.owner, args.model, args.meter)
    elif args.command == "pack":
        pack(args.out)
    else:
        retry(args.previous, args.out)


if __name__ == "__main__":
    main()
