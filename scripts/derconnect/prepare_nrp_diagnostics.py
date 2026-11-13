"""Package a bounded, pinned forecast reproduction; never submits a Kubernetes Job."""
import argparse
import json
import re
import shlex
import tarfile
from pathlib import Path

import yaml

from sarsim_fm.anomaly.derconnect_artifacts import BundleReader, ensure_new_output
from sarsim_fm.anomaly.export import write_json
from sarsim_fm.anomaly.forecast_diagnostics import reproduction_scope
from sarsim_fm.anomaly.provenance import ROOT, file_hash

SCRIPT = r'''set -euo pipefail
python3 - <<'CHECK'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['RUN_ROOT'])
for name, expected in json.loads((root/'upload-sha256.json').read_text()).items():
    digest=hashlib.sha256()
    with (root/name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('upload checksum mismatch: '+name)
CHECK
mkdir -p /work /scratch/input
tar -xzf "$RUN_ROOT/code.tar.gz" -C /work
tar -xzf "$RUN_ROOT/inputs.tar.gz" -C /scratch/input
cd /work
export UV_CACHE_DIR=/scratch/uv-cache
export UV_PROJECT_ENVIRONMENT=/scratch/venv
export UV_PYTHON_INSTALL_DIR=/scratch/python
export HF_HOME=/scratch/huggingface
uv sync --locked --extra cu130
nvidia-smi
command -v gcc >/dev/null
uv run --no-sync python scripts/derconnect/diagnose_forecasts.py reproduce \
  --bundle /scratch/input/dashboard --staged /scratch/input/staged \
  --device cuda --out "$RUN_ROOT/results"
'''


def prepare(source, staged, out, remote_root, namespace="ucsd-ubellur-ugcapstone-d05"):
    reader = BundleReader(source)
    scope = reproduction_scope(source)
    out, staged = Path(out).resolve(), Path(staged).resolve()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,61}[a-z0-9]", out.name):
        raise ValueError("diagnostic folder name must be a Kubernetes job name")
    remote = Path(remote_root)
    if not remote.is_absolute() or remote.parts[:2] != ("/", "data") or ".." in remote.parts:
        raise ValueError("remote root must be below /data")
    manifest = json.loads((staged / 'stage-manifest.json').read_text())
    if not manifest.get('complete') or manifest['preparation_id'] != reader.manifest['preparation_id']:
        raise ValueError('staged preparation differs from source pilot')
    for name, expected in manifest['prepared_hashes'].items():
        path = (staged / name).resolve()
        if not path.is_relative_to(staged) or file_hash(path) != expected:
            raise ValueError('staged hash mismatch')
    out = ensure_new_output(out)
    with tarfile.open(out/'inputs.tar.gz', 'w:gz') as archive:
        for name in sorted(set(reader.manifest['outputs']) | {'run-manifest.json'}):
            archive.add(reader.root/name, arcname=f'dashboard/{name}')
        archive.add(staged/'stage-manifest.json', arcname='staged/stage-manifest.json')
        for name in manifest['prepared_parts']:
            archive.add(staged/name, arcname=f'staged/{name}')
    with tarfile.open(out/'code.tar.gz', 'w:gz') as archive:
        for name in ('src', 'scripts', 'configs', 'schemas', 'pyproject.toml', 'uv.lock'):
            archive.add(ROOT/name, arcname=name, filter=lambda info: None if
                        '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc') else info)
    resources = {'cpu': '4', 'memory': '16Gi', 'nvidia.com/gpu': '1', 'ephemeral-storage': '30Gi'}
    job = {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {'name': out.name, 'namespace': namespace},
           'spec': {'backoffLimit': 0, 'ttlSecondsAfterFinished': 86400, 'template': {'spec': {
               'restartPolicy': 'Never', 'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                   'nodeSelectorTerms': [{'matchExpressions': [{'key': 'nvidia.com/gpu.compute.major',
                                                               'operator': 'Gt', 'values': ['7']}]}]}}},
               'containers': [{'name': 'diagnostics', 'image': 'ghcr.io/astral-sh/uv:python3.11-bookworm',
                               'command': ['bash', '-c', SCRIPT], 'env': [{'name': 'RUN_ROOT', 'value': str(remote)}],
                               'resources': {'requests': resources.copy(), 'limits': resources.copy()},
                               'volumeMounts': [{'name': 'data', 'mountPath': '/data'},
                                                {'name': 'scratch', 'mountPath': '/scratch'}]}],
               'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': 'd05-data'}},
                           {'name': 'scratch', 'emptyDir': {}}]}}}}
    (out/'diagnostic-job.yaml').write_text(yaml.safe_dump(job, sort_keys=False))
    q = shlex.quote
    commands = [
        "# Prepared diagnostic handoff (not submitted)",
        "", "Run on the Mac only after authorizing this bounded NRP diagnostic job.",
        "The CPU transfer pod ali-derconnect-transfer must be Running. If expired, recreate",
        "it using the pilot transfer-pod.yaml before these commands. No GPU is requested by that pod.", "", "```sh",
        f"kubectl -n {q(namespace)} exec ali-derconnect-transfer -- mkdir -p {q(str(remote))}",
    ]
    for name in ('code.tar.gz', 'inputs.tar.gz', 'upload-sha256.json'):
        commands.append(f"kubectl -n {q(namespace)} cp {q(str(out/name))} "
                        f"{q('ali-derconnect-transfer:'+str(remote/name))}")
    commands.extend([
        f"kubectl -n {q(namespace)} create -f {q(str(out/'diagnostic-job.yaml'))}",
        f"kubectl -n {q(namespace)} get pods -l job-name={q(out.name)} -w",
        "# Control-C stops watching. Once Running:",
        f"kubectl -n {q(namespace)} logs -f job/{q(out.name)}",
        "# After completion:",
        f"kubectl -n {q(namespace)} cp {q('ali-derconnect-transfer:'+str(remote/'results'))} {q(str(out/'results'))}",
        "# Verify diagnostic-manifest.json output hashes before releasing resources.",
        f"kubectl -n {q(namespace)} delete job {q(out.name)}", "```", "",
        "Scope: original immutable checkpoint; at most 40 target predictions; no training.",
        "Dependency/model caches are ephemeral scratch; completed results persist under the remote root.",
    ])
    (out/'README.md').write_text("\n".join(commands))
    write_json(out/'scope.json', scope)
    write_json(out/'upload-sha256.json', {name: file_hash(out/name) for name in ('code.tar.gz', 'inputs.tar.gz')})
    write_json(out/'handoff-manifest.json', {'complete': True, 'submitted': False,
                                           'remote_root': str(remote), 'namespace': namespace,
                                           'scope': scope, 'outputs': {p.name: file_hash(p)
                                           for p in out.glob('*') if p.is_file()}})
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('bundle', 'staged', 'out', 'remote-root'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    result = prepare(args.bundle, args.staged, args.out, args.remote_root)
    print(json.dumps({'maximum_target_predictions': result['maximum_target_predictions'], 'submitted': False}))


if __name__ == '__main__':
    main()
