# GPU runs on NRP (Nautilus)

NRP is a shared national Kubernetes cluster with hundreds of GPU nodes (A100, H100, L40/L40S, A6000, …).
Our group has its own namespace there, so any `sfm` command can run on a big GPU with one line from
your laptop. Results land on a shared volume; W&B logging works the same as on any other machine.

- Namespace: **`ucsd-ubellur-ugcapstone-d05`**
- Launcher: [`scripts/nrp-run.sh`](../scripts/nrp-run.sh) · Manifests: [`k8s/`](../k8s/)
- NRP docs: https://nrp.ai/documentation/ · Policies: https://nrp.ai/documentation/userdocs/start/policies/

## Contents

- [How it works](#how-it-works)
- [One-time setup](#one-time-setup)
- [Cheat sheet](#cheat-sheet)
- [Running a job](#running-a-job)
- [Where results go](#where-results-go)
- [Dev pod](#dev-pod)
- [Which GPU you get](#which-gpu-you-get)
- [Cluster rules](#cluster-rules)
- [Shared namespace resources](#shared-namespace-resources)
- [Troubleshooting](#troubleshooting)

## How it works

```text
your laptop                           NRP (namespace ucsd-ubellur-ugcapstone-d05)
───────────                           ───────────────────────────────────────────
git push
scripts/nrp-run.sh <label> "<cmd>" ─► Job ─► pod on any free Ampere+ GPU
                                             1. git clone this repo @ the commit you launched from
                                             2. symlink outputs/ data/ external/ → /data (shared volume)
                                             3. uv sync --locked --extra cu130 --extra wandb
                                             4. build sim + M4 eval sets if missing (once, under a
                                                lock so parallel first jobs don't race; then shared)
                                             5. run <cmd>; log to /data/logs/<job>.log
                                             6. exit → GPU released; Job deleted after 24h
```

- **Jobs, not interactive pods, do the training.** Jobs have no runtime limit; interactive pods are
  killed after 6 hours.
- The Job runs the **pushed** commit, not your working tree. Push first.
- `outputs/`, `data/` and `external/` are already gitignored; on NRP they point at the shared
  `d05-data` volume, so every teammate's runs and eval sets live in one place.

## One-time setup

You need an NRP account in our namespace. If `kubectl get pods` (below) says *Forbidden*, ask the TA
(Amir) to add you.

Install `kubectl` and the `kubelogin` (`kubectl oidc-login`) plugin:

```bash
# macOS
brew install kubectl int128/kubelogin/kubelogin

# Linux / WSL (x86_64)
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/ && rm kubectl
curl -LO https://github.com/int128/kubelogin/releases/latest/download/kubelogin_linux_amd64.zip
unzip -o kubelogin_linux_amd64.zip kubelogin && sudo install kubelogin /usr/local/bin/kubectl-oidc_login
rm kubelogin kubelogin_linux_amd64.zip
```

Then, on any OS:

```bash
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL https://nrp.ai/config
kubectl config get-contexts  # note the context name (expected: nautilus)

kubectl get nodes        # opens a browser to log in with UCSD SSO; then prints a long node list
kubectl config set contexts.nautilus.namespace ucsd-ubellur-ugcapstone-d05
kubectl get pods         # "No resources found" or a list of pods = working
```

On WSL, if no browser opens, copy the URL kubelogin prints into a Windows browser. Full NRP
instructions: https://nrp.ai/documentation/userdocs/start/getting-started/.

Set your name once so your Jobs and pods are prefixed with it (defaults to your login name):

```bash
echo 'export NRP_USER=<firstname>' >> ~/.bashrc   # or ~/.zshrc
```

zsh users: pasted `# comments` break commands unless you enable them once:

```bash
echo 'setopt interactivecomments' >> ~/.zshrc && source ~/.zshrc
```

Nothing else to create: the shared volume and W&B secret already exist (see
[Shared namespace resources](#shared-namespace-resources)).

## Cheat sheet

| Task | Command |
|---|---|
| Timing check (s/step, peak VRAM) | `scripts/nrp-run.sh check "sfm check --device cuda --steps 100"` |
| 100-step smoke train | `scripts/nrp-run.sh smoke "sfm train configs/train/chronos2-ft-smoke.yaml --set name=$NRP_USER-smoke"` |
| Real run with W&B | `scripts/nrp-run.sh ft-5k "sfm train configs/train/chronos2-ft-5k.yaml --set name=chronos2-ft-5k-nrp --set wandb.enabled=true"` |
| Real run, uploaded to the Hub | `scripts/nrp-run.sh ft-5k "sfm train configs/train/chronos2-ft-5k.yaml --set name=chronos2-ft-5k-nrp --set wandb.enabled=true --set hub.enabled=true"` (needs the `hf-token` secret, [huggingface.md](huggingface.md#on-nrp)) |
| Run under another name | `NRP_USER=ali scripts/nrp-run.sh ...` |
| More CPU/RAM | `CPU=16 MEM=64Gi scripts/nrp-run.sh ...` |
| Run a specific branch/commit | `scripts/nrp-run.sh <label> "<cmd>" <ref>` |
| Print the Job manifest, don't launch | `DRY_RUN=1 scripts/nrp-run.sh <label> "<cmd>"` |
| List pods | `kubectl get pods` |
| Follow logs | `kubectl logs -f job/<job-name>` |
| Why is it Pending? | `kubectl describe pod <pod> \| tail -20` |
| Stop a run | `kubectl delete job <job-name>` |

## Running a job

```bash
git push
scripts/nrp-run.sh <label> "<command>"
```

- Inside `<command>`, **`sfm` already means** `uv run --locked --extra cu130 --extra wandb sfm`, so
  write commands exactly as in the README but with plain `sfm`.
- `<label>` only names the Job (`<NRP_USER>-<label>-<MMDD-HHMM>`). The run's output folder is still
  the config's `name`. **Change `name` for every distinct experiment** (`--set name=...`); the volume
  is shared and `sfm` refuses to overwrite an existing run. Names from [naming.md](naming.md) that
  were already used locally (e.g. `chronos2-ft-5k`) may still be free on the volume, but pick a new
  one if it's a different run.
- The first job on a fresh volume is slower: it fills the uv and Hugging Face caches and builds the
  eval sets. Later jobs reuse all three.
- Feed the GPU: SarSim0 generates data on the CPU. For long runs add
  `--set data.num_workers=8` (and keep `CPU` above the worker count).

Watching a job:

```bash
kubectl get pods                    # Pending → ContainerCreating → Running → Completed / Error
kubectl logs -f job/<job-name>      # Ctrl-C stops watching, not the job
```

## Where results go

| On the volume | What |
|---|---|
| `/data/outputs/<name>/` | the usual run folder: `run-config.json`, `train-log.csv`, `checkpoint-final/`, `eval/` |
| `/data/logs/<job-name>.log` | full stdout/stderr of the job |
| `/data/data/eval/` | shared eval sets (`sim_iid`, `sim_ood_season`, `m4_quarterly`, `m4_monthly`) |
| `/data/external/M4-data/` | M4 checkout used by `sfm stage-m4` |
| `/data/.cache/` | uv and Hugging Face caches |

With `--set wandb.enabled=true`, runs also appear in the shared W&B project ([docs/wandb.md](wandb.md)).
With `--set hub.enabled=true`, the trained model is uploaded to the team's Hugging Face org ([huggingface.md](huggingface.md)).
To copy files to your laptop, start a [dev pod](#dev-pod) and:

```bash
kubectl cp $NRP_USER-dev:/data/outputs/<name> ./outputs/<name>
kubectl delete pod $NRP_USER-dev    # right after; it holds a GPU
```

## Dev pod

[`k8s/dev-pod.yaml`](../k8s/dev-pod.yaml) is an interactive GPU pod with the shared volume mounted.
Use it to debug a failing job or to read/copy results. The manifest uses a `__USER__` placeholder that `sed` fills in with
your name (applying the file directly fails on purpose):

```bash
sed "s/__USER__/$NRP_USER/g" k8s/dev-pod.yaml | kubectl apply -f -
kubectl get pods -w                       # wait for Running
kubectl exec -it $NRP_USER-dev -- bash    # setup commands are in the file's header comment
kubectl delete pod $NRP_USER-dev          # always, when done
```

The dev pod lives at most 6 hours and holds a GPU the whole time, so delete it as soon as you're done.

## Which GPU you get

Jobs and the dev pod request **any GPU with CUDA compute capability ≥ 8 (Ampere or newer)**, via
the node label `nvidia.com/gpu.compute.major > 7`. That guarantees bf16 and avoids old cards like the
V100 (no bf16) and 2080 Ti (11 GB, no bf16), while leaving a large pool to schedule from.

- **Memory.** Chronos-2 (~120M) fits on any of these. PatchTST-FM (~385M) peaks at ~8.5 GB at
  micro-batch 16 (`patchtst-fm-ft-5k`), which fits full cards (A10/A40/A6000/L40/A100/H100) but may
  not fit small ones or MIG slices of an A100. The job log prints the GPU name and memory first; if
  it runs out, lower `data.batch_size` and raise `train.grad_accum` to keep their product.
- **Driver.** `--extra cu130` installs torch built for CUDA 13, which needs NVIDIA driver ≥ 580.
  The job log prints the driver version. If torch reports no CUDA device on an older node, see
  [Troubleshooting](#troubleshooting).

To see what exists: `kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.compute.major`.
Pinning one model (e.g. only H100s) is possible with an `nvidia.com/gpu.product` affinity, but
usually means waiting much longer.

## Cluster rules

Breaking these gets the namespace flagged or users banned; see the
[NRP policies](https://nrp.ai/documentation/userdocs/start/policies/).

- **Never `sleep` in a Job.** Sleeping is only OK in interactive pods (the dev pod).
- **Don't hold idle GPUs.** Delete dev pods when done; utilization in the namespace should stay above 40%.
- **Interactive pods**: max 6 hours, 2 GPUs, 32 GB RAM, 16 cores.
- **Limits within 20% of requests.** The launcher sets them equal. Exceeding the memory limit kills the pod (OOMKilled).
- **One GPU per job** unless you've measured that a single GPU is near 100% busy.
- Check the *Violations* page on the NRP portal if anything looks off.

## Shared namespace resources

Already created; **don't recreate them**, and check with the team before deleting anything.

| Resource | Purpose |
|---|---|
| PVC `d05-data` (100 Gi, `rook-cephfs`, ReadWriteMany) | shared `/data` volume ([`k8s/pvc.yaml`](../k8s/pvc.yaml) is the record of how it was made) |
| Secret `wandb-api-key` (key `api-key`) | sets `WANDB_API_KEY` in jobs |
| Secret `hf-token` (key `token`), optional | sets `HF_TOKEN` in jobs for Hub uploads ([huggingface.md](huggingface.md#on-nrp)) |
| Secret `chronos2-github` | Git credentials; unused since the repo is public |

Everyone in the namespace can see and `exec` into everyone's pods and read these secrets. Prefix your
pods/jobs with your name (`NRP_USER`) and only delete your own.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Forbidden ... cannot list resource "pods"` | Not in the namespace, or stale token. `kubectl oidc-login clean`, retry; else ask Amir to add you. Make sure you use the `-d05` namespace. |
| Pod stuck in `Pending` | GPUs busy. `kubectl get pod <pod> -o jsonpath='{.status.conditions[*].message}'`: `Insufficient nvidia.com/gpu` = wait (it starts by itself); `untolerated taint` = reserved nodes, ignore. |
| `Failed to find C compiler` (Triton) | Image without gcc. Use the default `ghcr.io/astral-sh/uv:python3.11-bookworm`, not a `-slim` image. |
| `warning: ... isn't on any pushed branch` | Push first; the Job clones from GitHub. |
| `CUDA driver version is insufficient` / torch sees no GPU | Node's driver is older than CUDA 13 needs (≥ 580; the log prints it). Relaunch to land on another node, and tell the team so we can add a node filter. |
| Job stuck before running, log ends at eval sets | Another job is building the eval sets and holds `/data/.eval-sets.lock`; it continues when that finishes. If no other job is running, a crashed one may have left partial files: delete them in `/data/data/eval/` from a dev pod. |
| `outputs/<name> already has a run` | Name clash on the shared volume. `--set name=<new-name>`. |
| `OOMKilled` | Raise `MEM=...`, or lower `data.batch_size` / `data.num_workers`. |
| `zsh: bad pattern: #` | Enable `interactivecomments` (see setup) or paste commands without comments. |
| CILogon "User Attributes" page instead of the portal | Login worked; go back to nrp.ai and click Login again. |