# Training on NRP (Nautilus) GPUs

Kubernetes Jobs in namespace `ucsd-ubellur-ugcapstone` for runs that need more than a laptop or an
8 GB card. Each job:

1. clones a branch and installs the project (`uv sync --extra cu130 --extra wandb`);
2. checks the Hugging Face token can see the model repo, so a bad token fails in seconds, not hours;
3. rebuilds the eval sets and prints `sfm check` (peak GPU memory, seconds per step);
4. runs `sfm train` with W&B on, which trains and then evaluates the final checkpoint;
5. uploads the run to [`teyquila/sarsim0-anomaly-models`](https://huggingface.co/teyquila/sarsim0-anomaly-models)
   and exits, so the Job completes and frees the GPU.

| File | Config | GPU |
|---|---|---|
| `k8s/patchtst-scratch-20k-job.yaml` | `configs/train/patchtst-scratch-20k.yaml` (paper PatchTST, 0.4M) | any current card |
| `k8s/patchtst-fm-ft-5k-job.yaml` | `configs/train/patchtst-fm-ft-5k.yaml` | 80 GB |
| `k8s/patchtst-fm-scratch-20k-job.yaml` | `configs/train/patchtst-fm-scratch-20k.yaml` | 80 GB |

`scripts/nrp-train.sh` is what runs inside the pod; `scripts/hub_upload.py` does the upload.

## Where results go

- **W&B** ([sarsim-fm](https://wandb.ai/chewycrunch-ucsd/sarsim-fm), group `patchtst` or
  `patchtst-fm`): loss and learning-rate curves live, eval metrics in the run summary at the end.
- **Hugging Face Hub**: `<run>/checkpoint-final/` plus `train-log.csv`, `run-config.json`,
  `timing.json` and `eval/`, e.g. `patchtst-scratch-20k/checkpoint-final`. Intermediate checkpoints
  are not uploaded. The upload also runs if evaluation fails after training finished.

Nothing else survives: the pod's disk is deleted with it, so a run that crashes mid-training
leaves only its W&B curves.

## Credentials

Tokens reach the pod as environment variables from the Kubernetes Secret `sarsim-secrets`; they are
never in the manifests or the repo.

| Key | Env var | What |
|---|---|---|
| `wandb-api-key` | `WANDB_API_KEY` | https://wandb.ai/authorize |
| `hf-token` | `HF_TOKEN` | Fine-grained token with write access to `teyquila/sarsim0-anomaly-models` |
| `github-token` | `GITHUB_TOKEN` | Fine-grained, read-only Contents on this (private) repo, for the clone |

Create it once. `read -s` keeps the values out of shell history (zsh syntax; in bash use
`read -sp "prompt: " VAR`):

```bash
read -s "WANDB_API_KEY?W&B key: " && echo && read -s "HF_TOKEN?HF token: " && echo && read -s "GITHUB_TOKEN?GitHub token: " && echo && kubectl create secret generic sarsim-secrets -n ucsd-ubellur-ugcapstone --from-literal=wandb-api-key="$WANDB_API_KEY" --from-literal=hf-token="$HF_TOKEN" --from-literal=github-token="$GITHUB_TOKEN"
```

To rotate a token, `kubectl delete secret sarsim-secrets -n ucsd-ubellur-ugcapstone` and create
it again. You must be a member of the namespace (`kubectl auth can-i create jobs -n
ucsd-ubellur-ugcapstone` says `yes`); ask the namespace admin otherwise.

## Running

The job trains whatever is pushed to the branch in its `BRANCH` variable (`patchtst-chronos2`), so
push first.

```bash
kubectl apply -f k8s/patchtst-scratch-20k-job.yaml
kubectl get pods -n ucsd-ubellur-ugcapstone -l job-name=patchtst-scratch-20k
kubectl logs -f -n ucsd-ubellur-ugcapstone job/patchtst-scratch-20k
```

- A pod stays `Pending` until a GPU of the listed types is free.
- Out of GPU memory in the `sfm check` output: delete the job and add e.g.
  `--set data.batch_size=8 --set train.grad_accum=32` after the config path in the job's `args`
  (keep the product at the config's effective batch, 256).
- `backoffLimit: 0`: a crashed run is not retried, since there is no resume and a retry would
  start from step 0. A finished Job and its logs are deleted after a day
  (`ttlSecondsAfterFinished`); delete it sooner to apply it again:
  `kubectl delete job <name> -n ucsd-ubellur-ugcapstone`.
- Uploading a run name that is already in the repo replaces those files in a new commit.

## Using the checkpoints

```bash
uv run hf download teyquila/sarsim0-anomaly-models --include "patchtst-scratch-20k/*" --local-dir outputs
uv run sfm results
uv run sfm eval --model patchtst:outputs/patchtst-scratch-20k/checkpoint-final
```
