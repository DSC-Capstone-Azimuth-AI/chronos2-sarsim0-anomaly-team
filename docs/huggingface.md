# Sharing trained models on the Hugging Face Hub

`outputs/` is gitignored, so a clone of this repo has no trained weights. Models worth sharing go to
the team's Hugging Face organization, **[azimuth-anomaly-detection](https://huggingface.co/azimuth-anomaly-detection)**
(private repos), and anyone in the org loads them by repo id.

- Code: [`src/sarsim_fm/hub.py`](../src/sarsim_fm/hub.py)

## Using a shared model

Join the org (ask an admin for an invite), then once per machine:

```bash
uv run hf auth login          # a read token is enough for downloading
```

Use `<type>:<org>/<run name>` anywhere a local checkpoint path works. It downloads on first use and is
cached after that:

```bash
uv run sfm eval --model chronos2:azimuth-anomaly-detection/chronos2-ft-5k
```

| Repo | What it is |
|---|---|
| `chronos2-ft-5k` | fine-tune, 5k steps × 256, lr 1e-5 (best on simulated data) |
| `chronos2-ft-500` | same config cut to 500 steps |
| `chronos2-ft-1k-b8` | fine-tune, 1k steps × batch 8, CPU |
| `chronos2-scratch-20k` | Chronos-2 architecture from random init, 20k steps (local folder `chronos2-scratch-gpu`) |

Scores are in [results.md](results.md) and in each repo's model card.

## Uploading

Uploading needs a **write** token (https://huggingface.co/settings/tokens, then `uv run hf auth login`).
An `HF_TOKEN` environment variable takes precedence over the login, so an old read-only `HF_TOKEN`
in your shell profile will cause "token is read-only" errors.

Each upload is one commit to `<org>/<run name>` with `checkpoint-final/` at the repo root (so
`from_pretrained` works), plus `run-config.json`, `train-log.csv`, `eval/summary.json` and a generated
model card.

**After training (opt-in).** Off by default, so smoke tests and sweeps don't land on the Hub:

```bash
uv run sfm train configs/train/chronos2-ft-5k.yaml --set name=chronos2-ft-5k-v2 --set hub.enabled=true
```

Login and repo name are checked **before** training starts, so a bad token or a name clash fails in
seconds rather than after hours. The upload runs after eval; if it fails, the run is still in
`outputs/<name>/` and `sfm push` retries it.

**A finished run, by hand.** The folder name becomes the repo name:

```bash
uv run sfm push outputs/chronos2-ft-500-lr3e-5
uv run sfm push outputs/chronos2-scratch-gpu --repo azimuth-anomaly-detection/chronos2-scratch-20k
```

| Setting | `sfm push` flag | Default |
|---|---|---|
| `hub.org` | `--org` | `azimuth-anomaly-detection` |
| `hub.repo` | `--repo` | `<org>/<run name>` |
| `hub.private` | `--public` | private |
| `hub.overwrite` | `--force` | refuse if the repo exists |

An existing repo is never overwritten by default, because teammates may already be using those weights.
With `--force` / `hub.overwrite=true` the upload becomes a new commit; earlier versions stay in the repo
history and can be loaded by revision.

## On NRP

Jobs and dev pods read `HF_TOKEN` from the optional secret `hf-token`. Create it once for the namespace
with a write token that has access to the org:

```bash
kubectl create secret generic hf-token --from-literal=token=hf_...
```

Then add `--set hub.enabled=true` to the job's `sfm train` command ([nrp.md](nrp.md#cheat-sheet)).
