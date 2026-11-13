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

Current models (check the [org page](https://huggingface.co/azimuth-anomaly-detection) for the full list):

| Repo | What it is |
|---|---|
| `chronos2-ft-5k` | fine-tune, 5k steps × 256, lr 1e-5 (best on simulated data) |
| `chronos2-ft-500` | same config cut to 500 steps |
| `chronos2-ft-1k-b8` | fine-tune, 1k steps × batch 8, CPU |
| `chronos2-scratch-20k` | Chronos-2 architecture from random init, 20k steps (local folder `chronos2-scratch-gpu`) |

Scores are in [results.md](results.md) and in each repo's model card.

## Uploading

Uploading needs a token that can write to the org. At <https://huggingface.co/settings/tokens> either
make a **Write** token, or a **Fine-grained** token with *Repositories → Write* granted for the
`azimuth-anomaly-detection` org (a fine-grained token without it passes the pre-training check but
fails at upload time). Then `uv run hf auth login`.
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
`outputs/<name>/` and `sfm push outputs/<name>` retries it. If the failed attempt had already
created the repo (check the org page), add `--force`, or the retry stops with "already exists".

**A finished run, by hand.** The folder name becomes the repo name:

```bash
uv run sfm push outputs/chronos2-ft-500-lr3e-5
uv run sfm push outputs/chronos2-scratch-gpu --repo azimuth-anomaly-detection/chronos2-scratch-20k
```

| Setting (`sfm train --set ...`) | `sfm push` flag | Default |
|---|---|---|
| `hub.org` | `--org` | `azimuth-anomaly-detection` |
| `hub.repo` | `--repo` | `<org>/<run name>` |
| `hub.private` | `--public` | private (`--set hub.private=false` for public) |
| `hub.overwrite` | `--force` | refuse if the repo exists |

Don't confuse this with `sfm train --overwrite`, which deletes and replaces the **local**
`outputs/<name>/` folder. The two are independent: re-training under an existing name needs
`--overwrite` for the folder *and* `--set hub.overwrite=true` for the Hub repo.

An existing repo is never overwritten by default, because teammates may already be using those weights.
With `--force` / `hub.overwrite=true` the upload becomes a new commit on top, and `sfm` loads the
latest one.

## On NRP

Jobs and dev pods read `HF_TOKEN` from the optional secret `hf-token`. Create it once for the namespace
with a write token (or fine-grained token with write access to the org, see [Uploading](#uploading)):

```bash
kubectl create secret generic hf-token --from-literal=token=hf_...
```

Then add `--set hub.enabled=true` to the job's `sfm train` command ([nrp.md](nrp.md#cheat-sheet)).
