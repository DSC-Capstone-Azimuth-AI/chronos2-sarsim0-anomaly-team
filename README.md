# SarSim0 forecasting and anomaly detection — D05 Anomaly team

Train time-series foundation models (Chronos-2, PatchTST-FM) on synthetic
[SarSim0](https://github.com/kashif/sarsim0) series. Evaluate them zero-shot on held-out simulated
and real data, and use their forecasts to detect anomalies in real building meter data
(DERConnect). Everything runs through one YAML-driven CLI, `sfm`.

- **Forecasting:** fine-tune or train Chronos-2 and PatchTST-FM on SarSim0. Compare against
  seasonal naive and the pretrained models on simulated IID/OOD sets, M4 Quarterly (validation) and
  M4 Monthly (held-out test).
- **Anomaly detection:** score each reading against its forecast, set thresholds on clean data,
  and group flags into events. Evaluated on labelled synthetic data and the UCR anomaly archive.
- **DERConnect:** an offline workflow that stages meter data, runs frozen models and exports
  dashboard files.

For how the code fits together, see [architecture](docs/architecture.md) and the
[glossary](docs/glossary.md).

## Current results

Details are in [docs/results.md](docs/results.md).

- **`chronos2-ft-500` is the best model on M4 Monthly.** It is a short fine-tune of Chronos-2 on
  SarSim0 only: MASE 0.904 vs. 0.922 for the pretrained model.
- **Training 10× longer (`chronos2-ft-5k`) helps on simulated data but erases the M4 gain.**
- **The from-scratch run (`chronos2-scratch-20k`) is still undertrained.**
- **Zero-shot PatchTST-FM is competitive but trails Chronos-2 on M4.**

Anomaly detection and DERConnect results are not recorded here yet.

## Setup

**Requires:** Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.11 or
3.12, and network access for first-time package and model downloads.

```bash
git clone https://github.com/DSC-Capstone-Azimuth-AI/chronos2-sarsim0-anomaly-team.git
cd chronos2-sarsim0-anomaly-team

# CPU / laptop
uv sync --locked --extra cpu
uv run --locked --extra cpu sfm check

# NVIDIA GPU with CUDA 13 (driver >= 580): use this extra instead, not both
uv sync --locked --extra cu130
uv run --locked --extra cu130 sfm check --device cuda
```

**Use the same extra on every `uv sync` and `uv run`.** Leaving it out, or switching it, swaps the
installed PyTorch build. Add other extras in the same command:

- `--extra dev` for pytest and ruff
- `--extra wandb` for experiment tracking
- `--extra notebooks` for plots

The examples below omit `--locked --extra ...` for brevity.

Pinned in `uv.lock`: `torch==2.14.1`, `chronos-forecasting==2.3.2`, `transformers==5.20.2`, and
SarSim0 at commit `71306403`. Not pinned: the Hugging Face model revisions and the M4 checkout.
Record those for strict reproduction.

## Forecasting workflow

### 1. Stage evaluation data

```bash
uv run sfm make-sim     # held-out sim sets from configs/data/sim-heldout.yaml

mkdir -p external
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Mcompetitions/M4-methods.git external/M4-data
git -C external/M4-data sparse-checkout set Dataset
git -C external/M4-data rev-parse HEAD   # record the M4 commit
uv run sfm stage-m4
```

### 2. Baselines

```bash
uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2 --model patchtst_fm
```

### 3. Train

```bash
uv run sfm train configs/train/chronos2-ft-cpu.yaml   # laptop CPU, ~25 min
uv run sfm train configs/train/chronos2-ft-5k.yaml    # one GPU, ~2 h on a 3070 Ti
uv run sfm results                                    # every run as a markdown table
```

| Config (`configs/train/`) | Init | Device | Batch × accum | Steps | LR |
|---|---|---|---|---:|---:|
| `chronos2-ft-smoke.yaml` | pretrained | auto | 2 × 1 | 100 | 1e-6 |
| `chronos2-ft-cpu.yaml` | pretrained | cpu | 8 × 1 | 1,000 | 1e-5 |
| `chronos2-ft-5k.yaml` | pretrained | cuda | 64 × 4 | 5,000 | 1e-5 |
| `chronos2-scratch-20k.yaml` | scratch | cuda | 64 × 4 | 20,000 | 1e-4 |
| `patchtst-fm-ft-smoke.yaml` | pretrained | auto | 2 × 1 | 100 | 1e-6 |
| `patchtst-fm-ft-cpu.yaml` | pretrained | mps | 4 × 2 | 1,000 | 1e-5 |
| `patchtst-fm-ft-5k.yaml` | pretrained | cuda | 16 × 16 | 5,000 | 1e-5 |
| `patchtst-fm-scratch-20k.yaml` | scratch | cuda | 16 × 16 | 20,000 | 1e-4 |

Shared defaults are in `_base-chronos2.yaml` and `_base-patchtst-fm.yaml`. **Use M4 Quarterly to
choose models and keep M4 Monthly for the final test.** Comparing candidates repeatedly on M4
Monthly turns it into a validation set.

### Common flags

| Flag | Effect |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:1\|mps` | Override `runtime.device` (`auto` = CUDA if available, else CPU) |
| `--precision auto\|bf16\|fp32` | `auto` = bf16 on GPUs that support it, fp32 otherwise |
| `--set a.b=value` | Override any config field; values are parsed as YAML (`--set eval.datasets=[sim_iid]`) |
| `--overwrite` | Replace an existing run directory with the same `name` (destructive) |

Change `name` for every distinct experiment; run names follow [docs/naming.md](docs/naming.md).
`sfm check --config X --set data.batch_size=128` measures seconds per step and peak VRAM before a
long run.

### Run outputs

Each run writes `outputs/<name>/` (git-ignored):

| File | Contents |
|---|---|
| `run-config.json` | Resolved config, device, package versions, git SHA and dirty flag, SarSim0 commit, timing |
| `train-log.csv` | Loss, gradient norm and learning rate per logging step |
| `checkpoint-final/` | Loadable with `--model <type>:<path>`, e.g. `chronos2:outputs/<name>/checkpoint-final` |
| `eval/<set>.parquet` | Per-series metrics and 13 quantiles for every horizon step |
| `eval/summary.json` | Aggregate metrics per eval set |

## Anomaly detection

Full guide: [docs/anomaly-detection.md](docs/anomaly-detection.md).

```bash
uv run sfm make-anomaly-sim configs/data/sim-anomaly-dev.yaml
uv run sfm make-anomaly-sim configs/data/sim-anomaly-test.yaml
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model chronos2:amazon/chronos-2
uv run sfm select-anomaly outputs/anomaly/synthetic-dev-chronos2_amazon_chronos-2
uv run sfm detect configs/anomaly/synthetic-locked.yaml --model chronos2:amazon/chronos-2
uv run sfm anomaly-results
```

| Config | Role |
|---|---|
| `configs/data/sim-anomaly-dev.yaml` | Synthetic tuning set: 30 series, seed 3100000 |
| `configs/data/sim-anomaly-test.yaml` | Synthetic test set: 30 anomaly series and 30 normal series |
| `configs/anomaly/synthetic-chronos2.yaml` | Detector development on the tuning set |
| `configs/anomaly/synthetic-locked.yaml` | Locked synthetic test; requires `outputs/anomaly/synthetic-selection.json` |
| `configs/anomaly/ucr-chronos2.yaml` | Locked UCR evaluation; requires the same selection file |

**Don't regenerate or re-tune the detector selection during final tests.** Archive
`synthetic-selection.json` and its checksum with the results. Use `sfm prepare-anomaly-handoff` to
package a run for a remote GPU.

## DERConnect

DERConnect building meter data has its own offline workflow, with versioned file schemas for the
dashboard:

- [docs/derconnect.md](docs/derconnect.md): staging, detection, evaluation, export
- [docs/derconnect-output-contract.md](docs/derconnect-output-contract.md): dashboard file contract
- [docs/derconnect-alert-policy.md](docs/derconnect-alert-policy.md): schedule-policy overlays

The raw data has no anomaly labels, so nothing in the repo yet measures detection accuracy on real
meters.

## Shared infrastructure

| Tool | Use | Guide |
|---|---|---|
| Hugging Face Hub | Trained checkpoints in the private [azimuth-anomaly-detection](https://huggingface.co/azimuth-anomaly-detection) org. Load with `--model chronos2:azimuth-anomaly-detection/chronos2-ft-5k`; upload with `sfm push outputs/<run>` | [docs/huggingface.md](docs/huggingface.md) |
| Weights & Biases | Optional shared run log at [chewycrunch-ucsd/sarsim-fm](https://wandb.ai/chewycrunch-ucsd/sarsim-fm). Enable with `--set wandb.enabled=true` | [docs/wandb.md](docs/wandb.md) |
| NRP (Nautilus) | Team GPU namespace. `scripts/nrp-run.sh <label> "<cmd>"` runs any `sfm` command as a Job | [docs/nrp.md](docs/nrp.md) |

## Tests

```bash
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu --extra dev pytest -q
uv run --locked --extra cpu --extra dev ruff check .
```

CI runs the offline test suite on every push
([.github/workflows/derconnect-offline.yml](.github/workflows/derconnect-offline.yml)).

## Reproducibility checklist

1. Start from a clean working tree at a recorded commit.
2. Use `uv sync --locked` with the same extras everywhere.
3. Record the M4 commit and the Hugging Face model revisions.
4. Keep the exact YAMLs, `--set` overrides and seeds; use a unique run name.
5. Freeze `synthetic-selection.json` before locked tests; never tune on M4 Monthly or UCR.
6. Keep `run-config.json`, logs and hardware details with the results.

CPU and GPU results are not bit-identical. Compare runs statistically.

## Docs

| Doc | Contents |
|---|---|
| [architecture](docs/architecture.md) | Training step, package layout, adding a model |
| [sarsim0](docs/sarsim0.md) | Simulator configuration, training stream, held-out splits |
| [evaluation](docs/evaluation.md) | Datasets, metrics, limitations |
| [results](docs/results.md) | Experiment log |
| [patchtst-fm](docs/patchtst-fm.md) | PatchTST-FM r2 integration |
| [anomaly-detection](docs/anomaly-detection.md) | Detector, thresholds, events, UCR |
| [derconnect](docs/derconnect.md) | DERConnect workflow |
| [huggingface](docs/huggingface.md) · [wandb](docs/wandb.md) · [nrp](docs/nrp.md) | Shared infrastructure |
| [naming](docs/naming.md) · [glossary](docs/glossary.md) | Conventions and terms |
