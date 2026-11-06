# SarSim0 forecasting and anomaly detection — D05 Anomaly team

Train Chronos-2 forecasters on synthetic [SarSim0](https://github.com/kashif/sarsim0) series, evaluate them zero-shot on held-out simulated and M4 data, and use forecasts to support anomaly detection. The project provides one YAML-driven CLI: `sfm`.

- **Forecasting:** compare seasonal-naive and pretrained Chronos-2 baselines; fine-tune or train Chronos-2 on SarSim0.
- **Evaluation:** use simulated IID/OOD sets, M4 Quarterly for model selection, and M4 Monthly as the held-out final test.
- **Anomalies:** evaluate rolling forecast-based scores and event grouping on labelled synthetic and UCR datasets. Detector modes and thresholds depend on the chosen config; not every detector is simply a q0.01/q0.99 exceedance test. See [anomaly detection](docs/anomaly-detection.md).
- **Remote GPUs:** `sfm prepare-anomaly-handoff` packages materials for remote evaluation; it does not perform remote training by itself.

For the architecture and terminology, see [architecture](docs/architecture.md) and the [glossary](docs/glossary.md).

## Quick start

**Requires:** Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), network access for first-time package/model downloads, and Python **3.11 or 3.12** (`pyproject.toml`: `>=3.11,<3.13`). Commands use a Linux/macOS shell; NVIDIA CUDA 13 support also requires a compatible driver and platform. The lockfile records the concrete resolved versions.

```bash
git clone https://github.com/DSC-Capstone-Azimuth-AI/chronos2-sarsim0-anomaly-team.git
cd chronos2-sarsim0-anomaly-team

# CPU / laptop: explicitly select the CPU PyTorch distribution.
uv sync --locked --extra cpu
uv run --locked --extra cpu sfm check
```

On a supported **NVIDIA CUDA 13** machine, use the **other** extra, not both:

```bash
nvidia-smi
uv sync --locked --extra cu130
uv run --locked --extra cu130 sfm check --device cuda
```

Use the **same extra on every later `uv run`**. `uv run` synchronizes the environment by default; a bare `uv run sfm ...` may change the selected PyTorch distribution. `--locked` refuses to re-resolve a stale lockfile rather than updating it silently. This is especially important when moving between CPU and GPU machines. For a self-contained setup check, see the optional `bootstrap-env.sh` in the review bundle.

### Pinned packages and what's *not* pinned

The uploaded `pyproject.toml`/`uv.lock` specify `torch==2.14.1` (platform-dependent `+cpu` / `+cu130` wheels), `chronos-forecasting==2.3.2`, `transformers==5.20.2`, and SarSim0 from commit `71306403a60dd22df031453dbc51aa0567babaec`. Direct packages and optional extras resolve through `uv.lock`. For closer reproduction, also standardize the exact Python patch version, uv version, OS/driver, model revision, and dataset revision. Pinning the lockfile does **not** pin the upstream Hugging Face model or the M4 checkout.

## Reproduce the forecasting workflow

Run from the repository root. The CPU example is **CPU-compatible end to end**; for GPU use `--extra cu130` and the GPU-specific training YAML shown below.

### 1. Stage held-out evaluation data

```bash
uv run --locked --extra cpu sfm make-sim

mkdir -p external
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Mcompetitions/M4-methods.git external/M4-data
git -C external/M4-data sparse-checkout set Dataset
git -C external/M4-data rev-parse HEAD  # Record the exact M4 commit
uv run --locked --extra cpu sfm stage-m4
```

The M4 clone tracks the upstream default branch. For a strict reproduction, check out the original experiment's recorded M4 commit before staging. The provided files contain no canonical M4 commit. Simulated data is controlled by `configs/eval/sim-heldout.yaml` (2,000 series each; seeds `1000003`, `2000003`). Anomaly synthetic splits are separate; see the configuration table below.

### 2. Establish baselines

```bash
uv run --locked --extra cpu sfm eval \
  --model seasonal_naive \
  --model chronos2:amazon/chronos-2
 
```

# PatchTST-FM r2: IBM's pretrained PatchTST, zero-shot / fine-tuned / scratch (docs/patchtst-fm.md)
uv run sfm eval --model patchtst_fm
uv run sfm train configs/train/patchtst-fm-ft-5k.yaml

# 4. Every result so far, as a markdown table (paste into the Experiments log)
uv run sfm results


The model ID is not immutable. The uploaded train configs specify `model.base: amazon/chronos-2` but **no commit/revision**. Record the Hugging Face revision actually downloaded; pin it in the model-loading code/config when supported before claiming checkpoint-level reproducibility.

| Flag | Effect |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:1\|mps` | override `runtime.device` (`auto` = CUDA if available, else CPU) |
| `--precision auto\|bf16\|fp32` | `auto` = bf16 on GPUs that support it, fp32 otherwise |
| `--set a.b=value` | override any config field; values are parsed as YAML (`--set eval.datasets=[sim_iid]`) |
| `--overwrite` | replace an existing run directory with the same `name` |
| `sfm check --config X --set data.batch_size=128` | measure s/step and peak VRAM before a long run |

Each run writes `outputs/<name>/`:

```text
run-config.json      resolved config + device, package versions, git SHA, dirty flag, SarSim0 commit,
                     and `timing` (train/eval/total seconds, finish time) once the run ends
train-log.csv        loss, grad norm, learning rate per logging step
checkpoint-final/    loadable with `--model <model.type>:<path>` (e.g. chronos2:<path>, patchtst_fm:<path>)
eval/<set>.parquet   per-series metrics + 13 quantiles (0.01 … 0.99) for every horizon step
eval/summary.json    aggregate metrics per eval set
```

```bash
# CPU configuration: runtime.device=cpu, batch_size=8, 1,000 steps.
uv run --locked --extra cpu sfm train configs/train/chronos2-ft-cpu.yaml

# A distinct 500-step CPU experiment (same CPU-compatible config).
uv run --locked --extra cpu sfm train configs/train/chronos2-ft-cpu.yaml \
  --set train.max_steps=500 --set name=chronos2-ft-500-cpu

uv run --locked --extra cpu sfm results
```

**GPU training** uses the 5,000-step config, which explicitly requires `runtime.device: cuda` (batch 64, gradient accumulation 4):

```bash
uv run --locked --extra cu130 sfm train configs/train/chronos2-ft-5k.yaml
```

The common defaults come from `configs/train/_base-chronos2.yaml`. Not every YAML is device-neutral: `chronos2-ft-cpu.yaml` forces CPU, while `chronos2-ft-5k.yaml` and `chronos2-scratch-20k.yaml` force CUDA. Use the corresponding extra and config, or an explicitly tested `--device` override with memory settings appropriate to that device. Long runs also require the evaluation datasets specified in `eval.datasets` if you want complete comparisons; the current base config comments say missing sets are skipped.

Use M4 Quarterly to tune and choose models. Reserve M4 Monthly for the final evaluation: repeatedly comparing candidates on the latter makes it a de facto validation set.

## Training configuration matrix

| Config path (under `configs/train/`) | Initialization | Device in YAML | Batch / accumulation | Steps | Run name |
|---|---|---|---|---:|---|
| `chronos2-ft-cpu.yaml` | Pretrained | `cpu` | 8 / 1 | 1,000 | `chronos2-ft-1k-b8` |
| `chronos2-ft-smoke.yaml` | Pretrained | `auto` (inherited) | 2 / 1 | 100 | `chronos2-ft-smoke` |
| `chronos2-ft-5k.yaml` | Pretrained | `cuda` | 64 / 4 | 5,000 | `chronos2-ft-5k` |
| `chronos2-scratch-20k.yaml` | Scratch | `cuda` | 64 / 4 | 20,000 | `chronos2-scratch-20k` |

`auto` and `--device` affect **runtime choice**, not which PyTorch wheel uv installs. The smoke config inherits a default seed of 42 and default forecast context/target lengths of 512/18. Run names are used for output directories, so change `name` for every distinct experiment; `--overwrite` replaces existing output.

## Evaluation and anomaly configuration matrix

| Config path | Role | Important reproducibility inputs |
|---|---|---|
| `configs/eval/sim-heldout.yaml` | Simulated forecasting evaluation | `sim_iid` seed `1000003`; `sim_ood_season` seed `2000003`; 2,000 series per split |
| `configs/anomaly-data/sim-anomaly-dev.yaml` | Synthetic anomaly tuning | 30 series; seed `3100000` |
| `configs/anomaly-data/sim-anomaly-test.yaml` | Synthetic anomaly final test | 30 anomaly series, seed `4100000`; 30 normal series, seed `5100000` |
| `configs/anomaly/synthetic-chronos2.yaml` | Synthetic detector development | `rolling`, `robust_residual`, target FPR `0.02` |
| `configs/anomaly/synthetic-pilot.yaml` | Three-item pilot | Inherits `synthetic-chronos2.yaml` |
| `configs/anomaly/synthetic-locked.yaml` | Locked synthetic test | Requires `outputs/anomaly/synthetic-selection.json` |
| `configs/anomaly/ucr-chronos2.yaml` | Locked UCR evaluation | Requires the same selection artifact; explicit `interval_exceedance` settings |

*Paths above reflect the logical config groups indicated by the uploaded YAML filenames and original README; check those paths in the complete checkout before running. The original upload contained file basenames without their repository directory paths.*

**Do not regenerate or re-tune the locked detector selection during final tests.** Archive `synthetic-selection.json` with its checksum, the source checkpoint/revision used, and the dataset manifests. The provided YAMLs do not establish how selection-file contents and explicit UCR detector fields are reconciled; inspect the implementation before changing those fields. Likewise, the `sim_ood_season` config includes pairs with period `7`, despite its comment asserting that every period is greater than 52; confirm the generator's semantics before describing that split as fully out-of-distribution.

For actual anomaly commands, scoring, event post-processing, and the handoff command, use [docs/anomaly-detection.md](docs/anomaly-detection.md). The uploaded configuration files alone do not verify those CLI interfaces.

## Outputs and run metadata

A training run is documented to create `outputs/<name>/`:

| Output | Purpose |
|---|---|
| `run-config.json` | Resolved config, device, package versions, source revision/dirty flag, SarSim0 commit, timings |
| `train-log.csv` | Loss, gradient norm, learning rate |
| `checkpoint-final/` | Trained model for later evaluation |
| `eval/<set>.parquet` | Per-series evaluation and forecast quantiles |
| `eval/summary.json` | Aggregated evaluation scores |

Record the following **in addition** to the generated metadata: the exact `uv.lock` hash, Python and uv versions, CUDA driver/GPU type, source revision, M4 Git SHA, pretrained Chronos-2 commit, dataset manifest/checksums, and (for locked detector tests) `synthetic-selection.json` hash. Source/config/lock consistency can be statically checked by the `check-project-configs.py` helper included in this review bundle.

## CLI reference

| Parameter | Meaning |
|---|---|
| `--device auto\|cpu\|cuda\|cuda:1\|mps` | Override YAML `runtime.device`; `auto` prefers CUDA when available, otherwise CPU per project README |
| `--precision auto\|bf16\|fp32` | Override precision; auto uses bf16 when supported |
| `--set a.b=value` | Override an individual YAML setting; quote values containing shell-sensitive characters |
| `--overwrite` | Overwrite an existing run directory; destructive |
| `sfm check --config X --set data.batch_size=128` | Check environment and estimate time/peak VRAM for a config |

## Optional Weights & Biases tracking

The project references [its W&B workspace](https://wandb.ai/chewycrunch-ucsd/sarsim-fm) and [docs/wandb.md](docs/wandb.md). Add the tracking extra **without dropping the backend extra**:

```bash
uv sync --locked --extra cu130 --extra wandb
uv run --locked --extra cu130 --extra wandb wandb login
uv run --locked --extra cu130 --extra wandb sfm train configs/train/chronos2-ft-5k.yaml \
  --set wandb.enabled=true --set wandb.group=sarsim-season-ablation
```

**Sweep warning:** the provided `chronos2-ft-500-lr.yaml` W&B sweep command uses bare `uv run` and forces the same `name=chronos2-ft-500-sweep-lr` for all four learning rates. Without unique output names, runs can collide. Fix and test per-agent naming in the runner before launching the sweep; preserving W&B run IDs alone does not make filesystem output paths unique.

## Reproducibility checklist

1. Check out the source SHA and keep the working tree clean, or record an intentional diff.
2. Use an agreed Python patch version, the tracked `uv.lock`, `uv sync --locked`, and the same `--extra` on every `uv run`.
3. Pin/record the external M4 Git SHA and Hugging Face model revision; archive held-out data or checksums.
4. Preserve the exact YAMLs, overrides, and seeds; use unique run names.
5. Preserve the anomaly selection file before evaluating locked splits, and do not tune on M4 Monthly or UCR test.
6. Save hardware/driver, run logs, resolved config, source SHA, lockfile hash, and result artifacts.

Exact bit-for-bit results across CPU and GPU are not guaranteed due to differing kernels and floating-point behavior; start by seeking repeatable procedures and statistically comparable metrics.

## Further reading

- [Architecture](docs/architecture.md) — training step and package layout
- [Evaluation](docs/evaluation.md) — datasets, metrics, limitations
- [Results](docs/results.md) — recorded experiments
- [Anomaly detection](docs/anomaly-detection.md) — detector configuration and evaluation
- [Weights & Biases](docs/wandb.md) — tracking and sweeps
- [Naming](docs/naming.md) — experiment conventions
- [Glossary](docs/glossary.md) — terms

**Current headline** ([details](docs/results.md)): a short fine-tune of Chronos-2 on SarSim0 only (`chronos2-ft-500`)
is the best model on M4 (MASE 0.904 vs. 0.922 pretrained); training 10× longer helps simulated
data but erases the M4 gain; the from-scratch run is still undertrained.