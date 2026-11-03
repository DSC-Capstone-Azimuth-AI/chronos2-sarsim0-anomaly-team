# Architecture

How the code is laid out and what happens inside a run. For setup and commands, see the [README](../README.md).

## One training step, concretely

1. SarSim0 simulates `data.batch_size` series of 6,200 points and drops the first 200 (*burn-in*).
2. A random 530-point window is cut from each: the first 512 points are the *context*, the
   next 18 are the *target*.
3. Chronos-2 reads the context and predicts 21 *quantiles* (0.01 … 0.99) for each of the 18
   future steps.
4. The *quantile loss* scores those predictions against the true 18 values.
5. Backpropagation computes gradients and AdamW updates the weights. With `grad_accum: k`, the
   gradients of k batches are added up before a single update.

Nothing is stored: each step sees brand-new simulated series, so there are no epochs and no fixed
training set to overfit. This is why a separate held-out set is needed to measure progress.

## Layout

```text
configs/
  data/sim-heldout.yaml          held-out SarSim0 eval sets (seeds, sizes, unseen seasonalities)
  train/_base-chronos2.yaml      shared defaults; other train configs `inherit:` it
  train/chronos2-ft-smoke.yaml   original 100-step plumbing check
  train/chronos2-ft-cpu.yaml     fine-tune on a laptop CPU (~25 min)
  train/chronos2-ft-5k.yaml      fine-tune, 5k steps on one GPU (~2 h on a 3070 Ti)
  train/chronos2-scratch-20k.yaml  random init, SarSim0 only ("simulation alone"), 20k steps
  train/_base-patchtst.yaml      PatchTST defaults (same data/eval as Chronos-2)
  train/patchtst-scratch-smoke.yaml  100-step PatchTST plumbing check
  train/patchtst-scratch-20k.yaml    PatchTST matched to chronos2-scratch-20k
  sweeps/chronos2-ft-500-lr.yaml learning-rate grid for the 500-step fine-tune
src/sarsim_fm/
  cli.py          `sfm` commands
  config.py       YAML loading, `inherit:`, `--set key=value` overrides
  runtime.py      device/precision selection, environment + git info for run records
  tracking.py     optional Weights & Biases logging (no-op unless enabled)
  metrics.py      sMAPE, MASE, scaled quantile loss, coverage
  evaluate.py     score any model on any eval set, save per-series quantiles
  data/           eval-set format, M4 staging, SarSim0 stream + held-out generation
  models/         registry, shared HF Trainer setup, one file per model family
                  (chronos2, patchtst, seasonal naive)
tests/            fast unit tests (`uv run pytest`)
docs/             guides: architecture, evaluation, results, patchtst, naming, glossary, wandb
data/  outputs/  external/        generated locally, git-ignored
```

Use notebooks (`notebooks/`, install with `--extra notebooks`) for looking at results: they read
`outputs/**/eval/*.parquet`. Keep training and evaluation in the CLI so every number has a config
and a `run-config.json` behind it.

## What each module does

| Module | Role | Main functions |
|---|---|---|
| `cli.py` | The `sfm` command. Each subcommand reads its config, picks the device, and calls the modules below. | `cmd_check`, `cmd_make_sim`, `cmd_stage_m4`, `cmd_train`, `cmd_eval`, `cmd_results`, `cmd_wandb_import` |
| `config.py` | Reads a YAML config, merges the file named in `inherit:` underneath it, and applies `--set` overrides. | `load_config`, `set_dotted` |
| `runtime.py` | Decides where and how to compute: device (`auto` → CUDA if present, else CPU) and precision (bf16 on capable GPUs). Collects versions, GPU name and git SHA for `run-config.json`. | `resolve_runtime`, `environment_info` |
| `data/sim.py` | Everything SarSim0. Turns the YAML `sarsim:` block into a `SarSimConfig` (defaults: univariate, no covariates, length 6,000), builds the endless training stream, and writes the fixed held-out sets. | `sarsim_config`, `training_stream`, `make_heldout_split` |
| `data/m4.py` | Converts M4 Monthly (test) and Quarterly (validation) CSVs into the eval-set format and checks counts and horizons. | `stage_m4`, `FREQUENCIES` |
| `data/eval_sets.py` | The single on-disk format all eval sets share, with validation on write. | `write_eval_set`, `load_eval_set`, `EvalSet` |
| `models/__init__.py` | The registry: maps a spec like `chronos2:outputs/run/checkpoint-final` to the code that loads, trains or times it. | `load_forecaster`, `get_trainer`, `get_timer`, `Forecaster` |
| `models/hf.py` | Hugging Face `TrainingArguments` built from a config (shared, so every model trains under identical settings), a Trainer for pre-batched streams, and the raw-step timer behind `sfm check`. | `training_arguments`, `StreamTrainer`, `timed_steps` |
| `models/chronos2.py` | Chronos-2 specifics: load a checkpoint for forecasting, build a model (pretrained or random weights), train it with the Hugging Face Trainer, and time raw training steps for `sfm check`. | `load`, `build_model`, `train`, `timed_steps` |
| `models/patchtst.py` | PatchTST specifics: adapt the SarSim0 stream to fixed-length masked inputs, build and train the model, and turn its Student-t head into quantiles. See [patchtst.md](patchtst.md). | `load`, `build_model`, `train`, `PatchTSTStream`, `student_t_quantiles` |
| `models/naive.py` | Seasonal naive baseline. | `SeasonalNaive` |
| `evaluate.py` | Runs any forecaster on any eval set, scores each series, and saves quantiles plus a summary. | `evaluate`, `save_eval` |
| `metrics.py` | The metric formulas. | `smape`, `mase`, `naive_scale`, `scaled_quantile_loss`, `quantile_loss`, `coverage` |
| `tracking.py` | Optional W&B logging: opens a run with the resolved config, writes eval metrics and timing to its summary. Does nothing unless `wandb.enabled` (or under `wandb agent`). | `start_run`, `log_eval`, `finish`, `import_run` |

What `sfm train configs/train/chronos2-ft-cpu.yaml` calls, in order:

```text
config.load_config            read YAML + inherit + --set
runtime.resolve_runtime       pick cpu/cuda, bf16 or fp32
→ write outputs/<name>/run-config.json
models.get_trainer("chronos2") → chronos2.train
    chronos2.build_model          pretrained weights or random init
    data.sim.training_stream      endless SarSim0 batches
    Chronos2Trainer.train()       the training loop (Hugging Face)
    → checkpoint-final/, train-log.csv
models.load_forecaster("chronos2:<checkpoint-final>")
for each eval set in eval.datasets:
    data.eval_sets.load_eval_set → evaluate.evaluate → evaluate.save_eval
```

## Adding a model

1. Write `src/sarsim_fm/models/<name>.py` with `load(path, runtime, context_length)` returning an
   object with `predict_quantiles(contexts, horizon, levels, batch_size, season_lengths)` →
   array `(n_series, horizon, n_levels)`. Add `train(cfg, runtime, run_dir)` if it is trainable
   (`models/hf.py` has the shared `TrainingArguments`) and `timed_steps(cfg, runtime, steps)` for
   `sfm check`. `models/patchtst.py` is a compact example.
2. Register them in `src/sarsim_fm/models/__init__.py`.
3. Add `configs/train/<name>-*.yaml` with `model.type: <name>`.

Next planned: Chronos-Small (T5), the paper's model. Chronos v1 mean-scales and bins values
into tokens and trains with cross-entropy, so it needs an adapter from SarSim0 windows to
Chronos v1 inputs rather than the Chronos-2 batch stream.
