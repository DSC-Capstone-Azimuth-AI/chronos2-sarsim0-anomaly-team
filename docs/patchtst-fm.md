# PatchTST-FM r2: pretrained vs. scratch

[PatchTST-FM r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) is IBM's
zero-shot forecasting foundation model in the PatchTST family. It plays the role for PatchTST that
`amazon/chronos-2` plays for Chronos-2, so it gets the same three variants:

| Variant | Spec / config | Question |
|---|---|---|
| Zero-shot | `patchtst_fm` (= `patchtst_fm:ibm-granite/granite-timeseries-patchtst-fm-r2`) | How good is the pretrained model as is? |
| Fine-tuned | `configs/train/patchtst-fm-ft-5k.yaml` (`init: pretrained`) | Does SarSim0 add to broad pretraining? |
| Scratch | `configs/train/patchtst-fm-scratch-20k.yaml` (`init: scratch`) | What does the architecture learn from simulation alone? |

The configs match their Chronos-2 counterparts (`chronos2-ft-5k`, `chronos2-scratch-20k`): same
SarSim0 stream, seed, effective batch, steps, learning rate and schedule. That gives the grid
{Chronos-2, PatchTST-FM} × {zero-shot, fine-tuned, scratch}.

It is a different model from the small paper PatchTST in [patchtst.md](patchtst.md) (`patchtst`),
which has no pretrained weights and stays as a size baseline.
Code: [`src/sarsim_fm/models/patchtst_fm.py`](../src/sarsim_fm/models/patchtst_fm.py).

## The model

| | Chronos-2 | PatchTST-FM r2 |
|---|---|---|
| Pretraining data | Real + synthetic corpora | GiftEvalPretrain subset (real), KernelSynth, CauKer, TSMixup (non-GIFT-Eval sources) |
| Size | ~120M | ~385M: 30 Conformer blocks, d_model 1024, patch 16 / stride 8 |
| Output | 21 quantiles | 99 quantiles (0.01 … 0.99), so every eval level is read off exactly |
| Loss | Quantile loss | Quantile loss over the 99 levels, Hamming-weighted within patches |
| Input | Any length | Any length up to 8,192 minus the forecast slots; padding is masked |

The model code is vendored in `src/sarsim_fm/models/vendor/patchtst_fm/` (Apache-2.0) because
IBM's `granite-tsfm` package pins `torch<2.12` and this project uses 2.14.1. The README there
lists the one change made.

**M4 leakage.** The model card does not mention M4. IBM says the TSMixup portion excludes the
GIFT-Eval evaluation datasets, but check the GiftEvalPretrain dataset list before calling its M4
scores zero-shot.

## How training works

PatchTST-FM forecasts by filling masked slots. At inference IBM's `forward` builds one window:
left padding, the history, then at least 128 masked future slots (two pretraining mask spans), and
reads the first `horizon` slots back. Training (`forecast_window`, `forecast_loss`) builds the
same window, puts the true 18-step SarSim0 future into the first 18 slots, marks the other 110 as
missing, and applies IBM's pretraining loss to the 18. Input scaling uses only the history.

`tests/test_patchtst_fm.py` checks that this training window reproduces IBM's inference output
exactly, and that future values cannot leak into the predictions.

## Hardware

About 385M parameters: AdamW in fp32 needs ~6 GB before activations, so fine-tuning and scratch
training **do not fit an 8 GB GPU** (e.g. the 3070 Ti used for Chronos-2). The 5k and 20k configs
use micro-batch 16 × 16 accumulation (effective 256, as for Chronos-2). Measure first:

```bash
uv run sfm check --config configs/train/patchtst-fm-ft-5k.yaml
```

and lower `data.batch_size` while raising `train.grad_accum` to keep their product at 256. Zero-shot
evaluation and anomaly detection run on any GPU or CPU (slowly on CPU).

For a cheaper scratch run, shrink the architecture and name the run for it:

```bash
uv run sfm train configs/train/patchtst-fm-scratch-20k.yaml \
  --set model.architecture='{n_layer: 12, conv_kernel_size: 3}' --set name=patchtst-fm-scratch-20k-l12
```

## Running it

```bash
# Zero-shot, next to the other baselines (downloads ~1.5 GB once)
uv run sfm eval --model patchtst_fm --model chronos2:amazon/chronos-2 --model seasonal_naive

# 100-step plumbing check, then the matched runs
uv run sfm train configs/train/patchtst-fm-ft-smoke.yaml
uv run sfm train configs/train/patchtst-fm-ft-5k.yaml
uv run sfm train configs/train/patchtst-fm-scratch-20k.yaml

uv run sfm results
```

Pick between runs on `m4_quarterly` (validation) and report `m4_monthly` (test) once
([evaluation](evaluation.md)).

### Anomaly detection

Any `patchtst_fm[:<checkpoint>]` spec works wherever a Chronos-2 spec does:

```bash
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model patchtst_fm
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml \
  --model patchtst_fm:outputs/patchtst-fm-ft-5k/checkpoint-final
```

Rolling mode forecasts one step ahead, which PatchTST-FM does by filling its 128 masked slots and
keeping the first, so it costs about as much per point as a full forecast.
