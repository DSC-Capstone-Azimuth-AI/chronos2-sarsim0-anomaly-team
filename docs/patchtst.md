# PatchTST vs. Chronos-2

PatchTST ([Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730);
[Hugging Face blog](https://huggingface.co/blog/patchtst)) is the second model family here,
added so it can be compared with Chronos-2 on the same data, metrics and anomaly pipeline.
Code: [`src/sarsim_fm/models/patchtst.py`](../src/sarsim_fm/models/patchtst.py).

## How it differs from Chronos-2

| | Chronos-2 | PatchTST (this repo) |
|---|---|---|
| Implementation | `chronos-forecasting` | `transformers.PatchTSTForPrediction` |
| Pretrained weights | `amazon/chronos-2`, trained on large real + synthetic corpora | None. Public checkpoints fit one dataset each (e.g. ETTh1), so we always train it on SarSim0 |
| Size | ~120M parameters | ~0.4M with the defaults (PatchTST/64: patch 16, stride 8, d_model 128, 3 layers, 16 heads) |
| Output | 21 quantiles directly | A Student-t per future step; the 13 eval quantiles are computed from it exactly, not sampled |
| Loss | Quantile (pinball) loss | Student-t negative log-likelihood |
| Input length | Any; NaNs allowed | Fixed `data.context_length` (512). Shorter histories are left-filled and masked |
| Horizon | Any | Fixed `data.prediction_length` (18). Eval horizons must be ≤ 18 (all current sets are) |

Two consequences for the comparison:

- **Short histories.** Half the M4 series have fewer than 202 (Monthly) or 88 (Quarterly) points,
  while SarSim0 contexts are always 512. Training therefore masks a random-length prefix of half
  the contexts (`data.short_context`, minimum 16 points) so short inputs are not out of
  distribution. Set `data.short_context.prob: 0` to see what that buys.
- **Model size.** The defaults follow the paper, so PatchTST is ~300× smaller than Chronos-2.
  That is a fair "paper model vs. paper model" comparison, but if PatchTST loses it may simply be
  capacity. Larger variants go through `model.architecture`, e.g.
  `--set model.architecture='{d_model: 256, num_hidden_layers: 6, ffn_dim: 512}'`.

## Running the comparison

PatchTST is small enough to train on a laptop CPU (~0.5 s/step at the 20k config's batch, ≈3 h).
`sfm check --config configs/train/patchtst-scratch-20k.yaml` measures your machine first.

```bash
# Plumbing check (~1 min on CPU): train 100 steps, then evaluate every eval set
uv run sfm train configs/train/patchtst-scratch-smoke.yaml

# Same SarSim0 data, seed, effective batch (256), steps, LR and schedule as chronos2-scratch-20k
uv run sfm train configs/train/patchtst-scratch-20k.yaml
uv run sfm train configs/train/chronos2-scratch-20k.yaml        # GPU (~7 h on a 3070 Ti)

# Pretrained Chronos-2 and seasonal naive, if not already in outputs/baselines/
uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2

uv run sfm results                                             # one table, every run
```

Questions the table answers:

| Rows to compare | Question |
|---|---|
| `patchtst-scratch-20k` vs. `chronos2-scratch-20k` | Which architecture learns more from simulation alone, at equal data and budget? |
| `patchtst-scratch-20k` vs. `baselines/chronos2_amazon_chronos-2` | Does a small sim-trained model compete with a large pretrained foundation model? |
| Either vs. `baselines/seasonal_naive` | Is the model better than repeating the last season (MASE < 1 means yes)? |

Pick between runs on `m4_quarterly` (validation) and report `m4_monthly` (test) once, as for
Chronos-2 ([evaluation](evaluation.md)).

### Anomaly detection

Any `patchtst:<checkpoint>` spec works wherever a Chronos-2 spec does. After generating the
synthetic anomaly splits ([anomaly detection](anomaly-detection.md)):

```bash
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml \
  --model patchtst:outputs/patchtst-scratch-20k/checkpoint-final
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model chronos2:amazon/chronos-2
uv run sfm anomaly-results
```

The detector configs use a 256-step context; PatchTST sees those 256 points and masks the rest
of its 512 input. Rolling mode forecasts one step ahead, which PatchTST does by predicting its
full 18 steps and keeping the first.
