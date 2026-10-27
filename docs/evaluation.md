# Evaluation

What every model is tested on, how it is scored, and known data caveats. Terms are defined in the [Glossary](glossary.md).

## Eval sets

All eval sets share one parquet format (`item_id`, `context`, `target`, `season_length`; see
`src/sarsim_fm/data/eval_sets.py`), so every model and metric works on every set. New datasets
(DERCONNECT, PLAID) only need a writer for this format.

| Set | Series | Horizon | Season | What it tests |
|---|---:|---:|---:|---|
| `sim_iid` | 2,000 | 18 | 1 | Same SarSim0 distribution as training, seed 1000003 |
| `sim_ood_season` | 2,000 | 18 | 1 | SarSim0 with periods 60–120; training only produces ≤ 52 |
| `m4_quarterly` | 24,000 | 8 | 4 | **Validation.** Real data for choosing between models (sweeps, checkpoints). Official M4 split |
| `m4_monthly` | 48,000 | 18 | 12 | **Test.** The zero-shot real-data result. Never used to choose a model. Official M4 split |

**Validation vs. test.** Pick learning rates, steps and checkpoints by `m4_quarterly`, then report
`m4_monthly` once for the chosen model. If M4 Monthly scores decided which model to keep, it would
stop being a zero-shot test. Simulated scores are not a substitute: `chronos2-ft-5k` is the best
model on both sim sets but no better than pretrained Chronos-2 on M4 Monthly
([results](results.md)). Both M4 subsets come from the same competition, so they are related but
share no series.

Held-out series are generated one per call: SarSim0 draws one set of SARIMA parameters per
vectorized call, so generating them in bulk would yield only a few distinct processes.
Their seeds are far from training seeds (training worker *i* uses `data.seed + i`).

## Metrics

| Metric | Definition | Notes |
|---|---|---|
| sMAPE % | mean over series of symmetric MAPE, ×100 | M4 convention. Inflated on sim sets, where many series sit near zero |
| MASE | MAE of median / in-sample seasonal-naive MAE | Main point metric |
| SQL | 2 × mean pinball loss (q = 0.1…0.9) / same naive scale, averaged per series | Probabilistic MASE; equals MASE for a point forecast. Main metric for sim sets |
| WQL | Σ pinball / Σ \|y\| over the whole set | Chronos/GiftEval convention; dominated by large-magnitude series on sim sets |
| Cov 80 / Cov 98 | share of actuals inside [q0.1, q0.9] / [q0.01, q0.99] | Calibration; Cov 98 is the false-alarm side of a residual anomaly detector |

## Known issues and caveats

- **Exploding SarSim0 series.** About 3% of simulated series grow to SarSim0's ±1e6 clip (some
  alternate sign each step) and ~0.25% have all-zero contexts. Training sees the same mix.
  Candidate fix: reject or regenerate such series in the generator.
- **Little diversity per training batch.** Each `data.batch_size` batch shares at most two SARIMA
  parameter sets (see the held-out note under [Eval sets](#eval-sets)). Prefer gradient accumulation over very large
  micro-batches; more `num_workers` also helps.
- **Fine-tuning is not "simulation alone".** `init: pretrained` starts from `amazon/chronos-2`,
  which was pretrained on real data. `init: scratch` matches the paper's setup.
- **M4 may not be strictly zero-shot for pretrained Chronos models.** Check whether M4 is in
  Chronos-2's pretraining corpus before calling those M4 numbers zero-shot.
- `sfm check` times a raw training loop; inside the HF Trainer, steps take about 2× longer.
