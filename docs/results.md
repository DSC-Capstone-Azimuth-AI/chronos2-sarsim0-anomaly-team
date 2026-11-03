# Results

Experiments log. Regenerate the raw table with `uv run sfm results`; metrics are defined in [evaluation.md](evaluation.md#metrics). Runs are also on [W&B](https://wandb.ai/chewycrunch-ucsd/sarsim-fm).

From `sfm results` (RTX 3070 Ti for evaluation, context length 512). Runs are named by the
[naming convention](naming.md); the four runs here predate it, so their `outputs/` folders keep the
old names listed there.

| Model | Series seen | Train time | sim_iid MASE | sim_iid SQL | sim_ood_season MASE | sim_ood_season SQL | M4 sMAPE % | M4 MASE | M4 Cov 98 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive | — | — | 1.1901 | 1.1901 | 1.1375 | 1.1375 | 15.99 | 1.2597 | 0.014 |
| `amazon/chronos-2` (pretrained, zero-shot) | — | — | 0.6343 | 0.5255 | 0.6234 | 0.5102 | 13.11 | 0.9221 | 0.973 |
| `patchtst_fm` (IBM PatchTST-FM r2, pretrained, zero-shot; Apple MPS, bf16) | — | — | 0.6186 | 0.5126 | 0.6397 | 0.5267 | 13.88 | 0.9756 | 0.956 |
| `chronos2-ft-1k-b8` (fine-tune, 1k steps × batch 8, lr 1e-5, CPU) | 8k | 24.7 min (CPU) | 0.6289 | 0.5216 | 0.6158 | 0.5045 | 13.08 | 0.9174 | 0.960 |
| `chronos2-ft-500` (fine-tune, `chronos2-ft-5k` cut to 500 steps × 256, lr 1e-5) | 128k | 11.0 min (3070 Ti) | 0.6248 | 0.5176 | 0.6105 | 0.4987 | **12.75** | **0.9038** | 0.973 |
| `chronos2-ft-5k` (fine-tune, full config: 5k steps × 256, lr 1e-5) | 1.28M | 1.83 h (3070 Ti) | **0.6176** | **0.5114** | **0.6054** | **0.4937** | 12.89 | 0.9242 | 0.965 |
| `chronos2-scratch-20k` (random init, SarSim0 only: 20k steps × 64 × grad_accum 4, lr 1e-4 cosine) | 5.12M | 7.21 h (3070 Ti) | 0.7024 | 0.5795 | 0.8300 | 0.6726 | 16.63 | 1.3712 | 0.867 |

Commands:
`sfm train configs/train/chronos2-ft-5k.yaml --set train.max_steps=500 --set name=chronos2-ft-500`,
`sfm train configs/train/chronos2-ft-5k.yaml` and
`sfm train configs/train/chronos2-scratch-20k.yaml`.

- The pretrained Chronos-2 M4 numbers reproduce the original pipeline exactly (sMAPE 0.1311438,
  MASE 0.9220801).
- **`chronos2-ft-500` is the best model on real data (M4).** Training only on SarSim0 lowered M4 MASE by
  2.0%. Paired per-series bootstrap vs. pretrained Chronos-2: ΔMASE −0.018 on M4
  (95% CI [−0.020, −0.017]), −0.010 on sim_iid ([−0.017, −0.003]), −0.013 on sim_ood_season
  ([−0.019, −0.008]); 56–59% of series improved on each set. Interval calibration held: M4 98%
  coverage stayed at 0.973 and 80% coverage moved toward nominal (0.758 → 0.772).
- **Training 10× longer improved simulated data but erased the M4 gain.** `chronos2-ft-5k`
  is the best model on both sim sets (ΔMASE vs. pretrained −0.017 iid, −0.018 OOD; also better
  than `chronos2-ft-500`), but on M4 it is no better than the untouched pretrained model (ΔMASE +0.002,
  CI [−0.000, +0.004]) and clearly worse than `chronos2-ft-500` (+0.020, CI [+0.019, +0.022]). Its M4
  98% coverage also dropped to 0.965. M4 sMAPE (12.89) is still below pretrained, so the loss
  is concentrated in series where MASE penalizes larger errors. Training loss was flat after
  ~500 steps (2.82 → 2.80 averaged over 500-step windows), so the extra 4,500 steps mostly moved
  the model toward SarSim0-specific patterns rather than more general skill. This is a
  sim-to-real gap or forgetting effect: more simulated training stops helping real data.
- **The from-scratch ("simulation alone") run beats seasonal naive but not pretrained Chronos-2,
  and does not yet transfer to M4.** On the sim sets it is clearly better than seasonal naive
  (MASE 0.70 vs. 1.19 iid) but behind every pretrained-init model (0.62–0.63). It generalizes
  worse to unseen seasonal periods (sim_ood_season MASE 0.83 vs. 0.70 iid), where fine-tuned
  models barely change between the two sets. On M4 it is worse than seasonal naive (MASE 1.37
  vs. 1.26, sMAPE 16.63 vs. 15.99), and its intervals are too narrow (M4 98% coverage 0.867).
  Training loss was still falling at the end (4.10 → 3.11 averaged over 2,000-step windows,
  vs. ~2.80 for the fine-tunes), so this run is undertrained rather than a negative result: it
  saw 5.12M series, while the paper trains 250k–500k steps at batch 1,024–4,096 (App. E).
- **Zero-shot PatchTST-FM r2 ([patchtst-fm.md](patchtst-fm.md)) is a competitive pretrained model but
  trails Chronos-2 on real data.** It beats zero-shot Chronos-2 on sim_iid (MASE 0.619 vs. 0.634,
  level with `chronos2-ft-5k`) but is worse on unseen seasonal periods (sim_ood_season 0.640 vs.
  0.623) and on M4 Monthly (MASE 0.976 vs. 0.922, +5.8%; sMAPE 13.88 vs. 13.11). Its intervals are
  slightly narrow on M4 (98% coverage 0.956). It is ~10× slower to evaluate than Chronos-2. Run on
  Apple MPS, where IBM's code uses bf16; on 64 M4 Quarterly series this moved the aggregate
  quantile loss by 0.1% vs. CPU. Not yet checked: whether its pretraining data includes M4.
  Fine-tuning (`patchtst-fm-ft-5k`) needs a GPU with more than 8 GB.
- M4 Quarterly (validation set), zero-shot, context 512, Apple MPS:

  | Model | sMAPE % | MASE | SQL | Cov 80 | Cov 98 |
  |---|---:|---:|---:|---:|---:|
  | Seasonal naive | 12.52 | 1.6022 | 1.6022 | 0.004 | 0.004 |
  | `amazon/chronos-2` | **10.43** | **1.2008** | **0.9581** | 0.756 | 0.981 |
  | `patchtst_fm` (r2) | 10.61 | 1.2223 | 0.9799 | 0.729 | 0.950 |

  Commands: `sfm eval --model patchtst_fm --device mps` and
  `sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2 --device mps --datasets m4_quarterly`.
- Caveats: one training seed per run (the CIs cover eval-set sampling, not run-to-run noise).
  `chronos2-ft-500` used the 200-step warmup sized for 5,000 steps, so 40% of its run was warmup. Only
  the last checkpoint was kept (`save_total_limit: 1`), so we cannot see where between step 500
  and 5,000 M4 performance peaked.
- Next: keep every checkpoint (or evaluate during training) to find where real-data performance
  peaks. Choose that step with a **real-data validation set that is not M4** (e.g. M4 Quarterly
  or a GiftEval dataset); picking checkpoints by M4 Monthly score would make M4 no longer a
  zero-shot test. Also worth trying: a lower fine-tuning learning rate, and a much longer
  from-scratch run (or a larger effective batch), since its loss had not plateaued.
- The CPU fine-tune is a small run (training loss 2.93 → 2.90). Read it as "the pipeline trains
  and does no harm" rather than an improvement. It narrowed M4 intervals (98% coverage
  0.973 → 0.960), which `chronos2-ft-500` did not.
