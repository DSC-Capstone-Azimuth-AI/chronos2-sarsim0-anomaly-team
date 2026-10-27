# Glossary

Terms used across the README and docs.

**Time-series basics**

| Term | Meaning |
|---|---|
| Time series | Values recorded at regular intervals, e.g. hourly kWh for one home. |
| Context (history) | The past values the model gets to see. 512 steps here; M4 uses each series' full training history, cut to the last 512. |
| Horizon / prediction length / target | How many future steps to forecast (18 here, M4 Monthly's official horizon), and the true values for those steps. |
| Seasonality, period *s* | A pattern that repeats every *s* steps: 24 (time of day, hourly data), 7 (day of week, daily data), 12 (month of year). |
| Trend, integration | Slow drift in level. An *integrated* series is a running sum of another series, so it wanders like a random walk. |
| Point vs. probabilistic forecast | One number per future step vs. a whole distribution of likely values. |
| Quantile *q* | A value the actual should fall below with probability *q*. q0.5 is the median; [q0.1, q0.9] is an 80% interval. |

**SarSim0**

| Term | Meaning |
|---|---|
| SARIMA | Seasonal AutoRegressive Integrated Moving Average, a classic statistical model. Each value is a weighted sum of recent values (AR, order *p*) and recent random shocks (MA, order *q*), plus the same at seasonal lags *s, 2s, …* (orders *P, Q*), then integrated (*d*, *D*). |
| SarSim0 | A simulator that draws **new random SARIMA settings for every series**, so it can produce endless, varied training data. Each series is built as y = N(I(S(ε))): |
| · S, base signal | SARIMA with random orders, coefficients constrained so the series stays stable, period *s* drawn from 0–52 (0 = no seasonality), fractional integration *d* ∈ [0, 1]. |
| · I, SARIMA-2 | For half the series, a second seasonal series (the "envelope") is added to or multiplied with the base, e.g. a daily cycle whose strength changes over the week. Period pairs come from paper Table 9: (24,7), (7,52), (0,7), (0,4), (0,24), (0,52). |
| · N, noiser | Heavy-tailed observation noise: Poisson (count-like), Gamma or Lognormal. |
| Burn-in | The first 200 simulated points are discarded so start-up effects don't appear in the data. |
| Seed | The number that fixes the random generator: the same seed gives the same series. Training and held-out sets use far-apart seeds. |

**Models**

| Term | Meaning |
|---|---|
| Foundation forecaster | A large model trained on many series and then used on new series without retraining. |
| Chronos-2 (`amazon/chronos-2`) | Amazon's ~120M-parameter transformer forecaster. It scales each series, reads the context in 16-step patches, and directly outputs 21 quantiles (0.01 … 0.99) for each future step. Amazon pretrained it on real and synthetic data. |
| Chronos v1 / Chronos-Small (T5) | The earlier Chronos family and the model the SarSim0 paper trains. It turns values into tokens (scale, then bin into a fixed vocabulary), predicts the next token like a language model, and samples to get forecast distributions. Not in this repo yet. |
| Seasonal naive | Baseline that repeats the last season (last 12 months for M4; the last value for sim sets, where season = 1). A model that can't beat it isn't useful. |
| Checkpoint | Saved model weights, loadable for forecasting or further training. |

**Training**

| Term | Meaning |
|---|---|
| Fine-tune (`init: pretrained`) | Start from Amazon's trained weights and keep training on SarSim0. |
| From scratch (`init: scratch`) | Same architecture, random weights, trained only on SarSim0: the paper's "simulation alone" setup. |
| Step | One weight update (optimizer step). |
| Batch size | Series per forward pass (`data.batch_size`). |
| Gradient accumulation | Add up gradients from `grad_accum` batches before one update. Effective batch = `batch_size × grad_accum`, without the extra memory. |
| Learning rate, warmup, scheduler | Size of each update; warmup ramps it up over the first `warmup_steps`, then the scheduler (linear/cosine) decays it. |
| Quantile (pinball) loss | The training objective. For level *q*, under-predictions cost *q* × error and over-predictions cost (1 − *q*) × error, so each output learns its quantile. |
| bf16 / tf32 / fp32 | Number precision. bf16 and tf32 are faster, lower-memory GPU formats; CPUs use full fp32. |
| Streaming, `num_workers` | Training data is generated on the fly, never stored; workers generate batches in parallel with training. |

**Evaluation**

| Term | Meaning |
|---|---|
| Zero-shot | Forecasting a dataset without ever updating the model on it. |
| Held-out set | Fixed test data that is never used for training or tuning. |
| IID vs. OOD | *In-distribution*: drawn the same way as training data (`sim_iid`). *Out-of-distribution*: deliberately different (`sim_ood_season`, periods 60–120). |
| M4 Monthly | 48,000 real monthly series (finance, industry, demographics, …) from the 2018 M4 forecasting competition, with an official 18-month test period. |
| Metrics | sMAPE, MASE, SQL, WQL, coverage: see [Metrics](evaluation.md#metrics). |
| Forecast-residual anomaly detection | Residual = actual − forecast. Flag a reading when it falls outside the forecast interval, e.g. [q0.01, q0.99]. The project's next phase. |
