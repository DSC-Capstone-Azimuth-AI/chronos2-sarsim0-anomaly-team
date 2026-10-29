# Anomaly detection

Use forecast errors to find unusual readings, then group them into events. Everything runs
through `sfm`, with configs for the data and detector. For the forecasting eval sets and metrics,
see [Evaluation](evaluation.md).

## How it works

1. **Forecast**: use earlier readings to predict the next value, including its likely range.
2. **Score**: compare the actual reading with the forecast. Larger scores mean more unusual values.
3. **Flag**: choose a threshold using clean data, then flag scores above it.
4. **Group**: join nearby flags into events and remove runs that are too short.
5. **Compare**: measure how many labelled anomalies were found and how often normal points
   were flagged.

The same code runs rolling median/MAD, seasonal naive and Chronos-2. This is evaluation with
existing weights; it does not train the model. Keep model training and the full Chronos UCR run
on the remote machine.

## Running it

```bash
uv sync --extra cpu --extra dev
uv run pytest -q
uv run ruff check src tests

# 1. Fixed simulated data: development, test and normal-only test sets
uv run sfm make-anomaly-sim configs/data/sim-anomaly-dev.yaml
uv run sfm make-anomaly-sim configs/data/sim-anomaly-test.yaml

# 2. Compare the baselines and pretrained Chronos on development data
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model rolling_mad
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model seasonal_naive
uv run sfm detect configs/anomaly/synthetic-chronos2.yaml --model chronos2:amazon/chronos-2

# 3. Pick the score, threshold and event settings using development results only
uv run sfm select-anomaly outputs/anomaly/synthetic-dev-chronos2_amazon_chronos-2

# 4. Evaluate the chosen settings on the untouched test sets
uv run sfm detect configs/anomaly/synthetic-locked.yaml --model rolling_mad
uv run sfm detect configs/anomaly/synthetic-locked.yaml --model seasonal_naive
uv run sfm detect configs/anomaly/synthetic-locked.yaml --model chronos2:amazon/chronos-2

# 5. Every saved anomaly result so far (JSON)
uv run sfm anomaly-results
```

`--device cpu|cuda|cuda:N` chooses the device. `--out` changes the output folder,
`--set key=value` overrides a config field, and `--overwrite` replaces a completed run.

To run the whole sequence, including UCR, use `uv run sfm anomaly-experiment --device cuda`.
The simulated sets and UCR manifest must already exist. Add `--synthetic-only` to stop after
the simulated tests. Matching completed runs are reused. Progress goes to
`outputs/anomaly/experiment-state.json`; comparisons go to `outputs/anomaly/comparison.json`.

## Simulated data

Each set has 30 series of 2,048 points:

- **`sim-anomaly-dev`**: choose detector settings here. Seeds 3,100,000–3,100,029.
- **`sim-anomaly-test`**: final test with anomalies. Seeds 4,100,000–4,100,029.
- **`sim-normal-test`**: no injected anomalies; measures false alarms. Seeds 5,100,000–5,100,029.

Each series has its own simulator seed. Rebuilding with the same config gives the same data.
Files go to `data/anomaly/<set>.parquet`, with a `.manifest.json` recording the config, seeds,
file hash and rejected-series counts. These are our experiment settings, not settings specified
by the SarSim0 paper.

The first 1,024 points are clean. Anomalies are added only after that: spikes, dips, level shifts,
variance bursts, trend changes, seasonal disruptions, dropouts, stuck sensors, reversed segments
and time warps. Each kind appears three times per anomaly set. Durations are measured in the
configured seasonal period; changes in amplitude use the nearby clean data's median absolute
deviation (MAD) as their scale. A dropout, reversal or stuck sensor can leave an already zero or
constant segment unchanged; its label still records the attempted change.

Series with non-finite values, values reaching ±1e6, or an all-zero starting context are
regenerated. Generation stops with an error after 30 failed attempts for a series. Normal-only
rows have no anomaly start or end. All detector labels use zero-based `[start, end)` indices:
the start is included and the end is not.

## Forecasts and scores

**Rolling** mode predicts one point at a time using only earlier readings. This is the reference
mode. **Block** mode predicts 18 points at a time, then moves forward 18 points. Each saved
prediction keeps its horizon (1–18); the last block is shortened if needed. Both modes use the
observed history, including any earlier anomalous readings.

Contexts are built one batch at a time to limit memory use. Chronos's pinned
`cross_learning=False` default keeps predictions independent across contexts in the batch.
To compare the modes on three fixed development series:

```bash
uv run sfm anomaly-pilot configs/anomaly/synthetic-pilot.yaml \
  --model chronos2:amazon/chronos-2 --device cpu
```

Runtime and score differences go to `outputs/anomaly-pilot/comparison.json`.
For development block runs, pass `--set detector.mode=block` to `sfm detect`.

Quantiles are sorted before scoring so they do not cross. The three scores are:

- **`interval_exceedance`**: distance outside q0.01–q0.99, divided by the interval width.
  Zero inside the interval.
- **`robust_residual`**: absolute error of the median forecast, divided by the clean residual
  scale (1.4826 × MAD).
- **`quantile_tail`**: estimates how far into either tail the reading falls, then takes the
  negative log of its two-sided tail probability. Between quantiles it interpolates; beyond
  q0.01/q0.99 it extends the tail exponentially using the nearest distinct quantile spacing.
  That extension is an assumption, not a guaranteed probability estimate.

Scales and interval widths have a minimum of 1e-8 to handle constant data.

## Choosing thresholds and events

Thresholds come from clean forecast scores (*calibration data*). For simulated data, the first
quarter of the clean prefix is history, the second quarter fits the residual scale, and the
final half fits thresholds. Threshold scores are pooled within each simulated set. Points after
the clean prefix are used only for evaluation.

The target false-positive rate (FPR) is 0.005, 0.01 or 0.02. With `n` clean scores and target
`alpha`, the threshold is the `ceil((n+1)*(1-alpha))`-th smallest score. A point is flagged only
when its score is strictly above that threshold. The three rates need at least 199, 99 and 49
calibration points respectively, **for each forecast horizon**. Block mode fits a separate
threshold at each horizon and fails if there is not enough data. Correlated readings and changes
in the series can make the actual false-alarm rate differ from the target, so both are reported.

`select-anomaly` tries the three scores and FPRs, minimum runs of 1/2/3, merge gaps of 0/2/6,
and release ratios of 1/0.75: 162 combinations. It picks the highest development event F1 whose
normal-point FPR is no more than twice the target. Ties favour fewer false alarms, then simpler
events. The test sets and UCR results do not choose these settings.

The choice and every candidate's result go to `outputs/anomaly/synthetic-selection.json`.
The locked configs require this file and check that its settings match. An existing selection
file cannot be overwritten; use a new path for a new experiment.

Event settings:

- **`min_run`**: discard runs with fewer than this many consecutive flags.
- **`release_threshold_ratio`**: after a qualifying run starts, keep the event open while its
  score stays above this fraction of the entry threshold (*hysteresis*). A ratio of 1 uses
  the flag mask alone.
- **`merge_gap`**: join events separated by no more than this many unflagged points.

That is also the order they are applied. Scores are divided by their horizon's threshold for
hysteresis and event peaks, so the entry threshold is 1. Peak ties use the earliest point.

## UCR data

Stage the downloaded UCR/HexagonML archive, then check a small fixed subset before the full run:

```bash
uv run sfm stage-ucr-anomaly --source /path/to/UCR_Anomaly_FullData

# Small integration check, after choosing settings on simulated data
uv run sfm detect configs/anomaly/ucr-chronos2.yaml --model seasonal_naive \
  --out outputs/anomaly/ucr-smoke-naive --set 'subset_ids=[4,5,173]'

# Full baselines
uv run sfm detect configs/anomaly/ucr-chronos2.yaml --model rolling_mad
uv run sfm detect configs/anomaly/ucr-chronos2.yaml --model seasonal_naive

# Full Chronos runs: run these on the remote GPU
uv run sfm detect configs/anomaly/ucr-chronos2.yaml --model chronos2:amazon/chronos-2 --device cuda
# Only if the fine-tuned checkpoint is available:
uv run sfm detect configs/anomaly/ucr-chronos2.yaml \
  --model chronos2:outputs/chronos2-ft-500/checkpoint-final --device cuda
```

Staging checks all 250 files: unique IDs 1–250, finite values and valid train/anomaly indices.
Newlines, spaces and tabs are accepted. `data/anomaly/ucr.manifest.json` records file paths,
sizes, hashes, series lengths, clean-prefix lengths, annotations and mnemonic families. Original
data stays outside Git, and file hashes are checked before evaluation.

The filename's anomaly indices are **one-based and inclusive**. They stay that way in the
manifest and are converted once to zero-based `[start-1, end)` for the detector.

Each file is independent. The first 80% of its declared clean prefix is history: forecasts over
the 60–80% section fit the residual scale. The final 20% fits thresholds. Evaluation starts
after the clean prefix. No later point or other UCR file is used to fit that file's detector.
UCR does not declare a seasonal period, so seasonal naive uses period 1 (the last observed value).

Related variants are grouped by mnemonic family: remove `DISTORTED`/`NOISE` at the start and
replicate numbers at the end. This is a naming rule, not a domain label. Paired model comparisons
bootstrap whole families, keeping related files together when drawing samples.

## Metrics

**Event precision, recall and F1.** A detection is correct if it overlaps a labelled anomaly.
Only one detection can match each label; extra detections count as false positives. Detections
are matched in time order to the earliest-ending unmatched label. Delay is the first overlapping
point minus the label's start. There is no point-adjusted F1.

**False alarms and coverage.** Count raw flags per 1,000 normal points, including flags later
removed by event grouping. The realized FPR is the same count divided by normal points.
Cov 80 and Cov 98 measure how often normal readings fall inside the predicted intervals.
These use points after calibration, excluding labelled anomalies.

**AUPRC.** A secondary pointwise score: average precision, with tied scores grouped together.
The summary averages it across series with anomalies; normal-only series are excluded.
Event counts are summed before calculating precision/recall/F1. Coverage and delay are averaged
over series with valid values.

**UCR localization.** Pick the center of the event with the highest normalized peak. If there
is no event, pick the highest raw-score point. The archive's rule, from slide 4 of
`UCR_AnomalyDataSets.pptx`, uses the original indices: with `L=end-begin+1`, a prediction `P` is
correct when `min(begin-L, begin-100) < P < max(end+L, end+100)`. Both comparisons are strict.
Source: [UCR/HexagonML archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

Localization and false alarms are reported separately. UCR's clean-prefix false-alarm metric
uses the final 20% calibration section, so it is **in-sample**. The normal-point metric after
the prefix is the held-out check.

## Saved files

By default, `sfm detect` writes `outputs/anomaly/<name>-<model>/`:

```text
run-config.json          full config, environment, code/checkpoint hashes, seeds and timing
source-snapshot.zip      source, configs and lockfile, including uncommitted changes
dataset-manifest.json    datasets used and their hashes
forecasts/<item>.parquet  time, actuals, 13 quantiles and forecast horizon
forecasts/<item>.json     model, context, mode, batch size and file hashes
scores/<item>.parquet     all three scores, data-split masks, flags and labels
events.parquet           detected intervals, peak times and normalized peak scores
per-series.json          thresholds, calibration counts, scales and metrics for each series
summary.json             overall, per-dataset and family results
```

Forecasts are also cached in `outputs/anomaly-cache/`. They are reused only when the model,
forecast code, context, mode, batch size, levels, period and data match, and the cache file's hash
is correct. Each run keeps its own copy. A writer lock prevents two processes writing the same run.

Failure plots are for completed locked runs. Install matplotlib through the notebooks extra:

```bash
uv sync --extra cpu --extra dev --extra notebooks
uv run sfm anomaly-diagnostics outputs/anomaly/ucr-full-2 --max-plots 20
```

## Checks run locally

Local work covers tests, simulated data, forecasts with existing weights, both 250-file UCR
baselines, plots and the transfer package. The record is `outputs/anomaly/local-completion.json`.
Full Chronos UCR evaluation, the fine-tuned checkpoint comparison and full M4 checks run remotely.

M4 data and `outputs/chronos2-ft-500/checkpoint-final` were not available in the initial checkout.
The original eight tests passed, but the published M4 results could not be rerun. This is recorded
in `outputs/anomaly-baseline/baseline.json`. A small forecasting check used 20 series per sim set:

```bash
uv run sfm make-sim --set defaults.n_series=20
uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2
```

Those pilot scores are not the full published results. For M4, stage the data as described in
the [README](../README.md). Missing fine-tuned weights are reported as unavailable.

## Running remotely

After the local runs finish, package the code, data, chosen settings and results:

```bash
uv run sfm prepare-anomaly-handoff
```

This reruns the code checks and checks the remote script's shell syntax. It writes
`outputs/anomaly-handoff/anomaly-handoff.tar.gz` and `handoff.json` with its SHA-256 checksum.
The package includes all 250 UCR files under `external/ucr-anomaly/`, which stays outside Git.
It leaves out installed packages, credentials, model weights and the large UCR forecast caches.
Use `--without-ucr` to transfer UCR separately; set `SARSIM_UCR_SOURCE` to its remote folder.

Copy the package to a remote CUDA machine, compare its checksum with `handoff.json`, and unpack
it into a new directory:

```bash
tar -xzf anomaly-handoff.tar.gz
cd anomaly-handoff
bash scripts/run-anomaly-remote.sh
# Optional existing fine-tuned checkpoint:
bash scripts/run-anomaly-remote.sh --checkpoint /remote/path/chronos2-ft-500/checkpoint-final
```

Requires `uv` and a driver supporting CUDA 13. The script installs the pinned GPU packages,
verifies the copied files, checks the GPU, and runs the tests. It downloads the same pretrained
checkpoint version used locally and checks its weight hash. Then it stages UCR with the remote
paths, runs the small subset and full evaluations, and saves plots and comparisons.
It does not train the model or call `sfm check` (which runs timed training steps).

Useful options for `sfm anomaly-experiment`:

- **`--external-only`**: use the saved settings and run UCR without repeating development selection.
- **`--pretrained-checkpoint PATH`**: use a local copy of a fixed pretrained checkpoint version.
- **`--checkpoint PATH`**: also evaluate an existing fine-tuned checkpoint.

Completed runs can resume after moving the checkout: data hashes must match, but old absolute
paths need not exist. Restage UCR on the new machine. The package keeps local UCR summaries and
plots under `reports/local-ucr/`; complete raw UCR outputs are generated again remotely under
`outputs/anomaly/ucr-full-*`. New outputs and the new UCR manifest do not break package verification.

## When to train again

Train again only if the locked results show that forecast errors or poor interval coverage are
the main problem. Check scores, thresholds, context and event grouping first.

If training is needed, run it remotely. Apply the same bad-series filters to training data,
evaluate checkpoints every 100–250 steps, try a learning rate below 1e-5, and use at least three
seeds. Choose checkpoints by simulated development detection and normal false alarms, then check
that M4 forecasting performance is preserved. Keep training from random weights on synthetic data
separate from fine-tuning a model pretrained on real data.
