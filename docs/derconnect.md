# DERCONNECT file workflow

This workflow prepares timestamped meter demand, runs frozen models, and produces
files for a dashboard. The offline fixture works without model weights or network.
It does not establish detection quality on real meters. The supplied raw files have
no anomaly annotations. Missing labels remain unknown.

The dashboard contract is [derconnect-output-contract.md](derconnect-output-contract.md).
`schemas/derconnect/v1/contract.json` defines types, keys and nullability. Existing
synthetic/UCR artifacts and their frozen selection are separate and unchanged.

## Offline example

Run in the existing compatible environment (`.venv/bin/` on PATH):

```sh
sfm stage-derconnect --source tests/fixtures/derconnect \
  --config configs/derconnect/tiny-pilot.yaml --out outputs/derconnect/fixture-stage
sfm detect-derconnect configs/derconnect/tiny-pilot.yaml \
  --model rolling_median --device cpu --out outputs/derconnect/fixture-run
sfm evaluate-derconnect outputs/derconnect/fixture-run \
  --annotations tests/fixtures/derconnect/annotations.json
sfm export-derconnect outputs/derconnect/fixture-run --format csv --hourly \
  --out outputs/derconnect/fixture-dashboard
sfm make-derconnect-injections configs/derconnect/tiny-pilot.yaml \
  --out outputs/derconnect/fixture-injections
sfm review-derconnect outputs/derconnect/fixture-run --out outputs/derconnect/fixture-review
```

Use a new output directory for each operation. Detection supports `--resume`: it
checks data, config, code, dependency and model identities, then recomputes incomplete
work deterministically. It verifies completed outputs before reuse. Unsupported
staging/export resume and overwrites fail. Completion is the atomic final manifest;
intermediate files alone are never a completed bundle. The scratch spool is local
working data; completed prediction partitions and checksums are authoritative.

`--meter E2451` narrows a configured detector to that meter (repeat for more).
UTC output boundaries are `roles.evaluation.start/end`; loading retains the earlier
calibration windows and prior context. Temporal roles are ordered and disjoint.
Use distinct configs/runs for development and final evaluation; the command never
selects a winning detector. `--dry-run` lists scope without importing model loaders.

## Preparation and scoring

Staging uses bounded CSV chunks, a SQLite disk sort across all source files and
one prepared UTC day at a time. Source hashes are streamed and checked again after
reading. Identical duplicates collapse; conflicting same-time readings are excluded
and retained with reasons in the raw staging payload. Invalid timestamps go to
`quarantine.jsonl`. Staged raw payloads retain every source column and timestamp
precision. Manifests report exclusions, numeric errors, channel ranges, constants,
unexpected meters/months, source coverage and prepared hashes. Constant channels
are descriptive, not automatically faulty. The raw source directory is never modified.
Expected months refer to filename calendar months, which can differ from UTC dates.
Schema differences and meter/filename disagreement fail staging.

One-minute left-closed means use finite A+B+C phase readings only. Staging preserves
minimum, maximum, observation count, capped coverage, maximum gap including bin edges,
and quality flags. Defaults require 80% of 30 observations and no gap above ten
seconds. Oversampling is reported. No interpolation or forward fill occurs. A missing
or unusable bin resets model context and breaks events. File/day boundaries do not.
Measured zero and negative demand remain valid. Unconfirmed source units are retained
as `source_unconfirmed`; no power-factor sign correction or W/kW conversion is inferred.

Rolling forecasts use only prior valid bins; a bin is available at its end. Context
256 is 4h16m at this resolution. PatchTST's actual truncation is recorded separately.
Chronos-2 and PatchTST use interval-exceedance scores. `seasonal_naive:1`,
`seasonal_naive:1440` and `rolling_median` use robust residual scores with a separately
fitted residual MAD scale. Their quantiles are degenerate point forecasts, not calibrated
probabilistic intervals. Daily naive requires at least 1440 context bins: its UTC
24-hour lag is not a local-calendar/DST model. Compare full detector configurations
at reviewed alert budgets; score magnitudes are not directly comparable.

Scale and threshold windows require approved reviewed-normal annotations by default.
The `exploratory.yaml` override explicitly records an unverified baseline and provides
no clean-data false-positive guarantee. Calibration is per meter, frozen before the
output range, and records support counts. The finite-sample threshold rule rejects
insufficient support, even in exploratory mode. Calibration arrays have an explicit
point cap; forecasting and point exports use daily partitions. No automatic trimming
of high calibration scores occurs. Minimum-run and merge durations must be exact
minute multiples. Entry qualification precedes gap merging; invalid bins always end
an event. V1 uses the existing event primitive's no-hysteresis behavior.

## Annotations and evaluation

Annotations are JSON row lists or CSV, with `meter_id`, `signal`, timezone-aware
`start`, `end`, `label` (anomaly/reviewed_normal/unknown), `label_source`, and
`review_status` (approved/pending). Optional `ground_truth_type` is annotation truth,
never a predicted class. Overlaps must be consolidated; conflicting overlaps fail.
Use separate sources for expert, artificial and injected truth. Config `data_kind`
separates artificial fixtures, real DERCONNECT and injected DERCONNECT reports.

An optional `review_block` identifies an explicitly fully reviewed contiguous block.
Every subinterval of a block must have known approved truth, and every bin must be
scored before event metrics are eligible. Boundary-crossing events are excluded and
counted. Matching maximizes one-to-one half-open overlap; delay uses detection
confirmation time in seconds, clipped at zero for an already-active event. Sparse
positive labels alone cannot establish recall. Point FPR uses only scored approved
normal coverage. Unknown periods cannot become TNs or FPs. Undefined metrics are
null with a reason; evaluated counts and seconds accompany rates. Alert volume and
scored fraction are descriptive when there are no labels. Calibration in-sample
rates live in `calibration.json`, separate from held-out evaluation.

`evaluate-derconnect` prints a new evaluation to stdout and preserves the completed
run. Redirect it to a separate file if needed. It does not rewrite original metrics.
Injection copies are minute-resolution test artifacts with separate scenario IDs,
seeds, splits, original hashes and ground-truth types. Backgrounds must be reviewed,
disjoint, outside calibration, and at most one day for this fixture tool. Supported
kinds: spike/dip/level_shift/variance_burst/stuck_sensor/trend_change. Dropout explicitly
removes observations and produces quality truth; it does not invent a zero-demand
anomaly. Never use scenario labels as model input or select on final test scenarios.

Review bundles contain minute actuals, forecasts, scores, thresholds, events and
quality context. They select alerts plus reservoir-sampled non-alert windows with a
fixed seed. `--compare-run OTHER` adds disagreement windows from aligned completed
runs. The CSV template starts pending/uncertain; mapping review categories to truth
requires approved explicit rules. No anomaly classifier is shipped.

## Load dashboard files without models

```python
from pathlib import Path
import json
import pandas as pd

root = Path('outputs/derconnect/fixture-dashboard')
manifest = json.loads((root / 'run-manifest.json').read_text())
assert manifest['complete']
# Also verify manifest output SHA-256 checksums before publishing/consuming a bundle.
# Load only the requested meter/day partitions in a real dashboard.
predictions = pd.read_parquet(root / manifest['prediction_parts'][0])
events = pd.read_parquet(root / 'events.parquet')
hourly = pd.read_parquet(root / 'hourly-summary.parquet')
metrics = json.loads((root / 'metrics.json').read_text())
joined = predictions.merge(events, on=['run_id', 'detector_id', 'meter_id', 'signal', 'event_id'],
                           how='left', suffixes=('', '_event'), validate='many_to_one')
```

CSV empty cells mean null, including integer flags. Invalid and warmup rows stay on
the grid. An unscored interval is unavailable, not normal. Event membership can include
merged gaps whose raw flag is zero. No minute predictions are upsampled into artificial
two-second observations. Hourly summaries are display aggregates; all-unscored hours
have null status, and averaged minute medians are not hourly quantile distributions.
Hourly power is a coverage-qualified mean, not kWh.

## Remote pilot gate

Before real execution, confirm source units/sign convention, cadence, baseline review
status and dashboard v1 compatibility. `remote-pilot.yaml` is an explicitly exploratory
one-meter, one-week proposal; its January baseline and later periods are calendar
roles, not evidence of cleanliness. Replace its dates and staged path after review.
Final locked configs refuse to run until calibration and selection records exist;
no winning detector or approved baseline has been fabricated. A selection record
contains `detector` (the complete detection mapping), resolved `model` identity and
`preparation_id`; a calibration record is a prior `calibration.json`. Locked runs
verify both, and recompute calibration to detect drift.

On an existing remote machine, install the locked environment on scratch separately.
Keep HF/uv caches on scratch and completed results on persistent storage. Stage the
source once using a config that lists all five expected meters, then point the pilot
config at that staging directory. Staging validates every discovered row, even when
detection later selects a single meter. Review the dry-run before loading weights:

```sh
sfm detect-derconnect configs/derconnect/remote-pilot.yaml \
  --model chronos2:/persistent/models/chronos2-pinned --meter E2451 --dry-run
DERCONNECT_CONFIG=configs/derconnect/remote-pilot.yaml \
DERCONNECT_MODEL=chronos2:/persistent/models/chronos2-pinned \
DERCONNECT_OUT=/persistent/results/derconnect-one-week \
DERCONNECT_SCRATCH=/scratch/derconnect \
  bash scripts/derconnect/remote-pilot.sh
```

After inspecting memory/runtime, gaps, scores, event volume and reviewed examples,
copy the config to list all five meters and adjust output dates for the approved
expansion. Reuse the same script with a new output path. The existing NRP launcher
was inspected: it submits Jobs and prepares legacy benchmark data, so this workflow
does not invoke it. No cluster submission, cloud resource creation or training is
part of implementation. Foundation model loader checks remain opt-in remote work.

## Verification record and limits

Before edits: 79 tests passed; lint had ten existing findings (config import/type
exception style and unused names in reproducibility tests). The lint fixes preserve
the public ValueError config-validation behavior. The artificial fixture covers
multiple meters, jitter, out-of-order rows, offsets, duplicates, invalid phases,
missing bins, real zero readings, partial labels and hour/day event boundaries.
Regenerate it with `python scripts/derconnect/make_fixture.py`.

Verification runs use offline fake forecasters and short rolling-median inference.
No private dataset scan, real-data model inference, model download, training or NRP
submission was performed. Unit tests cover causality, partition continuity, review
eligibility, score-zero thresholds, UTC microseconds, empty event schemas, output
checksums, model-shard identities and deterministic interrupted-run recomputation.
The CLI fixture workflow is included in the repository's offline CI test suite.
Final verification: **111 tests passed**, including 32 DERCONNECT cases;
`ruff check .` passed. The fixture CLI produced a completed rolling-median dashboard
bundle, injection scenarios and review windows under `outputs/derconnect/verified-*`.

V1 is an offline sustained-demand pilot, with no waveform transients, physical fault
classification, live infrastructure, automatic control, probabilistic baseline
intervals or real-data accuracy claims. Review/event metadata scale with the number
of events; raw staging and minute forecasting are disk-backed and partitioned.
Dashboard integration and real-data/model performance remain release gates.
