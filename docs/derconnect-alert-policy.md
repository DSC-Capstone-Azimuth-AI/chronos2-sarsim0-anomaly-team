# DERCONNECT schedule-policy overlays

`sfm apply-derconnect-policy` reads a completed, checksum-verified DERCONNECT bundle
and writes a separate `derconnect-alert-policy/v1` overlay. It loads no models and
changes no source scores, flags, event IDs, calibration or files. Distribute the
source dashboard and overlay together. These policies express reviewed operating
expectations, not predicted fault classes.

```sh
sfm apply-derconnect-policy PATH/TO/dashboard \
  --config configs/derconnect/policies/e2451-schedule-preview.yaml \
  --mode preview --out outputs/derconnect/policy-preview
```

Preview is the default. The committed E2451 windows (05:55–06:10 and 22:55–23:10
America/Los_Angeles) are pending development hypotheses, with no fabricated magnitude
limits or approval. Preview always keeps effective event counts unchanged. It reports
schedule overlaps separately from full containment, direction, magnitude, QA and
approval eligibility. `would_suppress=null` means insufficient evidence, not false.

Active mode requires complete approved rules and QA evidence. An event must fit
wholly within one schedule occurrence, be short enough, and have at least five valid
preceding minute actuals. Signed changes are `(actual - median_prior_actual) /
abs(median_prior_actual)`, for every event member including merged gaps. Direction is
measured from actual demand, not the model median. Reference values too close to zero,
mixed direction, excessive changes, gaps, ambiguous matches, missing evidence and
unresolved forecast QA keep alerts visible. Near the output horizon, merge-gap plus
entry-run lookahead or an explicit invalid-bin boundary is required for finalization.
An overnight rule belongs to its local starting weekday. Invalid DST occurrences
are skipped with a reason; no artificial timestamps are interpolated.

Policies are retrospective finalized-event decisions. `decision_data_cutoff_utc` is
the source data horizon, not a live notification time. Original `detected_at_utc`
remains unchanged; approvals have separate timestamps. Actual execution time appears
only in the manifest, outside deterministic decision identity.

## Evidence and validation

The configuration is closed: unknown or missing fields, invalid timezones, duplicates,
wrong signs, invalid clocks and non-minute durations fail. See the committed preview
YAML for the complete rule structure. Each evidence reference has `path` and `sha256`;
relative paths resolve against the policy YAML. Evidence paths do not determine the
policy ID; content hashes, settings and approval records do.

An approved reference JSON must contain:

- `kind: derconnect_schedule_reference`, `review_status: approved`;
- matching rule_id, meter_id, signal, unit and expected_direction;
- the exact approved minimum_absolute_reference, relative_change_lower/upper,
  preceding_context_minutes and max_duration_seconds;
- source_hashes identifying reviewed evidence;
- reviewer, reviewed_at and schedule_evidence matching the rule's review record.

Do not create these records without actual review. No example real-data approver is
provided. The reference unit must match the exported source values; no conversion
is guessed. Rule/evidence changes invalidate identities.

`qa_report` refers to a `derconnect-forecast-qa/v1` JSON and its digest. Its source
run, exact source manifest digest, and model digest must match. Detector status must
be `pass` before suppressing. `review_required`, `unverified`, unavailable QA or
nonfinite event forecasts keep alerts visible. Baseline point forecasts intentionally
have degenerate quantiles and use residual-scale calibration checks. Saved-output
inspection alone does not approve probabilistic model stability.

## Output files and joins

The machine contract is `schemas/derconnect-alert-policy/v1/contract.json`.

- `event-decisions.parquet` / `.csv`: all original event keys and policy decisions,
  reasons, candidate rule IDs, evidence hashes and signed-change diagnostics.
- `point-decisions.parquet/` with CSV copies: raw nullable flags and effective
  actionable flags/membership. Isolated raw flags remain in raw fields but are not
  retained actionable events. Merged-gap flags stay zero. Unscored flags stay null.
- `hourly-policy-summary.parquet` / `.csv`: raw, retained and actionable minutes;
  raw/suppressed/actionable event starts and event overlap. Starts count once; overlap
  appears in both hours. All-unscored hours retain null overlap status.
- `policy-report.json`: raw/effective volumes, evidence stages and kept reasons.
- optional `policy-evaluation.json`: raw vs entry-filtered vs policy-adjusted metrics.
- `source-snapshot.zip` and `policy-manifest.json`: implementation/config snapshot,
  source hashes, resolved config, execution metadata, deterministic IDs and checksums.

Completion is the last atomic policy manifest. Existing output paths are rejected;
partial output is never reused as complete. `verify_policy_bundle(overlay, source)`
checks both the original bundle and overlay checksums/identities. Copying the original
manifest and rewriting its predictions is not supported.

```python
from pathlib import Path
import pandas as pd
from sarsim_fm.anomaly.alert_policy import verify_policy_bundle

source = Path('PATH/TO/dashboard')
overlay = Path('PATH/TO/policy-preview')
manifest = verify_policy_bundle(overlay, source)
events = pd.read_parquet(source / 'events.parquet')
decisions = pd.read_parquet(overlay / 'event-decisions.parquet')
joined = events.merge(decisions, left_on=['run_id','detector_id','meter_id','signal','event_id'],
                      right_on=['source_run_id','detector_id','meter_id','signal','event_id'],
                      validate='one_to_one', suffixes=('', '_policy'))
# Active dashboard review list: joined.loc[~joined.suppressed]
# Preview: all events remain; show eligibility and missing-evidence reasons.
```

## Reviewed evaluation and sampling

`--annotations reviewed.json` adds before/after evaluation on exactly the same known
coverage. Suppressing a labelled anomaly creates a miss; truth and time denominators
never disappear. Entry filtering is evaluated separately, so isolated-flag removal is
not misattributed to schedule suppression. Unknown time never supplies negatives.

```sh
sfm review-derconnect PATH/TO/dashboard --isolated-flags 10 \
  --out outputs/derconnect/review-with-isolated
sfm normalize-derconnect-reviews --reviews consolidated-reviews.csv \
  --mapping approved-truth-mapping.json --out outputs/derconnect/reviewed.json
```

The first command uses all alerts, seeded non-alert reservoir sampling, plus seeded
isolated-flag sampling (`seed+1`). It does not automatically label anything.

The normalizer requires an approved mapping JSON with status, reviewer, reviewed_at,
label_source and categories for expected_operation, measurement_issue,
potential_operational_anomaly and uncertain. Pending rows stay unknown. Measurement
issues/uncertain must map to unknown demand truth. An anomaly additionally requires
`truth_confirmed=true` in the reviewed row; potential findings alone stay unknown.
Approved rows require a reviewer. Overlapping windows must be consolidated by the
reviewer first: conflicts are rejected, not resolved silently. Preserve unknown gaps;
review_block requires complete known approved coverage for event recall.

## Forecast diagnostics

```sh
python scripts/derconnect/diagnose_forecasts.py inspect \
  --bundle PATH/TO/dashboard --out outputs/derconnect/forecast-qa
python scripts/derconnect/diagnose_forecasts.py scope --bundle PATH/TO/dashboard
python scripts/derconnect/prepare_nrp_diagnostics.py \
  --bundle PATH/TO/dashboard --staged PATH/TO/staged \
  --out outputs/derconnect/ali-forecast-diagnostics \
  --remote-root /data/aliahmed/derconnect/ali-forecast-diagnostics
```

Inspection uses existing artifacts only. It writes QA JSON, selected cases/plot and
verified source identity. Original unrearranged quantiles are unavailable, so its
reproduction status stays `not_run`. Extreme outputs remain in the data; negative
power alone is not declared faulty. Quantitative findings are not silently clipped.

The preparation helper verifies source and staged hashes, packs current code and only
prepared data/source artifacts, and writes `diagnostic-job.yaml`, upload digests and
scope. It never calls kubectl or downloads weights. The future Job reproduces at most
40 target predictions (10 UTC times, 256/1440 context, batch 1/16), pins the original
revision and verifies config/weight digests before loading. It records actual dtype,
raw/exported quantiles, scores and batch differences. Tests use fake adapters only.

After separate authorization, upload the generated code.tar.gz, inputs.tar.gz and
upload-sha256.json through a CPU transfer pod into the declared remote root, create
the prepared diagnostic-job.yaml, and retrieve results after completion. The exact
pilot-specific commands are in the generated handoff README. No training or final
April–May evaluation is part of diagnostics.
