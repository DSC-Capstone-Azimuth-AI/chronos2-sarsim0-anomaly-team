# DERCONNECT output contract v1

The machine-readable column/type/nullability/key definitions are in
`schemas/derconnect/v1/contract.json`. Parquet is canonical. All timestamps use
`timestamp[us, UTC]`; CSV uses ISO 8601 with six fractional digits and `Z`, with
empty cells for null. Intervals are half-open. Prediction keys are run, detector,
meter, signal and UTC bin start. Empty event files retain their complete schema.

`actual` is one-minute mean total source power; units remain `source_unconfirmed`
until confirmed. Negative and measured-zero demand remain valid values. `q50` and
`expected_power` are the median forecast. Other q columns are minute quantiles;
seasonal naive and rolling median use degenerate point quantiles (see manifest).
`available_at_utc` is bin end. `forecast_origin_utc` is the latest context bin end,
equal to the target bin start. `horizon_steps=1`.

Raw `anomaly_flag` is nullable integer 0/1: null means unscored, never normal.
`prediction_status` is scored, warmup, insufficient_coverage, invalid_input, or
uncalibrated. Quality flags are a JSON string array. `event_member` includes merged
gaps; isolated rejected flags can have no event ID. Scores and thresholds share raw
units and are not probabilities or comparable between detector configurations.
No anomaly classes are predicted: unknown / not_classified are fixed values.

Events use raw score peaks, earliest peak ties, and bin-boundary start/end.
`detected_at_utc` is the end of the first qualifying consecutive run, even after
merging. IDs depend on content and configuration. Unusable bins always break events.
Quality incidents have their own file and reasons, without fabricated scores.

`predictions.parquet/` is a partitioned dataset; `prediction_parts` in the completed
run manifest lists files. Consumers must verify `complete` and checksums before use.
`events.parquet`, `quality-events.parquet`, `metrics.json`, and `hourly-summary.parquet`
complete the bundle. Hourly summaries count events *starting* in the hour and mark
retained overlap in both hours. All-unscored hours have null `has_anomaly`. Hourly
power and matching-bin median-forecast means require 80% of a full hour; they are
means, not hourly forecast distributions. No hourly quantiles or energy are inferred.

Unknown labels never count as negatives. Evaluation records reviewed coverage and
reasons for undefined metrics, separating calibration from held-out results.
Dashboard v1 compatibility still needs confirmation by the dashboard team.
