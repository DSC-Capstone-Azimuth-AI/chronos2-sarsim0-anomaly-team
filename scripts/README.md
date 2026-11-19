# Scripts

Helpers that sit outside the `sfm` CLI. Each script's header has its full usage. Shell scripts need
bash: on Windows use Git Bash, not WSL.

## NRP cluster

See [docs/nrp.md](../docs/nrp.md).

| Script | Does | Run from |
|---|---|---|
| [`nrp-run.sh`](nrp-run.sh) | Launch any `sfm` command as a GPU Job on the pushed commit | Your machine |
| [`nrp-transfer.sh`](nrp-transfer.sh) | Copy files to/from the shared `/data` volume, or list it, through a CPU-only pod | Your machine |

## DERConnect via `sfm detect` (hourly and 2-second)

The lighter DERConnect path: stage per-building series, run the generic detector, export flat
files for the dashboard. Configs: `configs/anomaly/derconnect-*.yaml`.

| Script | Does | Run from |
|---|---|---|
| [`stage_derconnect.py`](stage_derconnect.py) | Raw DERConnect zip → per-building total-power series (`--freq 1h`, `2s`, ...) in `data/anomaly/` | Your machine |
| [`nrp-derconnect-2s.sh`](nrp-derconnect-2s.sh) | One NRP Job per building: 2-second `sfm detect` for one model, then dashboard export | Your machine |
| [`export_dashboard.py`](export_dashboard.py) | One `sfm detect` run → predictions, forecasts and events tables for the dashboard | Anywhere |
| [`derconnect/forecast_quality.py`](derconnect/forecast_quality.py) | E1: compare forecast quality of several models' `sfm detect` runs on the same meters | Anywhere |

## DERConnect via `sfm *-derconnect`

Helpers for the minute-resolution workflow in [docs/derconnect.md](../docs/derconnect.md) and
[docs/derconnect-alert-policy.md](../docs/derconnect-alert-policy.md).

| Script | Does | Run from |
|---|---|---|
| [`derconnect/make_fixture.py`](derconnect/make_fixture.py) | Regenerate the artificial test CSVs in `tests/fixtures/derconnect/` | Your machine |
| [`derconnect/diagnose_forecasts.py`](derconnect/diagnose_forecasts.py) | Inspect a dashboard bundle's saved forecasts; optionally reproduce a small pinned case set | Anywhere |
| [`derconnect/prepare_nrp_diagnostics.py`](derconnect/prepare_nrp_diagnostics.py) | Package a forecast-reproduction Job for NRP (does not submit it) | Your machine |
| [`derconnect/prepare_nrp_pilot.py`](derconnect/prepare_nrp_pilot.py) | Package an inference-only pilot for NRP (does not submit it) | Your machine |
| [`derconnect/remote-pilot.sh`](derconnect/remote-pilot.sh) | Dry-run, detect and export one pilot config on an existing remote machine | Remote machine |

## Synthetic and UCR anomaly benchmark

| Script | Does | Run from |
|---|---|---|
| [`run-anomaly-remote.sh`](run-anomaly-remote.sh) | Evaluate the packaged anomaly experiment (from `sfm prepare-anomaly-handoff`) on a CUDA machine | Remote machine |
