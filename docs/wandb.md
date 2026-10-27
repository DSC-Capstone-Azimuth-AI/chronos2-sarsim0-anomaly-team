# Experiment tracking with Weights & Biases

W&B gives the team one shared, filterable table of every run (any machine, anyone) with live
loss curves. It is **optional**: everything still writes to `outputs/<run>/` as before, and those
local files stay the source of truth. W&B is the shared logbook on top.

- Project: **`sarsim-fm`** under entity **`chewycrunch-ucsd`** → https://wandb.ai/chewycrunch-ucsd/sarsim-fm
- Code: [`src/sarsim_fm/tracking.py`](../src/sarsim_fm/tracking.py) (every function is a no-op unless a run is open)

## Contents

- [One-time setup](#one-time-setup)
- [Cheat sheet](#cheat-sheet)
- [How runs are organized](#how-runs-are-organized)
- [Logging a training run](#logging-a-training-run)
- [Logging baselines](#logging-baselines)
- [Backfilling old runs](#backfilling-old-runs)
- [Sweeps](#sweeps)
- [Using the W&B UI](#using-the-wb-ui)
- [Offline machines and environment variables](#offline-machines-and-environment-variables)
- [Troubleshooting](#troubleshooting)

## One-time setup

```bash
uv sync --extra cu130 --extra wandb      # GPU machines / DSMLP
uv sync --extra cpu --extra wandb        # laptops
uv run wandb login                       # paste the key from https://wandb.ai/authorize
```

Include **every** extra you use in a single `uv sync` (e.g. `--extra cu130 --extra wandb --extra dev`).
`uv sync` removes extras you leave out, so syncing without `--extra wandb` uninstalls W&B, and
syncing with `--extra cpu` on a GPU machine swaps in the CPU build of torch.

To log to the shared project you need to be a member of the `chewycrunch-ucsd` entity; ask whoever
owns it for an invite. Until then, log to your own account with `WANDB_ENTITY=<your-username>`.

## Cheat sheet

| Task | Command |
|---|---|
| Train and log | `uv run sfm train <config> --set wandb.enabled=true --set wandb.group=<question>` |
| Quick test that logging works | `uv run sfm train configs/train/chronos2-ft-smoke.yaml train.max_steps=20 --set name=dev-wandb-test --set wandb.enabled=true --set wandb.group=dev-test --set eval.datasets=[sim_iid]` |
| Log baselines | `uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2 --wandb` |
| Upload finished runs from `outputs/` | `uv run sfm wandb-import` (or `--only <run-folder>`) |
| Start a sweep | `uv run wandb sweep configs/sweeps/<file>.yaml` → prints `<sweep-id>` |
| Join a sweep (any machine) | `uv run wandb agent chewycrunch-ucsd/sarsim-fm/<sweep-id>` |
| Join for N runs only | `uv run wandb agent --count 3 chewycrunch-ucsd/sarsim-fm/<sweep-id>` |
| Pause / resume / stop a sweep | `uv run wandb sweep --pause` / `--resume` / `--stop chewycrunch-ucsd/sarsim-fm/<sweep-id>` |
| Log without internet | `WANDB_MODE=offline uv run sfm train ...`, later `uv run wandb sync wandb/offline-run-*` |
| Turn logging off for one command | `WANDB_MODE=disabled uv run sfm train ...` |
| Check login | `uv run wandb status` |

## How runs are organized

Everything goes into **one project**. Don't make a project per model or per experiment: the point
is to compare across all of them in one table. A run is described by:

| W&B field | What we put there | Use it to |
|---|---|---|
| **config** | The full resolved YAML (after `inherit:` and `--set`) | Filter or group by any setting: `model.type`, `model.init`, `data.sarsim.s_range`, `train.learning_rate` |
| **group** | `wandb.group`: the *question* the run answers | Keep related runs together, e.g. `lr-search`, `sarsim-season-ablation` |
| **job_type** | `train` or `baseline` | Show baselines as reference rows or hide them |
| **tags** | Added automatically: `model.type`, `model.init`, device. Add more with `wandb.tags` | Quick filters (`scratch`, `cuda`, `imported`) |
| **name** | The config's `name` (= the `outputs/<name>/` folder) | Find the local files for a run |
| **summary** | `eval/<set>/<metric>` and `timing/*` | Sort and compare runs |
| **charts** | `train/loss`, `train/learning_rate`, `train/grad_norm` per step | Watch training live |

Summary keys, per eval set (`sim_iid`, `sim_ood_season`, `m4_monthly`):

```text
eval/<set>/smape  eval/<set>/mase  eval/<set>/sql  eval/<set>/wql
eval/<set>/coverage_80  eval/<set>/coverage_98  eval/<set>/n_series  eval/<set>/predict_seconds
timing/train_seconds  timing/eval_seconds  timing/total_seconds  timing/finished
```

The three things we tune map to config namespaces, so they compare across models:

- **SarSim0 training data**: `data.sarsim.*`
- **Model**: `model.type` (+ model-specific keys under `model.*`)
- **Training recipe**: `train.*`

### Names and groups

Run names follow [naming.md](naming.md): `<model>-<init>-<steps>[-<what's different>]`, e.g.
`chronos2-ft-500-lr3e-5`, with no device in the name. Name a group after the question, not the
date or the person: `lr-search`, `scratch-vs-ft`, `sarsim-season-ablation`. Use `dev-test` for
plumbing checks you will delete. Imported runs land in `imported`, baselines in `baselines`.

## Logging a training run

Logging is off by default (`wandb.enabled: false` in `configs/train/_base-chronos2.yaml`), so CPU
smoke tests never need a login. Turn it on per run:

```bash
uv run sfm train configs/train/chronos2-ft-5k.yaml \
  --set wandb.enabled=true --set wandb.group=lr-search --set wandb.tags=[first-pass]
```

or permanently in a config file:

```yaml
wandb:
  enabled: true
  group: scratch-vs-ft
  tags: [overnight]
```

The run opens before training. The Hugging Face Trainer streams loss, learning rate and grad
norm every `train.logging_steps`, then the eval metrics and timing are written to the summary.
If the run crashes, it is marked as failed in W&B.

## Logging baselines

```bash
uv run sfm eval --model seasonal_naive --model chronos2:amazon/chronos-2 --wandb
```

Each `--model` becomes its own run with `job_type=baseline`, group `baselines`. Only needed again
when the eval sets change.

## Backfilling old runs

Runs finished before W&B was set up (or with it off) can be uploaded from their folders:

```bash
uv run sfm wandb-import                      # every run under outputs/ with eval/summary.json
uv run sfm wandb-import --only ft-500        # one run folder (outputs/ft-500)
uv run sfm wandb-import --group my-group     # group for imported train runs (default: imported)
```

Each folder becomes one run: config (plus git SHA, GPU, package versions) from `run-config.json`,
the loss curve replayed from `train-log.csv`, metrics from `eval/summary.json`. Run ids are
`import-<folder>`, so running it again skips what is already uploaded. Runs that were logged
live are skipped too.

Caveats: W&B shows the *upload* time as the run date (the real finish time is in
`timing/finished`); a run id you delete in W&B may not be reusable for a while.

## Sweeps

A sweep lets W&B pick settings to try, and any number of machines pull runs from it.
Use them for **cheap, short runs** (fine-tune LR, short-budget SarSim0 comparisons). Long runs
(the 7 h scratch run, anything on DSMLP) stay as files in `configs/train/`.

### Folders, for orientation

| Folder | Read by | Describes |
|---|---|---|
| `configs/data/` | `sfm make-sim` | The fixed held-out test sets. Rarely changes. |
| `configs/train/` | `sfm train` | One training run. |
| `configs/sweeps/` | `wandb sweep` | A search: which `train/` config to start from and which fields to vary. |

### Writing a sweep file

Start from [`configs/sweeps/chronos2-ft-lr.yaml`](../configs/sweeps/chronos2-ft-lr.yaml):

```yaml
program: src/sarsim_fm/cli.py
project: sarsim-fm
entity: chewycrunch-ucsd
name: chronos2-ft-lr
method: random              # random | grid | bayes
run_cap: 12                 # stop after this many runs
metric:
  name: eval/sim_ood_season/wql
  goal: minimize
parameters:                 # any config field, as a dotted path
  train.learning_rate:
    distribution: log_uniform_values
    min: 1.0e-6
    max: 1.0e-4
  data.sarsim.s_range:
    values: [[0, 24], [0, 52]]
command:
  - uv
  - run
  - sfm
  - train
  - configs/train/chronos2-ft-5k.yaml
  - ${args_no_hyphens}       # must come right after the config path
  - --set
  - name=chronos2-ft-1k-sweep-lr
  - --set
  - train.max_steps=1000
  - --set
  - eval.datasets=[sim_iid,sim_ood_season]
```

How it works: the agent turns each sample into
`sfm train configs/train/chronos2-ft-5k.yaml train.learning_rate=3.2e-05 data.sarsim.s_range=[0, 24] --set ...`.
Bare `key=value` arguments after the config path work like `--set`, but **only directly after
the config path**, which is why `${args_no_hyphens}` comes before the `--set` lines. Each sweep
run's folder gets the W&B run id appended (`outputs/chronos2-ft-1k-sweep-lr-<id>/`) so runs don't overwrite
each other. Logging is turned on automatically under an agent.

### Rules for sweeps

- **Never optimize on M4.** M4 is the real-data zero-shot test; tuning on it leaks the test into
  model selection. Use `eval/sim_iid/*` or `eval/sim_ood_season/*` as the sweep metric, and
  leave `m4_monthly` out of `eval.datasets` to save time. Evaluate the winner on M4 afterwards
  with a normal `sfm train` run.
- Keep runs short (`train.max_steps`) so the sweep finishes. `run_cap` bounds the total.
- `random` is a good default; `bayes` needs more runs to pay off; `grid` only for small,
  discrete spaces.

### Running it

```bash
uv run wandb sweep configs/sweeps/chronos2-ft-lr.yaml      # once; prints the sweep id
uv run wandb agent chewycrunch-ucsd/sarsim-fm/<sweep-id>   # on every machine that should help
```

Run agents from the repo root (the command uses relative paths). With several GPUs on one
machine, start one agent per GPU: `CUDA_VISIBLE_DEVICES=0 uv run wandb agent ...`,
`CUDA_VISIBLE_DEVICES=1 uv run wandb agent ...`. Pause, resume or stop from the sweep page or
with `wandb sweep --pause|--resume|--stop <entity>/<project>/<sweep-id>`.

## Using the W&B UI

- **Runs table**: open the project → *Runs*. Click *Columns* to add summary keys such as
  `eval/m4_monthly/wql` and config keys such as `train.learning_rate`, then sort.
- **Filter**: e.g. `job_type = train`, `group = lr-search`, `tags contains scratch`,
  `model.init = scratch`.
- **Group by**: group by `model.init` or `data.sarsim.s_range` to average over seeds.
- **Compare curves**: select runs in the sidebar; charts overlay `train/loss` on a shared
  `train/global_step` axis (imported runs use the same axis).
- **Sweep page**: the parallel-coordinates and parameter-importance panels show which settings
  moved the metric.
- **Reports**: for the Experiments log or a writeup, make a Report from the project page and
  embed the runs table and charts; it stays live as runs are added.

## Offline machines and environment variables

| Variable | Effect |
|---|---|
| `WANDB_MODE=offline` | Log locally to `wandb/`; upload later with `uv run wandb sync wandb/offline-run-*` |
| `WANDB_MODE=disabled` | Turn W&B off for this command even if the config enables it |
| `WANDB_ENTITY` / `WANDB_PROJECT` | Log somewhere other than `chewycrunch-ucsd/sarsim-fm` |
| `WANDB_DIR` | Where local W&B files go (default `./wandb/`, git-ignored) |
| `WANDB_SILENT=true` | Hide W&B console output |
| `WANDB_API_KEY` | Log in without `wandb login` (e.g. in a DSMLP job script) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'wandb'` | `uv sync` with `--extra wandb` (and your other extras) |
| `wandb.init` asks for an API key or fails in a batch job | `uv run wandb login` once, or set `WANDB_API_KEY` |
| Permission / entity not found error | You're not in `chewycrunch-ucsd` yet: get an invite, or use `WANDB_ENTITY=<your-username>` |
| `unrecognized arguments: key=value` | Bare `key=value` overrides must come right after the config path; otherwise use `--set` |
| Run shows no `eval/*` keys | It was run with `--no-eval` or `eval.datasets=[]`, or crashed before eval |
| `wandb-import` says `skipped (exists ...)` | Already uploaded. Delete it in W&B first to re-import |
| Torch lost CUDA after a sync | You synced with `--extra cpu`; rerun `uv sync --extra cu130 --extra wandb` |
