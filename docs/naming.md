# Naming runs and configs

One name follows a run everywhere: the config's `name:` is the `outputs/<name>/` folder and the
W&B run name. Pick it so the dashboard alone tells runs apart.

## Run names

```text
<model>-<init>-<steps>[-<what's different>]
```

| Part | Values | Notes |
|---|---|---|
| `model` | `chronos2`, later `chronos1`, `stfm`, ... | `model.type` |
| `init` | `ft` (from pretrained weights), `scratch` (random init) | `model.init` |
| `steps` | `500`, `1k`, `5k`, `20k` | Optimizer steps (`train.max_steps`) |
| suffix | Only what differs from the config the run came from | See tokens below; join several with `-` |

Suffix tokens:

| Token | Meaning | Example |
|---|---|---|
| `b<N>` | Effective batch (`data.batch_size × train.grad_accum`) when it isn't 256 | `chronos2-ft-1k-b8` |
| `lr<X>` | Learning rate | `chronos2-ft-500-lr3e-5` |
| `s<lo>-<hi>` | SarSim0 seasonal range `data.sarsim.s_range` | `chronos2-ft-500-s0-24` |
| `warm<N>` | Warmup steps | `chronos2-ft-500-warm25` |
| `seed<N>` | Data seed, for repeats of the same setup | `chronos2-ft-500-seed7` |
| `sweep-<what>` | Sweep runs; the W&B run id is appended automatically | `chronos2-ft-500-sweep-lr-x8k2p1` |

Rules:

- Lowercase, hyphens only.
- **No device or GPU in the name.** The same config runs anywhere with `--device`, and the
  device is already a W&B tag (`cuda`, `cpu`) and recorded in `run-config.json`.
- Name what makes the run *different*, not everything. The full config is in W&B and
  `run-config.json`.
- Set the name when overriding settings:
  `sfm train configs/train/chronos2-ft-5k.yaml --set train.max_steps=500 --set train.learning_rate=3e-5 --set name=chronos2-ft-500-lr3e-5`

## Other run kinds

| Kind | Name | Set by |
|---|---|---|
| Baselines | `baseline-<model spec>`, e.g. `baseline-seasonal-naive`, `baseline-chronos2-amazon-chronos-2` | Automatic (`sfm eval --wandb`, `sfm wandb-import`) |
| Plumbing checks | `dev-<anything>`, W&B group `dev-test` | You; delete when done |

## Config files

`configs/train/<model>-<init>-<steps>.yaml`, with `name:` equal to the file name, e.g.
`chronos2-ft-5k.yaml` → `name: chronos2-ft-5k`. Exceptions named for their purpose:

| File | Purpose | `name:` |
|---|---|---|
| `_base-<model>.yaml` | Shared defaults, never run directly | none |
| `<model>-ft-smoke.yaml` | 100-step plumbing check; default for `sfm check` | `chronos2-ft-smoke` |
| `<model>-ft-cpu.yaml` | Small fine-tune sized for a laptop CPU | `chronos2-ft-1k-b8` |

Sweeps: `configs/sweeps/<model>-<init>-<steps>-<what is searched>.yaml`, e.g. `chronos2-ft-500-lr.yaml`.
Search one question per sweep.

## W&B groups

Name a group after the question it answers: `lr-search`, `scratch-vs-ft`,
`sarsim-season-ablation`. Fixed groups: `baselines`, `imported`, `dev-test`.
See [wandb.md](wandb.md#how-runs-are-organized).

## Runs from before this convention

These `outputs/` folders keep their original names (results, checkpoints and `run-config.json`
point at them); only the W&B display name was changed. The W&B run id is unchanged.

| `outputs/` folder | W&B name | W&B run id |
|---|---|---|
| `ft-500` | `chronos2-ft-500` | `import-ft-500` |
| `chronos2-ft-gpu-3070ti` | `chronos2-ft-5k` | `import-chronos2-ft-gpu-3070ti` |
| `chronos2-ft-cpu` | `chronos2-ft-1k-b8` | `import-chronos2-ft-cpu` |
| `chronos2-scratch-gpu` | `chronos2-scratch-20k` | `import-chronos2-scratch-gpu` |
| `baselines/seasonal_naive` | `baseline-seasonal-naive` | `import-baselines-seasonal_naive` |
| `baselines/chronos2_amazon_chronos-2` | `baseline-chronos2-amazon-chronos-2` | `import-baselines-chronos2_amazon_chronos-2` |

Old config names: `chronos2-ft-gpu.yaml` → `chronos2-ft-5k.yaml`,
`chronos2-scratch-gpu.yaml` → `chronos2-scratch-20k.yaml`.
