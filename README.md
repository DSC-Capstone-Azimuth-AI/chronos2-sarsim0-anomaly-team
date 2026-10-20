# Chronos-2 Training with SarSim0

This project connects the SarSim0 synthetic time-series generator to a Chronos-2 training pipeline and evaluates the resulting model on the M4 Monthly dataset.

### Paper and repository review

- Reviewed the SarSim0 paper and appendices to identify the datasets, simulator settings, training procedure, and evaluation approach.
- Cloned the SarSim0 repository into `external/sarsim0`.
- Chose Chronos-2 as the first model because an existing training framework was available.

### M4 Monthly preparation

- Downloaded the M4 Monthly data from the M4 repository.
- Confirmed that it contains 48,000 monthly time series.
- Confirmed that every series has an 18-step forecasting horizon.
- Staged the data in `data/m4/monthly.parquet` with separate training histories and test targets.
- Evaluated a Seasonal Naive baseline across all 48,000 series.
- Created 48,000 standardized 60-step windows for reproducing the paper's UMAP visualization.
- Verified that the staged data and visualization windows contain no invalid values.

The UMAP windows are visualization-only because they may include held-out test values. They are not used for training or model selection.

### Chronos-2 baseline

Evaluated the original `amazon/chronos-2` checkpoint on M4 Monthly using each series' training history as context and its official 18-step test portion as the target.

| Model | Mean sMAPE | Mean MASE |
|---|---:|---:|
| Original Chronos-2 | 0.1311438 | 0.9220801 |

The forecasts and per-series metrics were saved to:

```text
outputs/baselines/chronos2-monthly.parquet
```

### SarSim0 generator validation

Created `scripts/check_generator.py` and verified that SarSim0 produces finite tensors with the expected shapes.

The current generator configuration is:

```python
SarSimConfig(
    series_length=6000,
    burn_in=200,
    context_window=512,
    prediction_window=18,
    multivariate_prob=0.0,
    with_covariates_prob=0.0,
    pad_max=0,
)
```

For a batch size of 16, the generator produces:

```text
context: (16, 512)
target:  (16, 18)
```

The initial experiment is univariate and does not use covariates or variable-length padding.

### Training integration

Created `scripts/train_streaming.py` to connect the SarSim0 streaming generator to Chronos-2.

A 10-step integration run confirmed that:

- Synthetic batches reach Chronos-2 correctly.
- Forward and backward passes complete.
- Losses and gradients are finite.
- A trained checkpoint can be saved and reloaded.

The integration checkpoint loaded successfully through `Chronos2Pipeline`.

### 100-step smoke test

Completed a 100-step training run with:

```text
Model:             amazon/chronos-2
Context length:    512
Prediction length: 18
Batch size:        2
Learning rate:     1e-6
Training steps:    100
Seed:              42
```

The run completed all 100 optimizer steps with finite losses and gradients. The final checkpoint loaded successfully and was evaluated across all 48,000 M4 Monthly series.

| Model | Mean sMAPE | Mean MASE |
|---|---:|---:|
| Original Chronos-2 | 0.1311438 | 0.9220801 |
| 100-step SarSim0 checkpoint | 0.1315116 | 0.9243003 |

The smoke checkpoint is slightly worse than the original model. This run was intended to validate the complete implementation, not to produce a converged model.

The checkpoint evaluation was saved to:

```text
outputs/smoke-100/m4-monthly.parquet
```

## How the project works

```text
SarSim0 generates synthetic series
              |
              v
Each series is split into a 512-step context
and an 18-step future target
              |
              v
The streaming dataset packages the batch
for Chronos-2
              |
              v
Chronos-2 computes forecasting loss and
updates its parameters
              |
              v
The saved checkpoint is loaded for inference
              |
              v
M4 training histories are used as context and
forecasts are scored against held-out targets
```

### Synthetic data generation

SarSim0 generates new synthetic time series during training. The project does not create one large, fixed synthetic training file. This allows the model to see newly simulated examples as training proceeds.

### Streaming training

Each streaming dataset item is already a complete logical batch. The custom trainer passes these batches directly to Chronos-2 without adding another batch dimension.

The batch contains the context, future target, and the additional fields required by Chronos-2. The model computes its forecasting loss, backpropagates it, and updates its weights.

### M4 evaluation

M4 Monthly is used only for evaluation:

1. The official training portion of each series is provided as context.
2. Chronos-2 predicts the next 18 values.
3. Predictions are compared with the official held-out test values.
4. Per-series forecasts, sMAPE, and MASE are saved to Parquet.
5. Mean sMAPE and MASE are reported across all 48,000 series.

No M4 observations are used to update the model. This keeps the evaluation zero-shot.

## Reproducing the results

Run these commands from a GPU-enabled UCSD DSMLP session.

### 1. Clone the project

```bash
git clone \
  git@github.com:DSC-Capstone-Azimuth-AI/chronos2-sarsim0-anomaly-team.git

cd chronos2-sarsim0-anomaly-team
```

### 2. Create the Python environment

The completed experiments used Python 3.11. Create and activate an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-lock.txt

export HF_HOME="$PWD/.cache/huggingface"
```

Confirm that the environment and GPU are available:

```bash
which python

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA test passed:", y.shape)
PY
```

### 3. Install SarSim0

```bash
mkdir -p external

git clone \
  https://github.com/kashif/sarsim0.git \
  external/sarsim0

python -m pip install \
  -e ./external/sarsim0 \
  --no-deps
```

Verify the import:

```bash
python -c "import sarsim0; print(sarsim0.__file__)"
```

### 4. Download the M4 data

Use a sparse checkout so the project does not download the full M4 methods repository:

```bash
git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/Mcompetitions/M4-methods.git \
  external/M4-data

git -C external/M4-data sparse-checkout set Dataset
```

The required source files should now be available at:

```text
external/M4-data/Dataset/Train/Monthly-train.csv
external/M4-data/Dataset/Test/Monthly-test.csv
external/M4-data/Dataset/M4-info.csv
```

### 5. Stage and validate M4 Monthly

```bash
python scripts/stage_m4_monthly.py
python scripts/validate_m4_monthly.py
```

This creates `data/m4/monthly.parquet`. Validation should report 48,000 series, an 18-step target horizon, and no invalid values.

Run the Seasonal Naive baseline:

```bash
python scripts/evaluate_seasonal_naive.py
```

### 6. Evaluate the original Chronos-2 model

```bash
mkdir -p outputs/baselines

python scripts/evaluate_chronos2.py \
  --model amazon/chronos-2 \
  --output outputs/baselines/chronos2-monthly.parquet \
  --batch-size 64
```

Expected results are approximately:

```text
Mean sMAPE: 0.1311438
Mean MASE:  0.9220801
```

### 7. Validate the synthetic training path

```bash
python scripts/check_generator.py
python scripts/check_sarsim0_stream.py
python scripts/check_chronos2_training_step.py
```

These checks confirm that generated values are finite, batch shapes are correct, and Chronos-2 can complete a training step.

### 8. Run the 100-step smoke test

```bash
python -u scripts/train_streaming.py \
  --output-dir outputs/smoke-100 \
  --context-length 512 \
  --prediction-length 18 \
  --batch-size 2 \
  --max-steps 100 \
  --learning-rate 1e-6 \
  --save-steps 50 \
  --seed 42
```

The run should complete 100 optimizer steps with finite losses and gradients.

### 9. Evaluate the trained checkpoint

```bash
python scripts/evaluate_chronos2.py \
  --model outputs/smoke-100/checkpoint-100 \
  --output outputs/smoke-100/m4-monthly.parquet \
  --batch-size 16
```

Expected results are approximately:

```text
Mean sMAPE: 0.1315116
Mean MASE:  0.9243003
```

Small numerical differences may occur across GPU types and software builds. The main smoke-test success criteria are finite training values, completion of all 100 steps, successful checkpoint reload, and successful evaluation across all 48,000 series.

## Current project files

```text
chronos2-sarsim0-anomaly-team/
├── requirements-lock.txt
├── data/m4/monthly.parquet
├── external/sarsim0/
├── scripts/check_chronos2_training_step.py
├── scripts/check_generator.py
├── scripts/check_sarsim0_stream.py
├── scripts/evaluate_chronos2.py
├── scripts/evaluate_seasonal_naive.py
├── scripts/stage_m4_monthly.py
├── scripts/train_streaming.py
├── scripts/validate_m4_monthly.py
├── outputs/baselines/chronos2-monthly.parquet
└── outputs/smoke-100/
    ├── checkpoint-100/
    ├── m4-monthly.parquet
    └── run-config.json
```
