import torch

from sarsim0 import SarSimConfig, SarSim0StreamingDataset


config = SarSimConfig(
    series_length=6000,
    burn_in=200,
    context_window=512,
    prediction_window=18,
    multivariate_prob=0.0,
    with_covariates_prob=0.0,
    pad_max=0,
)

dataset = SarSim0StreamingDataset(
    context_length=512,
    prediction_length=18,
    batch_size=2,
    config=config,
    seed=42,
)

batch = next(iter(dataset))

for name, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(
            name,
            "shape=",
            tuple(value.shape),
            "dtype=",
            value.dtype,
        )
    else:
        print(name, value)

required = {
    "context",
    "future_target",
    "future_covariates",
    "group_ids",
    "num_output_patches",
}

assert required.issubset(batch)
assert batch["context"].shape[-1] == 512
assert batch["future_target"].shape[-1] >= 18
assert torch.isfinite(batch["context"]).all()

# Only the first 18 future positions must contain targets.
assert torch.isfinite(batch["future_target"][..., :18]).all()

print("Streaming batch passed")