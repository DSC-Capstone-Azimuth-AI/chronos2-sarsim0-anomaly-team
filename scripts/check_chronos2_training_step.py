import torch

from chronos import Chronos2Pipeline
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

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
)

model = pipeline.model
model.train()

device = next(model.parameters()).device
batch = next(iter(dataset))

batch = {
    name: value.to(device)
    if isinstance(value, torch.Tensor)
    else value
    for name, value in batch.items()
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-6,
)

optimizer.zero_grad(set_to_none=True)

output = model(**batch)

if isinstance(output, torch.Tensor):
    loss = output
elif hasattr(output, "loss"):
    loss = output.loss
elif isinstance(output, dict) and "loss" in output:
    loss = output["loss"]
else:
    raise TypeError(f"Unexpected model output: {type(output)}")

assert loss.ndim == 0
assert torch.isfinite(loss)

print("Loss before update:", loss.item())

loss.backward()
optimizer.step()
torch.cuda.synchronize()

print(
    "Peak GPU memory:",
    torch.cuda.max_memory_allocated() / 1e9,
    "GB",
)
print("Forward/backward test passed")