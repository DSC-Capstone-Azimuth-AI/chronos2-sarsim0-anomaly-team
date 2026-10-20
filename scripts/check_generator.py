from sarsim0 import SarSim0Generator, SarSimConfig
import torch

config = SarSimConfig(
    series_length=6000,
    burn_in=200,
    context_window=512,
    prediction_window=18,
    multivariate_prob=0.0,
    with_covariates_prob=0.0,
    pad_max=0,
)

generator = SarSim0Generator(config=config, seed=42)
context, target = generator.generate_batch(batch_size=16)

assert context.shape == (16, 512)
assert target.shape == (16, 18)
assert torch.isfinite(context).all()
assert torch.isfinite(target).all()

print(context.shape, target.shape)
print("Generator check passed")