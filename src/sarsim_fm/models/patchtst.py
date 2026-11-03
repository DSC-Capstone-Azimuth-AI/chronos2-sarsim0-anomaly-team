"""PatchTST (Nie et al., ICLR 2023) from Hugging Face transformers, trained on the SarSim0 stream.

There is no general-purpose pretrained PatchTST (public checkpoints are fit to one dataset each),
so every checkpoint here comes from `sfm train`. The head predicts a Student-t per future step
and trains on its negative log-likelihood. Quantiles are read off that distribution exactly
rather than sampled, so the q0.01/q0.99 tails used for anomaly detection are deterministic.

PatchTST needs a fixed-length input. Shorter or gappy histories are left-filled and marked
unobserved in `past_observed_mask`, which keeps them out of the input scaling; training does the
same to a share of SarSim0 contexts (`data.short_context`) so the model has seen short series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PatchTSTConfig, PatchTSTForPrediction

from sarsim_fm.data.sim import training_stream
from sarsim_fm.models import hf
from sarsim_fm.runtime import Runtime

# PatchTST/64 from the paper (Table 3 setup for the larger datasets): 64 patches over 512 steps.
ARCHITECTURE = {
    "patch_length": 16,
    "patch_stride": 8,
    "d_model": 128,
    "num_attention_heads": 16,
    "num_hidden_layers": 3,
    "ffn_dim": 256,
    "dropout": 0.2,
    "head_dropout": 0.2,
}


def fill_unobserved(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Carry the last observed value forward; leading gaps take the first observed value.

    Filled values only feed the patch embeddings, never the scaler, so a flat fill is the
    least surprising thing for the model to see there. Rows with nothing observed become 0.
    """
    positions = torch.arange(values.shape[-1], device=values.device).expand_as(values)
    last = torch.where(observed, positions, -1).cummax(dim=-1).values
    first = torch.where(observed, positions, values.shape[-1]).min(dim=-1, keepdim=True).values
    index = torch.where(last >= 0, last, first).clamp(max=values.shape[-1] - 1)
    filled = values.gather(-1, index)
    return torch.where(observed.any(dim=-1, keepdim=True), filled, torch.zeros_like(filled))


def model_inputs(contexts: list[np.ndarray], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-align each history in a `length` window; NaNs and left padding are unobserved."""
    values = torch.full((len(contexts), length), float("nan"))
    for i, history in enumerate(contexts):
        tail = torch.as_tensor(np.asarray(history, dtype=np.float32)[-length:])
        values[i, length - len(tail):] = tail
    observed = torch.isfinite(values)
    return fill_unobserved(torch.nan_to_num(values), observed), observed


class PatchTSTStream(IterableDataset):
    """The SarSim0 training stream in PatchTST's input format, with random history truncation."""

    def __init__(self, data_cfg: dict):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = data_cfg["batch_size"]

    def __iter__(self):
        worker = get_worker_info()
        short = {"prob": 0.0, "min_length": 16, **self.data_cfg.get("short_context", {})}
        generator = torch.Generator().manual_seed(
            self.data_cfg["seed"] + (worker.id if worker else 0) + 1_000_003
        )
        for batch in training_stream(self.data_cfg):
            context = batch["context"]
            n, length = context.shape
            keep = torch.randint(short["min_length"], length + 1, (n, 1), generator=generator)
            truncate = torch.rand(n, 1, generator=generator) < short["prob"]
            observed = torch.arange(length) >= torch.where(truncate, length - keep, 0)
            observed &= torch.isfinite(context)
            yield {
                "past_values": fill_unobserved(torch.nan_to_num(context), observed).unsqueeze(-1),
                "past_observed_mask": observed.unsqueeze(-1),
                "future_values": batch["future_target"].unsqueeze(-1),
            }


class PatchTSTForecaster:
    def __init__(self, model: PatchTSTForPrediction, device: torch.device, context_length: int):
        self.model = model
        self.device = device
        # Histories are cut to the shorter of what was asked for and what the model was trained on.
        self.context_length = min(context_length, model.config.context_length)

    @torch.no_grad()
    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        if horizon > self.model.config.prediction_length:
            raise ValueError(
                f"horizon {horizon} exceeds the trained prediction_length {self.model.config.prediction_length}"
            )
        model_length = self.model.config.context_length
        out = []
        for start in range(0, len(contexts), batch_size):
            batch = [np.asarray(c)[-self.context_length:] for c in contexts[start:start + batch_size]]
            values, observed = model_inputs(batch, model_length)
            pred = self.model(
                past_values=values.unsqueeze(-1).to(self.device),
                past_observed_mask=observed.unsqueeze(-1).to(self.device),
            )
            out.append(student_t_quantiles(pred, levels)[:, :horizon])
        return np.concatenate(out).astype(np.float32)


def student_t_quantiles(pred, levels) -> np.ndarray:
    """(batch, prediction_length, n_levels) quantiles of PatchTST's scaled Student-t head."""
    df, mu, sigma = (p.squeeze(-1).double().cpu().numpy() for p in pred.prediction_outputs)
    loc, scale = (x.squeeze(-1).double().cpu().numpy() for x in (pred.loc, pred.scale))
    z = student_t.ppf(np.asarray(levels)[None, None, :], df[..., None])
    return loc[..., None] + scale[..., None] * (mu[..., None] + sigma[..., None] * z)


def load(path: str | None, runtime: Runtime, context_length: int) -> PatchTSTForecaster:
    if not path:
        raise ValueError("patchtst needs a trained checkpoint, e.g. patchtst:outputs/<run>/checkpoint-final")
    model = PatchTSTForPrediction.from_pretrained(path).to(runtime.device).eval()
    return PatchTSTForecaster(model, runtime.device, context_length)


def build_model(cfg: dict) -> PatchTSTForPrediction:
    """`init: scratch` builds `model.architecture` with random weights; `init: pretrained` loads `base`."""
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    init = model_cfg.get("init", "scratch")
    if init == "pretrained":
        return PatchTSTForPrediction.from_pretrained(model_cfg["base"])
    if init != "scratch":
        raise ValueError(f"model.init must be 'pretrained' or 'scratch', got {init!r}")
    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=data_cfg["context_length"],
        prediction_length=data_cfg["prediction_length"],
        scaling="std",
        loss="nll",
        distribution_output="student_t",
        **{**ARCHITECTURE, **model_cfg.get("architecture", {})},
    )
    torch.manual_seed(data_cfg["seed"])
    return PatchTSTForPrediction(config)


def train(cfg: dict, runtime: Runtime, run_dir: Path) -> Path:
    model = build_model(cfg)
    print(f"PatchTST parameters: {sum(p.numel() for p in model.parameters()):,}")
    args = hf.training_arguments(cfg, runtime, run_dir, label_names=["future_values"])

    trainer = hf.StreamTrainer(model=model, args=args, train_dataset=PatchTSTStream(cfg["data"]))
    trainer.train()

    final_dir = run_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(run_dir / "train-log.csv", index=False)
    return final_dir


def timed_steps(cfg: dict, runtime: Runtime, steps: int) -> dict:
    """Run a few raw optimizer steps with the config's batch size for timing and peak memory."""
    stream = PatchTSTStream(cfg["data"])
    return hf.timed_steps(build_model(cfg), stream, cfg["train"]["learning_rate"], runtime, steps)
