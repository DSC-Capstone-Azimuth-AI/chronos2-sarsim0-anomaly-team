"""PatchTST-FM r2 (IBM Granite): a pretrained zero-shot PatchTST, fine-tuned or trained from scratch on SarSim0.

A ~385M foundation model pretrained on real and synthetic corpora, so it plays the role `amazon/chronos-2`
plays for Chronos-2: `init: pretrained` fine-tunes it, `init: scratch` trains the same architecture
from random weights. The model code is vendored (`models/vendor/patchtst_fm`, see its README).

PatchTST-FM is a masked forecaster. It sees one window of `config.context_length` steps holding
left padding, the history, and at least `forecast_slots(config)` masked future steps, and predicts
99 quantiles (0.01..0.99) at every masked step. Inference uses IBM's own `forward`. Training builds
the same window with `forecast_window`, puts the true SarSim0 future in the masked slots, and applies
IBM's pretraining loss (pinball over the 99 quantiles) to the slots that have a target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from transformers.utils import ModelOutput

from sarsim_fm.data.sim import training_stream
from sarsim_fm.models import hf
from sarsim_fm.models.vendor.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPrediction
from sarsim_fm.runtime import Runtime

BASE = "ibm-granite/granite-timeseries-patchtst-fm-r2"


def forecast_slots(config: PatchTSTFMConfig, horizon: int = 0) -> int:
    """Masked future steps per window: IBM's inference always masks at least two pretraining spans."""
    mask_cont = config.pretrain_mask_cont * (2 if config.randomize_cpm else 1)
    return max(horizon, config.d_patch * max(mask_cont, 2))


def forecast_window(
    config: PatchTSTFMConfig, context: torch.Tensor, future: torch.Tensor | None = None
) -> dict[str, torch.Tensor | int]:
    """Lay out (batch, length) histories the way `PatchTSTFMForPrediction.forward` does.

    NaNs in `context` are missing values. `future` (batch, horizon), when given, fills the first
    masked slots so a loss can be taken there; slots without a target are marked missing.
    """
    horizon = 0 if future is None else future.shape[1]
    slots = forecast_slots(config, horizon)
    context = context[:, -(config.context_length - slots):].float()
    batch, length = context.shape

    mean = torch.nan_to_num(context.nanmean(dim=1, keepdim=True))
    observed = torch.isfinite(context)
    target = torch.full((batch, slots), float("nan"), device=context.device)
    if future is not None:
        target[:, :horizon] = future.float()
    has_target = torch.isfinite(target)

    left = config.context_length - length - slots
    no, yes = torch.zeros(batch, left, dtype=torch.bool), torch.ones(batch, left, dtype=torch.bool)
    no, yes = no.to(context.device), yes.to(context.device)
    return {
        "inputs": torch.cat([mean.expand(batch, left), torch.where(observed, context, mean),
                             torch.where(has_target, target, mean)], dim=1),
        "pred_mask": torch.cat([no, torch.zeros_like(observed), torch.ones_like(has_target)], dim=1),
        "miss_mask": torch.cat([no, ~observed, ~has_target], dim=1),
        "pad_mask": torch.cat([yes, torch.zeros_like(observed), torch.zeros_like(has_target)], dim=1),
        "context_length": length + slots,
    }


@dataclass
class TrainingOutput(ModelOutput):
    loss: torch.Tensor | None = None


class PatchTSTFMForSarSim0(PatchTSTFMForPrediction):
    """IBM's prediction model plus a forecasting loss when `future_values` is given.

    The weights are unchanged (`backbone.*`), so checkpoints load as `PatchTSTFMForPrediction`.
    """

    def forward(self, past_values, future_values=None, past_observed_mask=None, pad_mask=None,
                prediction_length=None, quantile_levels=None, return_dict=None):
        if future_values is None:
            return super().forward(past_values, past_observed_mask=past_observed_mask, pad_mask=pad_mask,
                                   prediction_length=prediction_length, quantile_levels=quantile_levels,
                                   return_dict=return_dict)
        return TrainingOutput(loss=forecast_loss(self.backbone, past_values, future_values))


def forecast_loss(backbone, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """IBM's pretraining loss (`PatchTSTFMForPretraining`), on the future slots of a forecast window."""
    config = backbone.config
    out = backbone(**forecast_window(config, context, future), return_loss=True)
    target, q = out.normed_target.unsqueeze(-1), out.quantile_outputs
    levels = torch.tensor(backbone.quantile_levels, device=q.device).view(1, 1, -1)
    pinball = 2 * torch.abs((target - q) * ((target <= q).float() - levels))
    weight = out.loss_mask
    if config.patch_loss_windowing is not None:
        weight = (weight.reshape(-1, config.n_patch, config.d_patch) * backbone.patch_loss_window).reshape_as(weight)
    loss = (pinball * weight.unsqueeze(-1)).sum(dim=1) / weight.sum(dim=1, keepdim=True).clamp(min=1)
    return loss.sum(dim=-1).mean() / math.sqrt(config.num_quantile)


class PatchTSTFMStream(IterableDataset):
    """The SarSim0 training stream with PatchTST-FM's argument names."""

    def __init__(self, data_cfg: dict):
        super().__init__()
        self.data_cfg = data_cfg

    def __iter__(self):
        for batch in training_stream(self.data_cfg):
            yield {"past_values": batch["context"], "future_values": batch["future_target"]}


class PatchTSTFMForecaster:
    def __init__(self, model: PatchTSTFMForPrediction, device: torch.device, context_length: int):
        self.model = model
        self.device = device
        self.context_length = context_length

    @torch.no_grad()
    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        config = self.model.config
        length = min(self.context_length, config.context_length - forecast_slots(config, horizon))
        # The eval levels are all on the model's 0.01 grid (exact); others are interpolated linearly.
        pos = np.interp(levels, config.quantile_levels, np.arange(len(config.quantile_levels)))
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(config.quantile_levels) - 1)
        w = (pos - lo)[None, :, None]
        out = []
        for start in range(0, len(contexts), batch_size):
            batch = [np.asarray(c, dtype=np.float32)[-length:] for c in contexts[start:start + batch_size]]
            # Right-align histories of different lengths; the left fill is padding, not missing data.
            width = max(len(c) for c in batch)
            values = torch.full((len(batch), width), float("nan"))
            pad = torch.ones(len(batch), width, dtype=torch.bool)
            for i, c in enumerate(batch):
                values[i, width - len(c):] = torch.from_numpy(c)
                pad[i, width - len(c):] = False
            # (batch, time, 1): IBM's forward adds the channel axis to values but not to pad_mask.
            pred = self.model(
                past_values=values.unsqueeze(-1).to(self.device),
                past_observed_mask=torch.isfinite(values).unsqueeze(-1).to(self.device),
                pad_mask=pad.unsqueeze(-1).to(self.device),
                prediction_length=horizon,
                return_dict=True,
            )
            q = pred.quantile_outputs[..., 0].float().cpu().numpy()  # (batch, 99, horizon)
            out.append((q[:, lo] * (1 - w) + q[:, hi] * w).transpose(0, 2, 1))
        return np.concatenate(out).astype(np.float32)


def load(path: str | None, runtime: Runtime, context_length: int) -> PatchTSTFMForecaster:
    model = PatchTSTFMForPrediction.from_pretrained(path or BASE).to(runtime.device).eval()
    return PatchTSTFMForecaster(model, runtime.device, context_length)


def build_model(model_cfg: dict, seed: int) -> PatchTSTFMForSarSim0:
    """`init: pretrained` loads `base` weights; `init: scratch` uses its architecture with random weights.

    `model.architecture` overrides config fields for a scratch model (e.g. fewer layers).
    """
    base = model_cfg.get("base") or BASE
    init = model_cfg.get("init", "pretrained")
    if init == "pretrained":
        if model_cfg.get("architecture"):
            raise ValueError("model.architecture only applies to init: scratch")
        return PatchTSTFMForSarSim0.from_pretrained(base)
    if init != "scratch":
        raise ValueError(f"model.init must be 'pretrained' or 'scratch', got {init!r}")
    # n_patch and quantile_levels are derived in __init__, so rebuild the config rather than patch it.
    fields = PatchTSTFMConfig.from_pretrained(base).to_dict()
    for derived in ("n_patch", "quantile_levels", "architectures", "transformers_version", "_name_or_path"):
        fields.pop(derived, None)
    config = PatchTSTFMConfig(**{**fields, **model_cfg.get("architecture", {})})
    torch.manual_seed(seed)
    return PatchTSTFMForSarSim0(config)


def train(cfg: dict, runtime: Runtime, run_dir: Path) -> Path:
    data_cfg = cfg["data"]
    model = build_model(cfg["model"], seed=data_cfg["seed"])
    print(f"PatchTST-FM parameters: {sum(p.numel() for p in model.parameters()):,}")
    args = hf.training_arguments(cfg, runtime, run_dir, label_names=["future_values"])

    trainer = hf.StreamTrainer(model=model, args=args, train_dataset=PatchTSTFMStream(data_cfg))
    trainer.train()

    final_dir = run_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(run_dir / "train-log.csv", index=False)
    return final_dir


def timed_steps(cfg: dict, runtime: Runtime, steps: int) -> dict:
    """Run a few raw optimizer steps with the config's batch size for timing and peak memory."""
    data_cfg = cfg["data"]
    model = build_model(cfg["model"], seed=data_cfg["seed"])
    return hf.timed_steps(model, PatchTSTFMStream(data_cfg), cfg["train"]["learning_rate"], runtime, steps)
