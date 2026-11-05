"""Chronos-2: zero-shot forecasting and training on the SarSim0 stream."""

from __future__ import annotations

from pathlib import Path

import chronos.chronos2
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.trainer import Chronos2Trainer
from transformers import AutoConfig

from sarsim_fm.data.sim import training_stream
from sarsim_fm.models import hf
from sarsim_fm.runtime import Runtime


class Chronos2Forecaster:
    def __init__(self, pipeline: Chronos2Pipeline, context_length: int):
        self.pipeline = pipeline
        self.context_length = context_length

    def predict_quantiles(self, contexts, horizon, levels, batch_size, season_lengths):
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs=contexts,
            prediction_length=horizon,
            quantile_levels=list(levels),
            context_length=self.context_length,
            batch_size=batch_size,
        )
        # Each element is (n_variates=1, horizon, n_levels).
        return torch.cat(quantiles).float().cpu().numpy()


def load(path: str | None, runtime: Runtime, context_length: int) -> Chronos2Forecaster:
    pipeline = Chronos2Pipeline.from_pretrained(path or "amazon/chronos-2", device_map=str(runtime.device))
    pipeline.model.eval()
    return Chronos2Forecaster(pipeline, context_length)


def build_model(model_cfg: dict, seed: int) -> torch.nn.Module:
    """`init: pretrained` loads `base` weights; `init: scratch` uses its architecture with random weights."""
    base = model_cfg.get("base", "amazon/chronos-2")
    init = model_cfg.get("init", "pretrained")
    if init == "pretrained":
        return Chronos2Pipeline.from_pretrained(base, device_map="cpu").model
    if init == "scratch":
        config = AutoConfig.from_pretrained(base)
        torch.manual_seed(seed)
        return getattr(chronos.chronos2, config.architectures[0])(config)
    raise ValueError(f"model.init must be 'pretrained' or 'scratch', got {init!r}")


def train(cfg: dict, runtime: Runtime, run_dir: Path) -> Path:
    data_cfg = cfg["data"]
    model = build_model(cfg["model"], seed=data_cfg["seed"])
    args = hf.training_arguments(cfg, runtime, run_dir, label_names=["future_target"])

    trainer = Chronos2Trainer(model=model, args=args, train_dataset=training_stream(data_cfg))
    trainer.train()

    final_dir = run_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(run_dir / "train-log.csv", index=False)
    return final_dir


def timed_steps(cfg: dict, runtime: Runtime, steps: int) -> dict:
    """Run a few raw optimizer steps with the config's batch size for timing and peak memory."""
    data_cfg = cfg["data"]
    model = build_model(cfg["model"], seed=data_cfg["seed"])
    return hf.timed_steps(model, training_stream(data_cfg), cfg["train"]["learning_rate"], runtime, steps)
