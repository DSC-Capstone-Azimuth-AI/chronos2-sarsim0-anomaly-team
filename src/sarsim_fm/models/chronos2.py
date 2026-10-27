"""Chronos-2: zero-shot forecasting and training on the SarSim0 stream."""

from __future__ import annotations

from pathlib import Path

import chronos.chronos2
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.trainer import Chronos2Trainer
from transformers import AutoConfig, TrainingArguments

from sarsim_fm import tracking
from sarsim_fm.data.sim import training_stream
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
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    model = build_model(cfg["model"], seed=data_cfg["seed"])

    max_steps = train_cfg["max_steps"]
    save_steps = train_cfg.get("save_steps", max_steps)
    args = TrainingArguments(
        output_dir=str(run_dir),
        max_steps=max_steps,
        # The stream yields complete batches; batch size lives in data.batch_size.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=train_cfg.get("grad_accum", 1),
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler", "linear"),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=train_cfg.get("logging_steps", 10),
        report_to="wandb" if tracking.active() else "none",
        save_strategy="steps" if save_steps < max_steps else "no",
        save_steps=save_steps,
        save_total_limit=1,
        save_only_model=True,
        remove_unused_columns=False,
        label_names=["future_target"],
        dataloader_num_workers=data_cfg.get("num_workers", 0),
        dataloader_pin_memory=runtime.is_cuda,
        use_cpu=runtime.device.type == "cpu",
        bf16=runtime.bf16,
        tf32=runtime.tf32,
        seed=data_cfg["seed"],
    )

    trainer = Chronos2Trainer(model=model, args=args, train_dataset=training_stream(data_cfg))
    trainer.train()

    final_dir = run_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(run_dir / "train-log.csv", index=False)
    return final_dir


def timed_steps(cfg: dict, runtime: Runtime, steps: int) -> dict:
    """Run a few raw optimizer steps with the config's batch size for timing and peak memory."""
    import time

    data_cfg = cfg["data"]
    model = build_model(cfg["model"], seed=data_cfg["seed"]).to(runtime.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])
    stream = iter(training_stream(data_cfg))
    if runtime.is_cuda:
        torch.cuda.reset_peak_memory_stats(runtime.device)

    losses, step_times, gen_times = [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        batch = next(stream)
        t1 = time.perf_counter()
        batch = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=runtime.bf16):
            loss = model(**batch).loss
        assert torch.isfinite(loss), f"non-finite loss {loss.item()}"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if runtime.is_cuda:
            torch.cuda.synchronize(runtime.device)
        gen_times.append(t1 - t0)
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    # Skip the first step (CUDA init, numba JIT) when there is more than one.
    timed = slice(1, None) if steps > 1 else slice(None)
    result = {
        "losses": [round(x, 4) for x in losses],
        "sec_per_step": float(np.mean(step_times[timed])),
        "sec_generating_per_step": float(np.mean(gen_times[timed])),
    }
    if runtime.is_cuda:
        result["peak_gpu_gb"] = torch.cuda.max_memory_allocated(runtime.device) / 1e9
    return result
