"""Hugging Face Trainer settings shared by every trainable model, and the `sfm check` step timer."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from sarsim_fm import tracking
from sarsim_fm.runtime import Runtime


def training_arguments(cfg: dict, runtime: Runtime, run_dir: Path, label_names: list[str]) -> TrainingArguments:
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    max_steps = train_cfg["max_steps"]
    save_steps = train_cfg.get("save_steps", max_steps)
    return TrainingArguments(
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
        label_names=label_names,
        dataloader_num_workers=data_cfg.get("num_workers", 0),
        dataloader_pin_memory=runtime.is_cuda,
        use_cpu=runtime.device.type == "cpu",
        bf16=runtime.bf16,
        tf32=runtime.tf32,
        seed=data_cfg["seed"],
    )


class StreamTrainer(Trainer):
    """Trainer for an IterableDataset that already yields whole batches (no collation)."""

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )


def timed_steps(model: torch.nn.Module, stream, learning_rate: float, runtime: Runtime, steps: int) -> dict:
    """Run a few raw optimizer steps for timing and peak memory."""
    model = model.to(runtime.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    stream = iter(stream)
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
