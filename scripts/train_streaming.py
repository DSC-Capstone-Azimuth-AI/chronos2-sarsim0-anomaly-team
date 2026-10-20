from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from chronos import Chronos2Pipeline
from chronos.chronos2.trainer import Chronos2Trainer
from sarsim0 import SarSimConfig, SarSim0StreamingDataset


class StreamingChronos2Trainer(Chronos2Trainer):
    def get_train_dataloader(self):
        # Each dataset item is already one complete logical batch.
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = SarSimConfig(
        series_length=6000,
        burn_in=200,
        context_window=args.context_length,
        prediction_window=args.prediction_length,
        multivariate_prob=0.0,
        with_covariates_prob=0.0,
        pad_max=0,
    )

    dataset = SarSim0StreamingDataset(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        config=config,
        seed=args.seed,
    )

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map="cuda",
    )

    model = pipeline.model

    use_bf16 = torch.cuda.is_bf16_supported()
    capability = torch.cuda.get_device_capability()
    use_tf32 = capability[0] >= 8

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,

        # The iterable dataset already produces complete batches.
        per_device_train_batch_size=1,

        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="adamw_torch",

        logging_strategy="steps",
        logging_steps=1,
        report_to="none",

        save_strategy=(
            "steps"
            if args.save_steps <= args.max_steps
            else "no"
        ),
        save_steps=args.save_steps,

        remove_unused_columns=False,
        dataloader_num_workers=0,

        # GTX 1080 Ti should use both as False.
        bf16=use_bf16,
        tf32=use_tf32,
        fp16=False,

        label_names=["future_target"],
        save_total_limit=1,
        save_only_model=True,
    )

    run_config = {
        "base_model": args.model,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "logical_batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "multivariate_prob": 0.0,
        "with_covariates_prob": 0.0,
        "pad_max": 0,
        "bf16": use_bf16,
        "tf32": use_tf32,
        "gpu": torch.cuda.get_device_name(0),
    }

    (args.output_dir / "run-config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    trainer = StreamingChronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    final_dir = args.output_dir / "checkpoint-final"
    trainer.save_model(str(final_dir))
    trainer.save_state()

    print(f"Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()