#!/usr/bin/env python3
"""Run reproducible QLoRA training while keeping all data local."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset as hf_load_dataset

from road_ocr.mlx_compat import patch_scalar_repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-2B-Instruct-4bit")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen3_vl_2b_lora.safetensors"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    patch_scalar_repeat()
    import mlx.core as mx
    import numpy as np

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    import mlx_vlm.lora as lora

    # MLX-VLM's CLI accepts Hub dataset names only. This narrow adapter keeps
    # competition images local and loads our JSONL using Hugging Face Datasets.
    original_loader = lora.load_dataset

    def load_local_dataset(path, config=None, split="train"):
        candidate = Path(path)
        if candidate.is_file():
            return hf_load_dataset("json", data_files={split: str(candidate)}, split=split)
        return original_loader(path, config, split=split)

    lora.load_dataset = load_local_dataset
    train_args = argparse.Namespace(
        model_path=args.model,
        full_finetune=False,
        train_vision=args.train_vision,
        dataset=str(args.data),
        split="train",
        dataset_config=None,
        image_resize_shape=None,
        custom_prompt_format=None,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        iters=args.iterations or 1000,
        epochs=None if args.iterations is not None else args.epochs,
        steps_per_report=25,
        steps_per_eval=500,
        steps_per_save=500,
        val_batches=0,
        max_seq_length=256,
        grad_checkpoint=args.grad_checkpoint,
        grad_clip=1.0,
        train_on_completions=True,
        gradient_accumulation_steps=args.gradient_accumulation,
        assistant_id=77091,
        lora_alpha=args.alpha,
        lora_rank=args.rank,
        lora_dropout=args.dropout,
        train_mode="sft",
        beta=0.1,
        eps=1e-8,
        output_path=str(args.output),
        adapter_path=str(args.resume) if args.resume else None,
    )
    lora.main(train_args)


if __name__ == "__main__":
    main()
