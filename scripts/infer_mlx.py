#!/usr/bin/env python3
"""Deterministic MLX-VLM inference with resumable JSONL checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from road_ocr.metrics import clean_prediction, competition_metrics
from road_ocr.mlx_compat import patch_scalar_repeat


DEFAULT_PROMPT = (
    "Transcribe this handwritten line exactly, preserving original spelling and capitalization. "
    "Output only the transcription."
)


def read_checkpoint(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    if not path.is_file():
        return predictions
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            predictions[record["ID"]] = record["Target"]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/validation.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/images_1024"))
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-2B-Instruct-4bit")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation_predictions.csv"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()

    patch_scalar_repeat()
    from mlx_vlm import apply_chat_template, generate, load

    frame = pd.read_csv(args.input_csv, encoding="utf-8-sig", keep_default_na=False)
    if "ID" not in frame or frame.ID.duplicated().any():
        raise ValueError("Input CSV must contain unique IDs")
    if args.limit:
        frame = frame.iloc[: args.limit].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".jsonl")
    predictions = read_checkpoint(checkpoint)
    adapter_path = str(args.adapter) if args.adapter else None
    model, processor = load(args.model, adapter_path=adapter_path)
    formatted_prompt = apply_chat_template(
        processor,
        model.config,
        args.prompt,
        num_images=1,
    )

    started = time.time()
    with checkpoint.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            if row.ID not in predictions:
                image_path = args.image_dir / f"{row.ID}.jpg"
                result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    image=str(image_path),
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                prediction = clean_prediction(result.text)
                predictions[row.ID] = prediction
                handle.write(json.dumps({"ID": row.ID, "Target": prediction}, ensure_ascii=False) + "\n")
                handle.flush()
            if index % args.report_every == 0 or index == len(frame):
                rate = index / max(time.time() - started, 1e-9)
                print(f"processed={index}/{len(frame)} rate={rate:.2f} lines/s")

    result = pd.DataFrame({"ID": frame.ID, "Target": [predictions[x] for x in frame.ID]})
    if result.Target.eq("").any():
        raise ValueError("Empty predictions are not allowed")
    result.to_csv(args.output, index=False)
    print(f"saved={args.output} rows={len(result)}")

    if "Target" in frame.columns:
        metrics = competition_metrics(frame.Target.tolist(), result.Target.tolist())
        print(
            f"CER={metrics.cer:.6f} WER={metrics.wer:.6f} "
            f"error={metrics.error:.6f} score={metrics.score:.6f}"
        )


if __name__ == "__main__":
    main()
