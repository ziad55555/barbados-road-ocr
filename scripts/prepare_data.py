#!/usr/bin/env python3
"""Build deterministic local JSONL train/validation files for MLX-VLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROMPT = (
    "Transcribe this handwritten line exactly, preserving original spelling and capitalization. "
    "Output only the transcription."
)


def write_jsonl(frame: pd.DataFrame, output: Path, image_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            record = {
                "image": str((image_dir / f"{row.ID}.jpg").resolve()),
                "question": PROMPT,
                "answer": row.Target,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("Train.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/images_1024"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-size", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    frame = pd.read_csv(args.train_csv, encoding="utf-8-sig", keep_default_na=False)
    missing = [sample_id for sample_id in frame.ID if not (args.image_dir / f"{sample_id}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} training images are missing; first: {missing[0]}")

    # Stratify on transcription length so the holdout retains short and long lines.
    length_bins = pd.qcut(frame.Target.str.len(), q=8, labels=False, duplicates="drop")
    train, validation = train_test_split(
        frame,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
        stratify=length_bins,
    )
    train = train.sort_index()
    validation = validation.sort_index()
    write_jsonl(train, args.output_dir / "train.jsonl", args.image_dir)
    write_jsonl(validation, args.output_dir / "validation.jsonl", args.image_dir)
    train[["ID", "Target"]].to_csv(args.output_dir / "train.csv", index=False)
    validation[["ID", "Target"]].to_csv(args.output_dir / "validation.csv", index=False)
    print(f"train={len(train)} validation={len(validation)} seed={args.seed}")


if __name__ == "__main__":
    main()
