#!/usr/bin/env python3
"""Conservative train-transcription nearest-neighbour correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from road_ocr.metrics import clean_prediction, competition_metrics


def correct(prediction: str, lexicon: list[str], threshold: float) -> str:
    prediction = clean_prediction(prediction)
    if not prediction:
        return prediction
    match = process.extractOne(prediction, lexicon, scorer=Levenshtein.normalized_distance)
    if match is None:
        return prediction
    candidate, distance, _ = match
    return candidate if distance <= threshold else prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()
    train = pd.read_csv("Train.csv", encoding="utf-8-sig", keep_default_na=False)
    reference = pd.read_csv(args.reference, encoding="utf-8-sig", keep_default_na=False)
    prediction = pd.read_csv(args.prediction, encoding="utf-8-sig", keep_default_na=False)
    lexicon = train.Target.drop_duplicates().tolist()
    corrected = [correct(x, lexicon, args.threshold) for x in prediction.Target]
    result = prediction.assign(Target=corrected)
    result.to_csv(args.output, index=False)
    if "Target" in reference:
        merged = reference.merge(result, on="ID", suffixes=("_ref", "_pred"), validate="one_to_one")
        raw = competition_metrics(merged.Target_ref.tolist(), prediction.Target.tolist())
        new = competition_metrics(merged.Target_ref.tolist(), result.Target.tolist())
        print(f"raw_score={raw.score:.6f} corrected_score={new.score:.6f} threshold={args.threshold}")


if __name__ == "__main__":
    main()
