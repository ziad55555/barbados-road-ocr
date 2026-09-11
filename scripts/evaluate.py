#!/usr/bin/env python3
"""Validate a prediction CSV and print competition-aligned metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from road_ocr.metrics import competition_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference, encoding="utf-8-sig", keep_default_na=False)
    prediction = pd.read_csv(args.prediction, encoding="utf-8-sig", keep_default_na=False)
    if list(prediction.columns) != ["ID", "Target"]:
        raise ValueError("Prediction columns must be exactly ID,Target")
    if prediction.ID.duplicated().any() or prediction.Target.eq("").any():
        raise ValueError("Predictions must have unique IDs and non-empty targets")
    merged = reference[["ID", "Target"]].merge(
        prediction, on="ID", how="left", validate="one_to_one", suffixes=("_ref", "_pred")
    )
    if merged.Target_pred.isna().any() or len(prediction) != len(reference):
        raise ValueError("Prediction IDs do not match the reference exactly")
    metrics = competition_metrics(merged.Target_ref.tolist(), merged.Target_pred.tolist())
    print(f"rows={len(merged)} CER={metrics.cer:.6f} WER={metrics.wer:.6f}")
    print(f"error={metrics.error:.6f} score={metrics.score:.6f}")


if __name__ == "__main__":
    main()
