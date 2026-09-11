"""Competition-aligned OCR metrics and text normalization checks."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class OCRMetrics:
    cer: float
    wer: float

    @property
    def error(self) -> float:
        return 0.5 * (self.cer + self.wer)

    @property
    def score(self) -> float:
        return 1.0 - self.error


def _aggregate_error(references: list[list[str]], predictions: list[list[str]]) -> float:
    edits = sum(Levenshtein.distance(ref, pred) for ref, pred in zip(references, predictions))
    length = sum(len(ref) for ref in references)
    return edits / max(length, 1)


def competition_metrics(references: list[str], predictions: list[str]) -> OCRMetrics:
    """Compute length-weighted CER/WER via corpus-level edit counts."""

    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    char_refs = [list(x) for x in references]
    char_preds = [list(x) for x in predictions]
    word_refs = [x.split() for x in references]
    word_preds = [x.split() for x in predictions]
    return OCRMetrics(
        cer=_aggregate_error(char_refs, char_preds),
        wer=_aggregate_error(word_refs, word_preds),
    )


def clean_prediction(text: str) -> str:
    """Remove model wrappers without changing genuine transcription content."""

    text = text.replace("\r", " ").replace("\n", " ").strip()
    prefixes = (
        "Transcription:",
        "The transcription is:",
        "The handwritten text reads:",
    )
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'\"', "'"}:
        text = text[1:-1]
    return " ".join(text.split())
