"""Evaluation metrics: BLEU, ROUGE-L, and perplexity on a dataset."""

import math
from dataclasses import dataclass, field
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


@dataclass
class EvaluationResult:
    bleu: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    perplexity: float = float("inf")
    num_samples: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bleu": self.bleu,
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "perplexity": self.perplexity,
            "num_samples": self.num_samples,
            **self.extra,
        }


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU score using sacrebleu.

    Args:
        predictions: List of generated strings.
        references:  List of reference strings (one per prediction).

    Returns:
        BLEU score in [0.0, 100.0].
    """
    if not predictions or not references:
        return 0.0
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return float(result.score)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        dict with keys 'rouge1', 'rouge2', 'rougeL' in [0.0, 1.0].
    """
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals: dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

    n = len(predictions)
    return {k: v / n for k, v in totals.items()}


def compute_perplexity_on_dataset(losses: list[float]) -> float:
    """Compute perplexity from a list of per-sample cross-entropy losses.

    Returns exp(mean(losses)).
    """
    if not losses:
        return float("inf")
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


def evaluate(
    predictions: list[str],
    references: list[str],
    losses: list[float] | None = None,
) -> EvaluationResult:
    """Compute all evaluation metrics and return an EvaluationResult."""
    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    ppl = compute_perplexity_on_dataset(losses) if losses else float("inf")

    return EvaluationResult(
        bleu=bleu,
        rouge1=rouge["rouge1"],
        rouge2=rouge["rouge2"],
        rougeL=rouge["rougeL"],
        perplexity=ppl,
        num_samples=len(predictions),
    )
