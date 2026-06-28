"""Training-time metrics: perplexity and token accuracy."""

import math
from typing import Any

import numpy as np


def compute_perplexity(eval_loss: float) -> float:
    """Compute perplexity from cross-entropy eval loss.

    Perplexity = exp(loss). Valid range: [1, +inf).
    Lower is better.
    """
    return math.exp(eval_loss)


def compute_token_accuracy(predictions: np.ndarray, labels: np.ndarray, ignore_index: int = -100) -> float:
    """Compute the fraction of non-masked tokens predicted correctly.

    Args:
        predictions: Array of predicted token IDs, shape (batch, seq_len).
        labels:      Array of target token IDs, shape (batch, seq_len).
                     Positions with ignore_index are excluded.
        ignore_index: Value used to mask non-target positions.

    Returns:
        Accuracy in [0.0, 1.0].
    """
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (predictions[mask] == labels[mask]).sum()
    return float(correct) / float(mask.sum())


def compute_metrics_from_eval_output(eval_output: Any) -> dict[str, float]:
    """Extract perplexity from a HuggingFace EvalLoopOutput."""
    loss = eval_output.metrics.get("eval_loss", float("inf"))
    return {
        "eval_loss": loss,
        "eval_perplexity": compute_perplexity(loss),
    }
