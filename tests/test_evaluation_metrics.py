"""Unit tests for evaluation/metrics.py."""

import math

import pytest

from llm_finetuning.evaluation.metrics import (
    EvaluationResult,
    compute_bleu,
    compute_perplexity_on_dataset,
    compute_rouge,
    evaluate,
)


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------


def test_bleu_identical_prediction() -> None:
    """Identical prediction and reference should yield BLEU = 100.0."""
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    score = compute_bleu(preds, refs)
    assert score == pytest.approx(100.0, abs=0.1)


def test_bleu_completely_wrong_prediction() -> None:
    """Completely different prediction should yield BLEU near 0."""
    preds = ["xyz abc def ghi jkl"]
    refs = ["the cat sat on the mat"]
    score = compute_bleu(preds, refs)
    assert score == pytest.approx(0.0, abs=1.0)


def test_bleu_valid_range() -> None:
    preds = ["machine learning is a subset of artificial intelligence"]
    refs = ["machine learning is part of AI"]
    score = compute_bleu(preds, refs)
    assert 0.0 <= score <= 100.0


def test_bleu_empty_inputs_returns_zero() -> None:
    assert compute_bleu([], []) == 0.0


def test_bleu_higher_for_better_match() -> None:
    preds_good = ["the quick brown fox"]
    preds_bad = ["totally different words here"]
    ref = ["the quick brown fox jumps"]
    score_good = compute_bleu(preds_good, ref)
    score_bad = compute_bleu(preds_bad, ref)
    assert score_good > score_bad


# ---------------------------------------------------------------------------
# compute_rouge
# ---------------------------------------------------------------------------


def test_rouge_identical_returns_one() -> None:
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    scores = compute_rouge(preds, refs)
    assert scores["rougeL"] == pytest.approx(1.0, abs=1e-6)


def test_rouge_empty_inputs_returns_zero() -> None:
    scores = compute_rouge([], [])
    assert scores["rouge1"] == 0.0
    assert scores["rougeL"] == 0.0


def test_rouge_keys_present() -> None:
    scores = compute_rouge(["hello world"], ["hello there"])
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores


def test_rouge_values_in_valid_range() -> None:
    preds = ["machine learning enables computers to learn from data"]
    refs = ["machine learning allows systems to learn automatically"]
    scores = compute_rouge(preds, refs)
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_rougeL_higher_for_better_match() -> None:
    preds_good = ["the quick brown fox jumps over"]
    preds_bad = ["completely unrelated output text here"]
    ref = ["the quick brown fox jumps over the lazy dog"]
    good_score = compute_rouge(preds_good, ref)["rougeL"]
    bad_score = compute_rouge(preds_bad, ref)["rougeL"]
    assert good_score > bad_score


# ---------------------------------------------------------------------------
# compute_perplexity_on_dataset
# ---------------------------------------------------------------------------


def test_perplexity_from_zero_losses() -> None:
    """exp(0) = 1.0 — perfect model."""
    assert compute_perplexity_on_dataset([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_perplexity_always_ge_one() -> None:
    for loss in [0.1, 0.5, 1.0, 2.0, 5.0]:
        assert compute_perplexity_on_dataset([loss]) >= 1.0


def test_perplexity_empty_returns_inf() -> None:
    assert compute_perplexity_on_dataset([]) == float("inf")


def test_perplexity_averages_losses() -> None:
    losses = [1.0, 2.0, 3.0]
    expected = math.exp(sum(losses) / len(losses))
    assert compute_perplexity_on_dataset(losses) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


def test_evaluation_result_has_all_fields() -> None:
    result = EvaluationResult(
        bleu=50.0, rouge1=0.6, rouge2=0.4, rougeL=0.55,
        perplexity=3.2, num_samples=100
    )
    assert result.bleu == 50.0
    assert result.rouge1 == 0.6
    assert result.rougeL == 0.55
    assert result.perplexity == 3.2
    assert result.num_samples == 100


def test_evaluation_result_to_dict_complete() -> None:
    result = EvaluationResult(bleu=10.0, rouge1=0.3, rouge2=0.1, rougeL=0.25,
                               perplexity=5.0, num_samples=50)
    d = result.to_dict()
    for key in ["bleu", "rouge1", "rouge2", "rougeL", "perplexity", "num_samples"]:
        assert key in d


def test_evaluation_result_default_values() -> None:
    result = EvaluationResult()
    assert result.bleu == 0.0
    assert result.perplexity == float("inf")
    assert result.num_samples == 0


# ---------------------------------------------------------------------------
# evaluate (combined)
# ---------------------------------------------------------------------------


def test_evaluate_end_to_end() -> None:
    preds = ["machine learning is a field of AI"]
    refs = ["machine learning is a subset of artificial intelligence"]
    losses = [1.5]
    result = evaluate(preds, refs, losses)
    assert result.num_samples == 1
    assert 0.0 <= result.bleu <= 100.0
    assert 0.0 <= result.rougeL <= 1.0
    assert result.perplexity >= 1.0
