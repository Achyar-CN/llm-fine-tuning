"""Unit tests for data/loader.py."""

import json
from pathlib import Path

import pytest

from llm_finetuning.data.loader import (
    _validate_split_ratios,
    create_dataset_splits,
    load_dataset_splits,
    load_jsonl,
)

# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_valid(sample_jsonl_file: Path) -> None:
    samples = load_jsonl(sample_jsonl_file)
    assert len(samples) == 3


def test_load_jsonl_preserves_content(sample_jsonl_file: Path) -> None:
    samples = load_jsonl(sample_jsonl_file)
    assert "messages" in samples[0]
    assert samples[0]["messages"][0]["role"] == "system"


def test_load_jsonl_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nonexistent.jsonl")


def test_load_jsonl_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(bad_file)


def test_load_jsonl_skips_empty_lines(tmp_path: Path, sample_messages: list) -> None:
    out = tmp_path / "with_blanks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_messages[0]) + "\n")
        f.write("\n")  # blank line
        f.write(json.dumps(sample_messages[1]) + "\n")
    samples = load_jsonl(out)
    assert len(samples) == 2


# ---------------------------------------------------------------------------
# _validate_split_ratios
# ---------------------------------------------------------------------------


def test_valid_split_ratios() -> None:
    _validate_split_ratios(0.8, 0.1, 0.1)  # Should not raise


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        _validate_split_ratios(0.7, 0.1, 0.1)


def test_split_ratio_zero_raises() -> None:
    with pytest.raises(ValueError):
        _validate_split_ratios(0.9, 0.1, 0.0)


def test_split_ratio_one_raises() -> None:
    with pytest.raises(ValueError):
        _validate_split_ratios(1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# create_dataset_splits
# ---------------------------------------------------------------------------


def test_split_produces_three_keys(small_synthetic_dataset: list) -> None:
    splits = create_dataset_splits(small_synthetic_dataset)
    assert set(splits.keys()) == {"train", "validation", "test"}


def test_split_sizes_approximately_correct(small_synthetic_dataset: list) -> None:
    n = len(small_synthetic_dataset)
    splits = create_dataset_splits(small_synthetic_dataset, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
    # Allow ±2 samples tolerance due to integer rounding
    assert abs(len(splits["train"]) - int(n * 0.8)) <= 2
    assert abs(len(splits["test"]) - int(n * 0.1)) <= 2


def test_split_total_equals_input(small_synthetic_dataset: list) -> None:
    splits = create_dataset_splits(small_synthetic_dataset)
    total = sum(len(splits[k]) for k in splits)
    assert total == len(small_synthetic_dataset)


def test_split_reproducibility(small_synthetic_dataset: list) -> None:
    splits_a = create_dataset_splits(small_synthetic_dataset, seed=7)
    splits_b = create_dataset_splits(small_synthetic_dataset, seed=7)
    assert splits_a["train"][0] == splits_b["train"][0]


def test_different_seeds_give_different_splits(small_synthetic_dataset: list) -> None:
    splits_a = create_dataset_splits(small_synthetic_dataset, seed=1)
    splits_b = create_dataset_splits(small_synthetic_dataset, seed=2)
    # With high probability the first train samples differ
    assert splits_a["train"][0] != splits_b["train"][0]


# ---------------------------------------------------------------------------
# load_dataset_splits (integration)
# ---------------------------------------------------------------------------


def test_load_dataset_splits_end_to_end(small_jsonl_file: Path) -> None:
    splits = load_dataset_splits(small_jsonl_file)
    assert "train" in splits
    assert len(splits["train"]) > 0
