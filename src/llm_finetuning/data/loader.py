"""JSONL data loader — creates HuggingFace Dataset splits from a JSONL file."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    samples = []
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {p}: {exc}") from exc
    return samples


def create_dataset_splits(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Split a list of samples into train/validation/test HuggingFace DatasetDict."""
    _validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    dataset = Dataset.from_list(samples)
    # First split off test set
    test_size = test_ratio
    train_val = dataset.train_test_split(test_size=test_size, seed=seed)
    # Then split validation from the training portion
    val_fraction_of_train_val = validation_ratio / (train_ratio + validation_ratio)
    train_val_split = train_val["train"].train_test_split(test_size=val_fraction_of_train_val, seed=seed)
    return DatasetDict(
        {
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": train_val["test"],
        }
    )


def load_dataset_splits(
    path: str | Path,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Convenience wrapper: load JSONL then split into DatasetDict."""
    samples = load_jsonl(path)
    return create_dataset_splits(samples, train_ratio, validation_ratio, test_ratio, seed)


def _validate_split_ratios(train: float, validation: float, test: float) -> None:
    total = train + validation + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train}, val={validation}, test={test} (sum={total:.4f})"
        )
    for name, val in [("train", train), ("validation", validation), ("test", test)]:
        if val <= 0 or val >= 1:
            raise ValueError(f"Split ratio '{name}' must be in (0, 1), got {val}")
