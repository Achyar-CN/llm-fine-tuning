"""Shared pytest fixtures for the test suite.

All model-based tests use `sshleifer/tiny-gpt2` — a tiny GPT-2 variant
that downloads in seconds and runs on CPU. This keeps CI fast.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from llm_finetuning.data.synthetic_generator import SyntheticGeneratorConfig, generate_dataset
from llm_finetuning.data.templates import format_chat

TINY_MODEL_NAME = "sshleifer/tiny-gpt2"

_SAMPLE_MESSAGES = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI."},
        ],
        "domain": "general_qa",
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize: Deep learning uses neural networks."},
            {"role": "assistant", "content": "Deep learning uses neural networks to learn."},
        ],
        "domain": "summarization",
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Classify: I love this product!"},
            {"role": "assistant", "content": "Positive sentiment."},
        ],
        "domain": "classification",
    },
]


@pytest.fixture(scope="session")
def tiny_model_name() -> str:
    return TINY_MODEL_NAME


@pytest.fixture(scope="session")
def sample_messages() -> list[dict[str, Any]]:
    return _SAMPLE_MESSAGES


@pytest.fixture
def sample_jsonl_file(tmp_path: Path) -> Path:
    """Write sample messages to a temporary JSONL file and return the path."""
    out = tmp_path / "test_data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for sample in _SAMPLE_MESSAGES:
            f.write(json.dumps(sample) + "\n")
    return out


@pytest.fixture(scope="session")
def small_synthetic_dataset() -> list[dict[str, Any]]:
    """A small (30-sample) synthetic dataset for integration-level tests."""
    cfg = SyntheticGeneratorConfig(num_samples=30, seed=0)
    return generate_dataset(cfg)


@pytest.fixture
def small_jsonl_file(tmp_path: Path, small_synthetic_dataset: list) -> Path:
    """Write small_synthetic_dataset to a temp JSONL file."""
    out = tmp_path / "small_train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for sample in small_synthetic_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return out
