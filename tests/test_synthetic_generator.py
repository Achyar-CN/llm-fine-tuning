"""Unit tests for synthetic data generator."""

import pytest

from llm_finetuning.data.synthetic_generator import (
    DOMAINS,
    SyntheticGeneratorConfig,
    generate_dataset,
)
from llm_finetuning.data.templates import validate_message_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_config() -> SyntheticGeneratorConfig:
    return SyntheticGeneratorConfig(num_samples=100, seed=42)


@pytest.fixture(scope="module")
def dataset(default_config: SyntheticGeneratorConfig) -> list:
    return generate_dataset(default_config)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_output_schema_valid(dataset: list) -> None:
    """Every sample must pass the messages schema validator."""
    for i, sample in enumerate(dataset):
        assert validate_message_schema(sample), f"Sample {i} failed schema validation: {sample}"


def test_messages_field_present(dataset: list) -> None:
    """Every sample must contain a 'messages' key."""
    for sample in dataset:
        assert "messages" in sample


def test_required_roles_present(dataset: list) -> None:
    """Every sample must have at least a 'user' and 'assistant' turn."""
    for sample in dataset:
        roles = {msg["role"] for msg in sample["messages"]}
        assert "user" in roles, f"Missing 'user' role in: {sample}"
        assert "assistant" in roles, f"Missing 'assistant' role in: {sample}"


def test_no_empty_content(dataset: list) -> None:
    """No message content should be empty or whitespace-only."""
    for sample in dataset:
        for msg in sample["messages"]:
            assert msg["content"].strip(), f"Empty content in role '{msg['role']}': {sample}"


def test_valid_roles_only(dataset: list) -> None:
    """All roles must be one of: system, user, assistant."""
    valid_roles = {"system", "user", "assistant"}
    for sample in dataset:
        for msg in sample["messages"]:
            assert msg["role"] in valid_roles, f"Invalid role '{msg['role']}'"


# ---------------------------------------------------------------------------
# Volume and distribution tests
# ---------------------------------------------------------------------------


def test_sample_count(default_config: SyntheticGeneratorConfig, dataset: list) -> None:
    """Generated dataset size must match the requested num_samples."""
    assert len(dataset) == default_config.num_samples


def test_domain_field_present(dataset: list) -> None:
    """Each sample must carry its domain label."""
    for sample in dataset:
        assert "domain" in sample
        assert sample["domain"] in DOMAINS


def test_domain_distribution_balanced(dataset: list) -> None:
    """Each domain should appear — rough balance check (no domain < 5%)."""
    from collections import Counter

    counts = Counter(sample["domain"] for sample in dataset)
    total = len(dataset)
    for domain in DOMAINS:
        ratio = counts[domain] / total
        assert ratio > 0.05, f"Domain '{domain}' under-represented: {ratio:.2%}"


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output() -> None:
    """Two runs with the same seed must produce identical datasets."""
    cfg = SyntheticGeneratorConfig(num_samples=50, seed=99)
    ds1 = generate_dataset(cfg)
    ds2 = generate_dataset(cfg)
    assert ds1 == ds2


def test_different_seeds_produce_different_output() -> None:
    """Different seeds should (with very high probability) yield different datasets."""
    ds1 = generate_dataset(SyntheticGeneratorConfig(num_samples=50, seed=1))
    ds2 = generate_dataset(SyntheticGeneratorConfig(num_samples=50, seed=2))
    assert ds1 != ds2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_sample_generation() -> None:
    """Generator must handle num_samples=1 without error."""
    cfg = SyntheticGeneratorConfig(num_samples=1, seed=0)
    dataset = generate_dataset(cfg)
    assert len(dataset) == 1
    assert validate_message_schema(dataset[0])


def test_large_sample_count_no_error() -> None:
    """Generating 1000 samples should complete without exceptions."""
    cfg = SyntheticGeneratorConfig(num_samples=1000, seed=7)
    dataset = generate_dataset(cfg)
    assert len(dataset) == 1000
