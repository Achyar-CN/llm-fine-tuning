"""Unit tests for training configuration validation and training metrics."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_finetuning.config import TrainingConfig, training_config_from_yaml
from llm_finetuning.constants import HP_BOUNDS
from llm_finetuning.training.metrics import (
    compute_perplexity,
    compute_token_accuracy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "training"


@pytest.fixture(params=["sft_lora.yaml", "sft_qlora.yaml", "dpo.yaml"])
def training_yaml_path(request) -> Path:
    return CONFIGS_DIR / request.param


@pytest.fixture
def default_config() -> TrainingConfig:
    return TrainingConfig()


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def test_training_yaml_loads_without_error(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    assert isinstance(cfg, TrainingConfig)


def test_required_fields_present(training_yaml_path: Path) -> None:
    raw = yaml.safe_load(training_yaml_path.read_text())
    required = [
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "warmup_ratio",
    ]
    for field in required:
        assert field in raw, f"Required field '{field}' missing from {training_yaml_path.name}"


# ---------------------------------------------------------------------------
# Hyperparameter bound validation
# ---------------------------------------------------------------------------


def test_learning_rate_in_valid_range(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    lo, hi = HP_BOUNDS["learning_rate"]
    assert lo < cfg.learning_rate < hi, f"learning_rate {cfg.learning_rate} out of range ({lo}, {hi})"


def test_num_epochs_in_valid_range(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    lo, hi = HP_BOUNDS["num_train_epochs"]
    assert lo <= cfg.num_train_epochs <= hi


def test_warmup_ratio_in_valid_range(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    lo, hi = HP_BOUNDS["warmup_ratio"]
    assert lo <= cfg.warmup_ratio <= hi


def test_effective_batch_size_realistic(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    effective_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    assert 2 <= effective_bs <= 512, f"Effective batch size {effective_bs} outside realistic range [2, 512]"


def test_save_steps_positive(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    assert cfg.save_steps > 0


def test_max_grad_norm_positive(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    assert cfg.max_grad_norm > 0


def test_early_stopping_patience_positive(training_yaml_path: Path) -> None:
    cfg = training_config_from_yaml(training_yaml_path)
    assert cfg.early_stopping_patience >= 1


# ---------------------------------------------------------------------------
# compute_perplexity
# ---------------------------------------------------------------------------


def test_perplexity_from_zero_loss() -> None:
    """exp(0) = 1.0 — perfect model."""
    assert compute_perplexity(0.0) == pytest.approx(1.0)


def test_perplexity_from_ln2() -> None:
    """exp(ln(2)) = 2.0."""
    assert compute_perplexity(math.log(2)) == pytest.approx(2.0, rel=1e-5)


def test_perplexity_always_ge_one() -> None:
    for loss in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        assert compute_perplexity(loss) >= 1.0


def test_perplexity_increases_with_loss() -> None:
    assert compute_perplexity(1.0) < compute_perplexity(2.0)


# ---------------------------------------------------------------------------
# compute_token_accuracy
# ---------------------------------------------------------------------------


def test_token_accuracy_perfect() -> None:
    preds = np.array([[1, 2, 3]])
    labels = np.array([[1, 2, 3]])
    assert compute_token_accuracy(preds, labels) == pytest.approx(1.0)


def test_token_accuracy_zero() -> None:
    preds = np.array([[1, 2, 3]])
    labels = np.array([[4, 5, 6]])
    assert compute_token_accuracy(preds, labels) == pytest.approx(0.0)


def test_token_accuracy_partial() -> None:
    preds = np.array([[1, 2, 3, 4]])
    labels = np.array([[1, 2, 9, 9]])
    assert compute_token_accuracy(preds, labels) == pytest.approx(0.5)


def test_token_accuracy_masked_positions_excluded() -> None:
    preds = np.array([[1, 2, 3]])
    labels = np.array([[-100, 2, 3]])  # first token masked
    acc = compute_token_accuracy(preds, labels, ignore_index=-100)
    assert acc == pytest.approx(1.0), "Masked positions should not affect accuracy"


def test_token_accuracy_all_masked_returns_zero() -> None:
    preds = np.array([[1, 2, 3]])
    labels = np.array([[-100, -100, -100]])
    assert compute_token_accuracy(preds, labels, ignore_index=-100) == 0.0


def test_token_accuracy_in_valid_range() -> None:
    rng = np.random.default_rng(42)
    preds = rng.integers(0, 100, size=(8, 64))
    labels = rng.integers(0, 100, size=(8, 64))
    acc = compute_token_accuracy(preds, labels)
    assert 0.0 <= acc <= 1.0
