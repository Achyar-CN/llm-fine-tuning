"""Unit tests for models/peft_config.py."""

import pytest
from peft import LoraConfig

from llm_finetuning.config import ModelConfig
from llm_finetuning.models.loader import load_model
from llm_finetuning.models.peft_config import apply_lora, create_lora_config
from llm_finetuning.models.utils import count_trainable_parameters

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        model_name_or_path=TINY_MODEL,
        model_family="gpt2",
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype="float32",
        device_map="cpu",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["c_attn"],
    )


# ---------------------------------------------------------------------------
# create_lora_config
# ---------------------------------------------------------------------------


def test_create_lora_config_returns_lora_config(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert isinstance(cfg, LoraConfig)


def test_lora_rank_positive(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert cfg.r > 0


def test_lora_alpha_positive(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert cfg.lora_alpha > 0


def test_lora_target_modules_not_empty(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert len(cfg.target_modules) > 0


def test_lora_alpha_double_rank_convention(tiny_cfg: ModelConfig) -> None:
    """Best practice: alpha should be 2 × r."""
    cfg = create_lora_config(tiny_cfg)
    assert cfg.lora_alpha == 2 * cfg.r, f"Expected alpha={2 * cfg.r}, got {cfg.lora_alpha}"


def test_lora_dropout_valid_range(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert 0.0 <= cfg.lora_dropout < 1.0


def test_lora_task_type_causal_lm(tiny_cfg: ModelConfig) -> None:
    from peft import TaskType

    cfg = create_lora_config(tiny_cfg)
    assert cfg.task_type == TaskType.CAUSAL_LM


def test_lora_bias_none(tiny_cfg: ModelConfig) -> None:
    cfg = create_lora_config(tiny_cfg)
    assert cfg.bias == "none"


# ---------------------------------------------------------------------------
# apply_lora
# ---------------------------------------------------------------------------


def test_apply_lora_reduces_trainable_params(tiny_cfg: ModelConfig) -> None:
    """After applying LoRA, trainable params should be << total params."""
    base_model = load_model(tiny_cfg)
    _, total_before, _ = count_trainable_parameters(base_model)

    peft_model = apply_lora(base_model, tiny_cfg)
    trainable, total_after, pct = count_trainable_parameters(peft_model)

    assert trainable < total_after, "LoRA should freeze most base model params"
    assert pct < 100.0, f"Trainable pct should be < 100%, got {pct:.2f}%"


def test_apply_lora_model_still_callable(tiny_cfg: ModelConfig) -> None:
    """A LoRA-wrapped model must still be able to do a forward pass."""
    import torch

    base_model = load_model(tiny_cfg)
    peft_model = apply_lora(base_model, tiny_cfg)
    peft_model.eval()

    dummy_input = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = peft_model(dummy_input)
    assert output.logits is not None
