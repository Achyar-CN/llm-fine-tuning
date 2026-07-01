"""Unit tests for models/loader.py.

Uses sshleifer/tiny-gpt2 (CPU-friendly, ~50MB) for all model tests.
"""

import pytest
import torch

from llm_finetuning.config import ModelConfig
from llm_finetuning.models.loader import (
    build_bnb_config,
    load_model,
    load_model_and_tokenizer,
    load_tokenizer,
)
from llm_finetuning.models.utils import count_trainable_parameters, get_device

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


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_cfg: ModelConfig):
    return load_tokenizer(tiny_cfg)


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg: ModelConfig):
    return load_model(tiny_cfg)


# ---------------------------------------------------------------------------
# BitsAndBytes config
# ---------------------------------------------------------------------------


def test_bnb_config_none_when_no_quantization(tiny_cfg: ModelConfig) -> None:
    assert build_bnb_config(tiny_cfg) is None


def test_bnb_config_4bit() -> None:
    cfg = ModelConfig(model_name_or_path=TINY_MODEL, load_in_4bit=True)
    bnb = build_bnb_config(cfg)
    assert bnb is not None
    assert bnb.load_in_4bit is True


def test_bnb_config_8bit() -> None:
    cfg = ModelConfig(model_name_or_path=TINY_MODEL, load_in_8bit=True)
    bnb = build_bnb_config(cfg)
    assert bnb is not None
    assert bnb.load_in_8bit is True


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_loads(tiny_tokenizer) -> None:
    assert tiny_tokenizer is not None


def test_tokenizer_pad_token_set(tiny_tokenizer) -> None:
    assert tiny_tokenizer.pad_token is not None


def test_tokenizer_eos_token_set(tiny_tokenizer) -> None:
    assert tiny_tokenizer.eos_token is not None


def test_tokenizer_vocab_size_positive(tiny_tokenizer) -> None:
    assert tiny_tokenizer.vocab_size > 0


def test_tokenizer_encode_decode_roundtrip(tiny_tokenizer) -> None:
    text = "Hello, world!"
    ids = tiny_tokenizer.encode(text)
    decoded = tiny_tokenizer.decode(ids)
    assert text in decoded


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_loads(tiny_model) -> None:
    assert tiny_model is not None


def test_model_has_parameters(tiny_model) -> None:
    total = sum(p.numel() for p in tiny_model.parameters())
    assert total > 0


def test_model_dtype_float32_on_cpu(tiny_cfg: ModelConfig) -> None:
    model = load_model(tiny_cfg)
    # When loaded in float32 on CPU, all params should be float32
    for param in model.parameters():
        assert param.dtype == torch.float32
        break  # Check only first param


def test_model_device_cpu(tiny_model) -> None:
    param = next(tiny_model.parameters())
    assert param.device.type == "cpu"


# ---------------------------------------------------------------------------
# count_trainable_parameters
# ---------------------------------------------------------------------------


def test_count_trainable_all_params(tiny_model) -> None:
    trainable, total, pct = count_trainable_parameters(tiny_model)
    assert trainable > 0
    assert total >= trainable
    assert 0.0 <= pct <= 100.0


def test_count_trainable_with_frozen_params(tiny_model) -> None:
    # Freeze all params, check trainable = 0
    for p in tiny_model.parameters():
        p.requires_grad = False
    trainable, total, pct = count_trainable_parameters(tiny_model)
    assert trainable == 0
    assert pct == 0.0
    # Unfreeze for other tests
    for p in tiny_model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_is_valid_type() -> None:
    device = get_device()
    assert device.type in ("cpu", "cuda", "mps")
