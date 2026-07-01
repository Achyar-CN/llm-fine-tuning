"""Model and tokenizer loading with optional QLoRA quantization."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_finetuning.config import ModelConfig


def build_bnb_config(cfg: ModelConfig) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for 4-bit or 8-bit loading, or None."""
    if cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if cfg.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Load tokenizer and ensure pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(cfg: ModelConfig) -> PreTrainedModel:
    """Load a causal language model with optional quantization."""
    bnb_config = build_bnb_config(cfg)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.torch_dtype, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        device_map=cfg.device_map,
        trust_remote_code=True,
    )

    if bnb_config is not None and (cfg.load_in_4bit or cfg.load_in_8bit):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model


def load_model_and_tokenizer(
    cfg: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience wrapper: load model and tokenizer together."""
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    return model, tokenizer
