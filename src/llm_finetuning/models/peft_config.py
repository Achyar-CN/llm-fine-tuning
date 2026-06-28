"""LoRA / QLoRA PEFT configuration builder."""

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import PreTrainedModel

from llm_finetuning.config import ModelConfig
from llm_finetuning.constants import LORA_TARGET_MODULES


def create_lora_config(cfg: ModelConfig) -> LoraConfig:
    """Build a LoraConfig from ModelConfig.

    Falls back to model-family defaults if target_modules is not set.
    """
    target_modules = cfg.lora_target_modules or LORA_TARGET_MODULES.get(
        cfg.model_family, ["q_proj", "v_proj"]
    )
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model: PreTrainedModel, cfg: ModelConfig) -> PeftModel:
    """Wrap a base model with LoRA adapters and print trainable param info."""
    lora_cfg = create_lora_config(cfg)
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model
