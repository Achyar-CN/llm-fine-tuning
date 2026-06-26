"""Central configuration dataclasses for the fine-tuning pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    num_samples: int = 10000
    seed: int = 42
    output_path: str = "data/raw/train.jsonl"
    domain_weights: dict = field(
        default_factory=lambda: {
            "general_qa": 0.25,
            "summarization": 0.20,
            "classification": 0.20,
            "translation": 0.20,
            "code_explanation": 0.15,
        }
    )
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    max_length: int = 512


@dataclass
class ModelConfig:
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_family: str = "llama"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    early_stopping_patience: int = 3


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML file and return it as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_config_from_yaml(path: str | Path) -> DataConfig:
    raw = load_yaml_config(path)
    split = raw.pop("split", {})
    cfg = DataConfig(**{k: v for k, v in raw.items() if k != "split"})
    if split:
        cfg.train_ratio = split.get("train", cfg.train_ratio)
        cfg.validation_ratio = split.get("validation", cfg.validation_ratio)
        cfg.test_ratio = split.get("test", cfg.test_ratio)
    return cfg


def model_config_from_yaml(path: str | Path) -> ModelConfig:
    raw = load_yaml_config(path)
    return ModelConfig(**raw)


def training_config_from_yaml(path: str | Path) -> TrainingConfig:
    raw = load_yaml_config(path)
    return TrainingConfig(**raw)
