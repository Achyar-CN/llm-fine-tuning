"""CLI script: run full fine-tuning pipeline.

Usage:
    python scripts/train.py \\
        --model configs/model/tinyllama_1b.yaml \\
        --training configs/training/sft_lora.yaml \\
        --data configs/data/synthetic_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from llm_finetuning.config import (
    data_config_from_yaml,
    model_config_from_yaml,
    training_config_from_yaml,
)
from llm_finetuning.data.loader import load_dataset_splits
from llm_finetuning.data.preprocessor import preprocess_dataset
from llm_finetuning.models.loader import load_model_and_tokenizer
from llm_finetuning.models.peft_config import apply_lora
from llm_finetuning.models.utils import count_trainable_parameters, merge_and_save_adapter
from llm_finetuning.training.trainer import build_trainer
from llm_finetuning.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with LoRA/QLoRA")
    parser.add_argument("--model", type=str, default="configs/model/tinyllama_1b.yaml")
    parser.add_argument("--training", type=str, default="configs/training/sft_lora.yaml")
    parser.add_argument("--data", type=str, default="configs/data/synthetic_config.yaml")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Override JSONL data path from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = model_config_from_yaml(args.model)
    train_cfg = training_config_from_yaml(args.training)
    data_cfg = data_config_from_yaml(args.data)

    set_seed(data_cfg.seed)
    logger.info(f"Model: {model_cfg.model_name_or_path}")
    logger.info(f"Training config: {args.training}")

    # --- Data ---
    data_path = args.data_path or data_cfg.output_path
    logger.info(f"Loading data from {data_path}...")
    dataset = load_dataset_splits(
        data_path,
        train_ratio=data_cfg.train_ratio,
        validation_ratio=data_cfg.validation_ratio,
        test_ratio=data_cfg.test_ratio,
        seed=data_cfg.seed,
    )
    logger.info(f"Dataset splits: {', '.join(f'{k}={len(v)}' for k, v in dataset.items())}")

    # --- Model & Tokenizer ---
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # --- LoRA ---
    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, model_cfg)
    trainable, total, pct = count_trainable_parameters(model)
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    # --- Tokenize ---
    logger.info("Tokenizing dataset...")
    tokenized = {
        split: preprocess_dataset(dataset[split], tokenizer, max_length=data_cfg.max_length)
        for split in dataset
    }
    from datasets import DatasetDict
    tokenized_dict = DatasetDict(tokenized)

    # --- Train ---
    logger.info("Starting training...")
    trainer = build_trainer(model, tokenizer, tokenized_dict, train_cfg)
    trainer.train()

    # --- Save ---
    final_dir = Path(train_cfg.output_dir).parent / "final_adapter"
    logger.info(f"Merging and saving adapter to {final_dir}...")
    merge_and_save_adapter(model, tokenizer, final_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
