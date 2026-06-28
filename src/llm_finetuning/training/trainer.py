"""SFTTrainer setup and training entry point."""

from pathlib import Path

from datasets import DatasetDict
from peft import PeftModel
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import SFTTrainer

from llm_finetuning.config import TrainingConfig
from llm_finetuning.training.callbacks import EarlyStoppingOnLossCallback, MetricsLoggerCallback


def build_training_arguments(cfg: TrainingConfig) -> TrainingArguments:
    """Convert TrainingConfig to a HuggingFace TrainingArguments object."""
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        dataloader_pin_memory=False,
    )


def build_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    cfg: TrainingConfig,
    log_file: str | Path = "outputs/logs/metrics.jsonl",
) -> SFTTrainer:
    """Assemble and return a configured SFTTrainer."""
    training_args = build_training_arguments(cfg)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )
    callbacks = [
        MetricsLoggerCallback(log_file=log_file),
        EarlyStoppingOnLossCallback(patience=cfg.early_stopping_patience),
    ]
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        max_seq_length=None,
        dataset_text_field=None,
        packing=False,
    )
