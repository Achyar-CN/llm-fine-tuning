"""Custom HuggingFace training callbacks."""

import json
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llm_finetuning.training.metrics import compute_perplexity


class MetricsLoggerCallback(TrainerCallback):
    """Append training metrics (loss, perplexity) to a JSON Lines file after each evaluation."""

    def __init__(self, log_file: str | Path = "outputs/logs/metrics.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> None:
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 4),
            **metrics,
        }
        if "eval_loss" in metrics:
            record["eval_perplexity"] = compute_perplexity(metrics["eval_loss"])

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class EarlyStoppingOnLossCallback(TrainerCallback):
    """Stop training if eval_loss does not improve for `patience` evaluations."""

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self._best_loss = float("inf")
        self._no_improve_count = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ) -> None:
        eval_loss = metrics.get("eval_loss", float("inf"))
        if eval_loss < self._best_loss:
            self._best_loss = eval_loss
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
            if self._no_improve_count >= self.patience:
                control.should_training_stop = True
