"""Model utility functions: parameter counting, adapter merging, saving."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def count_trainable_parameters(model: PreTrainedModel) -> tuple[int, int, float]:
    """Return (trainable_params, all_params, trainable_pct)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def merge_and_save_adapter(
    peft_model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
) -> None:
    """Merge LoRA weights into the base model and save to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
