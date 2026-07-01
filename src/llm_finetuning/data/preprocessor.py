"""Tokenization and chat-template preprocessing for fine-tuning datasets."""

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from llm_finetuning.constants import IGNORE_INDEX


def apply_chat_template_and_tokenize(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> dict[str, list[int]]:
    """Tokenize one sample using the tokenizer's built-in chat template.

    Labels are masked (-100) for all tokens except the assistant's response,
    so the loss is only computed on the assistant turns.
    """
    messages = sample["messages"]
    # Full sequence: tokenize the whole conversation
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    # Build labels: start with a copy, then mask everything that is not
    # part of the assistant's response.
    labels = list(full_ids)
    labels = _mask_non_assistant_tokens(messages, tokenizer, full_ids, labels, max_length)

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _mask_non_assistant_tokens(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    full_ids: list[int],
    labels: list[int],
    max_length: int,
) -> list[int]:
    """Return labels where non-assistant tokens are set to IGNORE_INDEX."""
    # Build prefix (everything up to but not including the last assistant response)
    # by re-tokenizing the conversation without the last assistant turn.
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    if not assistant_messages:
        # No assistant turn → mask everything
        return [IGNORE_INDEX] * len(labels)

    # Locate the start of the last assistant content in the full token sequence
    # by comparing prefix (all tokens before last assistant response)
    messages_without_last = messages[:-1] if messages[-1]["role"] == "assistant" else messages

    prefix_text = tokenizer.apply_chat_template(
        messages_without_last,
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]

    prefix_len = len(prefix_ids)
    # Mask everything before the assistant response
    return [IGNORE_INDEX] * prefix_len + labels[prefix_len:]


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    num_proc: int = 1,
) -> Dataset:
    """Apply tokenization to an entire HuggingFace Dataset.

    Removes all columns except input_ids, attention_mask, and labels.
    """
    remove_cols = [c for c in dataset.column_names if c not in ("input_ids", "attention_mask", "labels")]

    tokenized = dataset.map(
        lambda sample: apply_chat_template_and_tokenize(sample, tokenizer, max_length),
        remove_columns=remove_cols,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    return tokenized
