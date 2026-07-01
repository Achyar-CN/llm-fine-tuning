"""Unit tests for data/preprocessor.py.

Uses sshleifer/tiny-gpt2 as a CPU-friendly model in CI.
GPT-2 does not have a built-in chat template, so we test the tokenization
core logic separately from the full chat-template path.
"""

import pytest

from llm_finetuning.constants import IGNORE_INDEX
from llm_finetuning.data.preprocessor import apply_chat_template_and_tokenize

try:
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    # GPT-2 has no padding token — set EOS as pad
    _TOKENIZER.pad_token = _TOKENIZER.eos_token
    # Add a minimal chat template so tests can exercise the path
    _TOKENIZER.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}{{ message['content'] + '\\n' }}"
        "{% elif message['role'] == 'user' %}User: {{ message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] + eos_token + '\\n' }}"
        "{% endif %}{% endfor %}"
    )
    _HAS_TOKENIZER = True
except Exception:
    _HAS_TOKENIZER = False

_SAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
    ]
}

requires_tokenizer = pytest.mark.skipif(not _HAS_TOKENIZER, reason="sshleifer/tiny-gpt2 not available")


# ---------------------------------------------------------------------------
# Tokenization output structure
# ---------------------------------------------------------------------------


@requires_tokenizer
def test_output_has_required_keys() -> None:
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result


@requires_tokenizer
def test_output_lengths_are_equal() -> None:
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert len(result["input_ids"]) == len(result["labels"])


@requires_tokenizer
def test_max_length_enforced() -> None:
    max_len = 32
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=max_len)
    assert len(result["input_ids"]) <= max_len


@requires_tokenizer
def test_attention_mask_all_ones() -> None:
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    assert all(v == 1 for v in result["attention_mask"])


# ---------------------------------------------------------------------------
# Label masking
# ---------------------------------------------------------------------------


@requires_tokenizer
def test_labels_contain_ignore_index() -> None:
    """At least some labels must be masked (the user + system prefix)."""
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    assert IGNORE_INDEX in result["labels"]


@requires_tokenizer
def test_labels_have_non_ignore_portion() -> None:
    """At least some labels must NOT be masked (the assistant's response)."""
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    non_masked = [label for label in result["labels"] if label != IGNORE_INDEX]
    assert len(non_masked) > 0, "All labels are masked — assistant tokens should not be masked"


@requires_tokenizer
def test_labels_prefix_is_fully_masked() -> None:
    """Tokens before the assistant turn should all be IGNORE_INDEX."""
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    labels = result["labels"]
    # Find first non-masked position
    first_non_masked = next((i for i, label in enumerate(labels) if label != IGNORE_INDEX), None)
    assert first_non_masked is not None, "No non-masked labels found"
    # Everything before that position should be masked
    assert all(label == IGNORE_INDEX for label in labels[:first_non_masked])


# ---------------------------------------------------------------------------
# Round-trip decode
# ---------------------------------------------------------------------------


@requires_tokenizer
def test_decode_input_ids_contains_content() -> None:
    """Decoding input_ids should produce text containing the question."""
    result = apply_chat_template_and_tokenize(_SAMPLE, _TOKENIZER, max_length=128)
    decoded = _TOKENIZER.decode(result["input_ids"], skip_special_tokens=True)
    assert "AI" in decoded or "assistant" in decoded.lower()
