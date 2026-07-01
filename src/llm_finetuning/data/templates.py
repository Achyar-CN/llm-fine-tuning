"""Chat format templates for instruction-following fine-tuning.

All models in this project use the ChatML / OpenAI messages format:
  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

HuggingFace tokenizers handle model-specific formatting via apply_chat_template(),
so this module only needs to produce the universal messages list.
"""

from typing import Any

# Model family to HuggingFace tokenizer chat template key
# (used to validate that the tokenizer has a suitable template at load time)
MODEL_FAMILY_TEMPLATES: dict[str, str] = {
    "llama": "llama-3",  # TinyLlama, Llama 3.x
    "qwen": "qwen",  # Qwen2, Qwen2.5 (multilingual, incl. Bahasa Indonesia)
    "mistral": "mistral",
    "gemma": "gemma",
    "gpt2": "gpt2",  # No official chat template — set manually in tests
}


SYSTEM_PROMPT_DEFAULT = (
    "You are a helpful, harmless, and honest AI assistant. " "Answer questions clearly and concisely."
)

SYSTEM_PROMPT_ID = (
    "Kamu adalah asisten AI yang membantu, aman, dan jujur. "
    "Jawab pertanyaan dengan jelas dan ringkas dalam Bahasa Indonesia."
)


def format_alpaca(instruction: str, input_text: str, output: str) -> dict[str, Any]:
    """Alpaca-style instruction format (single turn)."""
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def format_chat(
    user_message: str,
    assistant_message: str,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT,
) -> dict[str, Any]:
    """ChatML / OpenAI-style multi-turn format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def validate_message_schema(sample: dict[str, Any]) -> bool:
    """Return True if sample has valid messages schema."""
    if "messages" not in sample:
        return False
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in valid_roles:
            return False
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False
    return True
