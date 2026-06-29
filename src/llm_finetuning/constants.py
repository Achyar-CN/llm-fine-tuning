"""Shared constants across the fine-tuning pipeline."""

# Supported model families and their chat template identifiers
MODEL_FAMILIES = {
    "tinyllama": "llama",
    "llama": "llama",
    "mistral": "mistral",
    "qwen": "qwen",
    "gemma": "gemma",
    "gpt2": "gpt2",
}

# Default target modules per model family for LoRA
LORA_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj"],
}

# Label mask value for tokens that should not contribute to loss
IGNORE_INDEX = -100

# Minimum/maximum acceptable hyperparameter ranges (used in config validation)
HP_BOUNDS = {
    "learning_rate": (1e-8, 1e-1),
    "num_train_epochs": (1, 10),
    "warmup_ratio": (0.0, 0.3),
    "lora_r": (1, 256),
}
