"""Synthetic instruction-following dataset generator.

Generates a JSONL dataset with 5 domains:
  - general_qa      : general knowledge Q&A
  - summarization   : summarize a passage
  - classification  : classify a text snippet
  - translation     : translate English to Indonesian
  - code_explanation: explain a code snippet
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_finetuning.data.templates import SYSTEM_PROMPT_ID, format_chat

DOMAINS = ["general_qa", "summarization", "classification", "translation", "code_explanation"]

# ---------------------------------------------------------------------------
# Domain-specific templates
# ---------------------------------------------------------------------------

_QA_PAIRS: list[tuple[str, str]] = [
    (
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on building algorithms that can access data and use it to learn for themselves.",
    ),
    (
        "Explain the concept of overfitting in machine learning.",
        "Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor performance on new, unseen data. The model becomes too complex and memorizes rather than generalizes.",
    ),
    (
        "What is the difference between supervised and unsupervised learning?",
        "Supervised learning uses labeled training data where the algorithm learns to map inputs to known outputs. Unsupervised learning finds hidden patterns in data without predefined labels, discovering structure through clustering or dimensionality reduction.",
    ),
    (
        "What is gradient descent?",
        "Gradient descent is an optimization algorithm used to minimize a loss function by iteratively moving in the direction of steepest descent. It updates model parameters proportionally to the negative gradient of the loss with respect to those parameters.",
    ),
    (
        "What is a transformer architecture?",
        "A transformer is a deep learning model architecture based entirely on attention mechanisms, dispensing with recurrence. It processes all tokens in parallel using self-attention to capture long-range dependencies, making it highly efficient for sequence tasks like NLP.",
    ),
    (
        "What is transfer learning?",
        "Transfer learning is a technique where a model trained on one task is repurposed as the starting point for a model on a related task. It leverages knowledge from pre-training to improve performance with less data and compute.",
    ),
    (
        "What is regularization in machine learning?",
        "Regularization is a set of techniques used to prevent overfitting by adding a penalty term to the loss function. Common methods include L1 (Lasso), L2 (Ridge), and dropout, which constrain model complexity.",
    ),
    (
        "What is the attention mechanism?",
        "The attention mechanism allows a model to dynamically focus on relevant parts of the input when producing an output. It computes a weighted sum of values based on similarity between a query and a set of keys, enabling models to capture context effectively.",
    ),
    (
        "What is fine-tuning in the context of LLMs?",
        "Fine-tuning is the process of taking a pre-trained language model and continuing its training on a smaller, task-specific dataset. This adapts the model's weights to a new domain or task while preserving general knowledge acquired during pre-training.",
    ),
    (
        "What is LoRA (Low-Rank Adaptation)?",
        "LoRA is a parameter-efficient fine-tuning technique that freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer. This dramatically reduces the number of trainable parameters while maintaining model quality.",
    ),
]

_PASSAGES_FOR_SUMMARY: list[tuple[str, str]] = [
    (
        "Deep learning has revolutionized the field of artificial intelligence by enabling computers to learn from vast amounts of unstructured data. Unlike traditional machine learning, deep learning models automatically discover the representations needed for detection or classification directly from raw data. This capability has led to breakthroughs in image recognition, natural language processing, and speech recognition.",
        "Deep learning enables AI systems to automatically learn representations from raw unstructured data, leading to breakthroughs in image recognition, NLP, and speech recognition.",
    ),
    (
        "The transformer model, introduced in the paper 'Attention Is All You Need' in 2017, marked a paradigm shift in natural language processing. By replacing recurrent neural networks with self-attention mechanisms, transformers can process sequences in parallel, drastically reducing training time. Today, virtually all state-of-the-art language models are based on the transformer architecture.",
        "The transformer model (2017) replaced RNNs with self-attention, enabling parallel sequence processing. It is now the foundation of all state-of-the-art language models.",
    ),
    (
        "Reinforcement learning from human feedback (RLHF) is a training methodology used to align language model behavior with human preferences. The process involves collecting human comparisons between model outputs, training a reward model to predict human preferences, and then using reinforcement learning to optimize the language model against this reward model.",
        "RLHF aligns LLM behavior with human preferences by training a reward model on human comparisons and using RL to optimize the language model against it.",
    ),
    (
        "Quantization is a model compression technique that reduces the precision of model weights from 32-bit floating point to lower bit representations such as 8-bit or 4-bit integers. This significantly reduces memory usage and speeds up inference, often with minimal loss in model quality. Techniques like QLoRA combine quantization with LoRA to enable fine-tuning of large models on consumer hardware.",
        "Quantization reduces model weight precision (e.g., to 4-bit) to cut memory and speed up inference. QLoRA combines it with LoRA for efficient fine-tuning on consumer hardware.",
    ),
]

_CLASSIFICATION_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Classify the following text as positive, negative, or neutral sentiment.",
        "The new software update has significantly improved performance and the interface is much more intuitive now.",
        "Positive. The text expresses satisfaction with improved performance and a more intuitive interface.",
    ),
    (
        "Classify the following text as positive, negative, or neutral sentiment.",
        "The product arrived on time but the packaging was damaged and some items were missing.",
        "Negative. The text highlights problems with damaged packaging and missing items, indicating customer dissatisfaction.",
    ),
    (
        "Classify the following text as a question, statement, or command.",
        "Please submit your report before the end of the business day.",
        "Command. The text is a directive asking someone to perform an action by a deadline.",
    ),
    (
        "Classify the following text as technical or non-technical content.",
        "The API endpoint uses OAuth 2.0 bearer token authentication with JWT payload validation.",
        "Technical. The text contains specialized terminology (API, OAuth, JWT) indicating technical content about authentication.",
    ),
]

_TRANSLATION_PAIRS: list[tuple[str, str]] = [
    (
        "Machine learning is a subset of artificial intelligence.",
        "Machine learning adalah bagian dari kecerdasan buatan.",
    ),
    (
        "Fine-tuning a large language model requires significant computational resources.",
        "Fine-tuning model bahasa besar membutuhkan sumber daya komputasi yang signifikan.",
    ),
    (
        "The training loss decreased steadily over the epochs.",
        "Loss pelatihan menurun secara konsisten selama epoch berlangsung.",
    ),
    (
        "Attention mechanisms allow the model to focus on relevant parts of the input.",
        "Mekanisme atensi memungkinkan model untuk fokus pada bagian input yang relevan.",
    ),
    (
        "Gradient checkpointing reduces memory usage during training.",
        "Gradient checkpointing mengurangi penggunaan memori selama pelatihan.",
    ),
    (
        "The pre-trained model was fine-tuned on a domain-specific dataset.",
        "Model yang sudah di-pretrain di-fine-tune menggunakan dataset yang spesifik untuk domain tertentu.",
    ),
    (
        "Data preprocessing is a critical step in the machine learning pipeline.",
        "Preprocessing data adalah langkah krusial dalam pipeline machine learning.",
    ),
    (
        "The model achieved state-of-the-art performance on the benchmark.",
        "Model ini mencapai performa terbaik di kelasnya pada benchmark tersebut.",
    ),
]

_CODE_EXAMPLES: list[tuple[str, str]] = [
    (
        "```python\ndef compute_perplexity(loss: float) -> float:\n    import math\n    return math.exp(loss)\n```",
        "This function computes perplexity from a loss value. Perplexity is a metric used to evaluate language models — it measures how well the model predicts a sample. It is computed as e^loss (Euler's number raised to the power of the cross-entropy loss). Lower perplexity indicates a better-fitting model.",
    ),
    (
        "```python\nfrom peft import LoraConfig, get_peft_model\n\nconfig = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])\nmodel = get_peft_model(base_model, config)\n```",
        "This code applies LoRA (Low-Rank Adaptation) to a base model using the PEFT library. `LoraConfig` defines the LoRA hyperparameters: `r=16` is the rank of the decomposition matrices, `lora_alpha=32` is the scaling factor, and `target_modules` specifies which weight matrices to apply LoRA to. `get_peft_model` wraps the base model with the LoRA adapters.",
    ),
    (
        "```python\nfor batch in dataloader:\n    optimizer.zero_grad()\n    outputs = model(**batch)\n    loss = outputs.loss\n    loss.backward()\n    optimizer.step()\n```",
        "This is a standard PyTorch training loop. For each batch: (1) zero the gradients to prevent accumulation, (2) run the forward pass through the model, (3) compute the loss, (4) run backpropagation to compute gradients, and (5) update model parameters using the optimizer.",
    ),
    (
        "```python\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_name,\n    quantization_config=bnb_config,\n    device_map='auto',\n)\n```",
        "This loads a causal language model (auto-regressive, used for text generation) using HuggingFace Transformers. `quantization_config` applies BitsAndBytes quantization (e.g., 4-bit for QLoRA). `device_map='auto'` automatically distributes model layers across available GPUs and CPU memory.",
    ),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticGeneratorConfig:
    num_samples: int = 10000
    seed: int = 42
    output_path: str = "data/raw/train.jsonl"
    domain_weights: dict[str, float] = field(
        default_factory=lambda: {
            "general_qa": 0.25,
            "summarization": 0.20,
            "classification": 0.20,
            "translation": 0.20,
            "code_explanation": 0.15,
        }
    )


def _sample_domain(rng: random.Random, weights: dict[str, float]) -> str:
    domains = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(domains, weights=probs, k=1)[0]


def _generate_general_qa(rng: random.Random) -> dict[str, Any]:
    question, answer = rng.choice(_QA_PAIRS)
    return format_chat(question, answer)


def _generate_summarization(rng: random.Random) -> dict[str, Any]:
    passage, summary = rng.choice(_PASSAGES_FOR_SUMMARY)
    instruction = f"Summarize the following passage in one or two sentences:\n\n{passage}"
    return format_chat(instruction, summary)


def _generate_classification(rng: random.Random) -> dict[str, Any]:
    task, text, label = rng.choice(_CLASSIFICATION_EXAMPLES)
    instruction = f"{task}\n\nText: {text}"
    return format_chat(instruction, label)


def _generate_translation(rng: random.Random) -> dict[str, Any]:
    en, id_ = rng.choice(_TRANSLATION_PAIRS)
    instruction = f"Translate the following English sentence to Indonesian:\n\n{en}"
    return format_chat(instruction, id_, system_prompt=SYSTEM_PROMPT_ID)


def _generate_code_explanation(rng: random.Random) -> dict[str, Any]:
    code, explanation = rng.choice(_CODE_EXAMPLES)
    instruction = f"Explain what the following code does:\n\n{code}"
    return format_chat(instruction, explanation)


_DOMAIN_GENERATORS = {
    "general_qa": _generate_general_qa,
    "summarization": _generate_summarization,
    "classification": _generate_classification,
    "translation": _generate_translation,
    "code_explanation": _generate_code_explanation,
}


def generate_dataset(config: SyntheticGeneratorConfig) -> list[dict[str, Any]]:
    """Generate a list of synthetic samples according to config."""
    rng = random.Random(config.seed)
    samples = []
    for _ in range(config.num_samples):
        domain = _sample_domain(rng, config.domain_weights)
        sample = _DOMAIN_GENERATORS[domain](rng)
        sample["domain"] = domain
        samples.append(sample)
    return samples


def save_jsonl(samples: list[dict[str, Any]], path: str | Path) -> None:
    """Write samples to a JSONL file, one JSON object per line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
