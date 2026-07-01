# LLM Fine-Tuning Pipeline

[![Tests](https://github.com/achyar-cn/llm-fine-tuning/actions/workflows/tests.yml/badge.svg)](https://github.com/achyar-cn/llm-fine-tuning/actions/workflows/tests.yml)
[![Code Quality](https://github.com/achyar-cn/llm-fine-tuning/actions/workflows/lint.yml/badge.svg)](https://github.com/achyar-cn/llm-fine-tuning/actions/workflows/lint.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)

End-to-end LLM fine-tuning pipeline using **LoRA / QLoRA** with the HuggingFace ecosystem. Demonstrates best practices for parameter-efficient fine-tuning, including synthetic data generation, tokenization, training, evaluation, and multi-model progression.

---

## Architecture

```
llm-fine-tuning/
├── src/llm_finetuning/
│   ├── config.py              # Dataclass configs (Model, Training, Data)
│   ├── constants.py           # Shared constants and HP bounds
│   ├── data/
│   │   ├── synthetic_generator.py  # Template-based dataset generation (5 domains)
│   │   ├── loader.py               # JSONL → HuggingFace DatasetDict
│   │   ├── preprocessor.py         # Chat template + tokenization + label masking
│   │   └── templates.py            # ChatML format templates
│   ├── models/
│   │   ├── loader.py          # AutoModel + BitsAndBytes QLoRA
│   │   ├── peft_config.py     # LoraConfig builder + apply_lora()
│   │   └── utils.py           # Param counting, adapter merging
│   ├── training/
│   │   ├── trainer.py         # SFTTrainer setup
│   │   ├── callbacks.py       # MetricsLogger + EarlyStopping
│   │   └── metrics.py         # Perplexity, token accuracy
│   ├── evaluation/
│   │   ├── metrics.py         # BLEU, ROUGE-L, perplexity
│   │   └── inference.py       # GenerationPipeline wrapper
│   └── utils/
│       ├── seed.py            # Reproducible training
│       └── logging.py         # Loguru setup
│
├── tests/                     # Unit tests (>70% coverage)
├── configs/                   # YAML configs for models, training, data
├── scripts/                   # CLI entry points
├── notebooks/                 # EDA, token analysis, results
└── .github/workflows/         # CI: tests, lint, evaluation smoke test
```

---

## Models (Progressive Complexity)

| # | Model | Method | Min VRAM | Use Case |
|---|---|---|---|---|
| 1 | [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | LoRA (fp16) | 4 GB | Quick iteration, proof-of-concept |
| 2 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | LoRA (fp16) | 6 GB | Multilingual (English + Bahasa Indonesia) |
| 3 | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | QLoRA (4-bit) | 8 GB | Production-scale, best quality |

---

## Dataset

Synthetic instruction-following dataset (10,000 samples) generated programmatically across 5 domains:

| Domain | Ratio | Description |
|---|---|---|
| `general_qa` | 25% | General knowledge Q&A (ML concepts, AI fundamentals) |
| `summarization` | 20% | Summarize a technical passage |
| `classification` | 20% | Classify sentiment, type, or topic |
| `translation` | 20% | English → Bahasa Indonesia |
| `code_explanation` | 15% | Explain a Python code snippet |

Format: ChatML messages (`{"messages": [{"role": ..., "content": ...}]}`)

---

## Setup

### Requirements

- Python 3.10 or 3.11
- PyTorch 2.3+
- For QLoRA (Llama 3.1 8B): NVIDIA GPU with 16+ GB VRAM

### Installation

```bash
# Clone the repository
git clone https://github.com/achyar-cn/llm-fine-tuning.git
cd llm-fine-tuning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package and dev dependencies
pip install -e ".[dev]"
```

---

## Usage

### 1. Generate Synthetic Dataset

```bash
python scripts/generate_data.py --config configs/data/synthetic_config.yaml
# Output: data/raw/train.jsonl (10,000 samples)
```

### 2. Fine-Tune a Model

```bash
# Stage 1: TinyLlama 1.1B (fastest, CPU or any GPU)
python scripts/train.py \
    --model configs/model/tinyllama_1b.yaml \
    --training configs/training/sft_lora.yaml \
    --data-path data/raw/train.jsonl

# Stage 2: Qwen2.5 1.5B (multilingual)
python scripts/train.py \
    --model configs/model/qwen25_15b.yaml \
    --training configs/training/sft_lora.yaml \
    --data-path data/raw/train.jsonl

# Stage 3: Llama 3.1 8B + QLoRA (production-scale, requires 16GB+ GPU)
python scripts/train.py \
    --model configs/model/llama31_8b_qlora.yaml \
    --training configs/training/sft_qlora.yaml \
    --data-path data/raw/train.jsonl
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --adapter outputs/final_adapter \
    --data data/raw/test.jsonl \
    --output outputs/eval_results.json
```

### 4. Interactive Inference

```bash
python scripts/inference.py --model outputs/final_adapter
# Or with a single prompt:
python scripts/inference.py --model outputs/final_adapter --prompt "Jelaskan apa itu machine learning"
```

### 5. Run Tests

```bash
make test
# Or: pytest tests/ -v --cov=src/
```

---

## LoRA Configuration

| Parameter | TinyLlama / Qwen2.5 | Llama 3.1 8B (QLoRA) |
|---|---|---|
| Rank (`r`) | 16 | 64 |
| Alpha | 32 | 128 |
| Target modules | 7 linear layers | 7 linear layers |
| Dropout | 0.05 | 0.05 |
| Trainable params | ~41M (3.7%) | ~168M (2.0%) |

**Rule of thumb:** `alpha = 2 × r`. Higher rank → more capacity, more memory.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Scheduler | Cosine with warmup |
| Warmup ratio | 3% |
| Effective batch size | 16 (4 × 4 grad accum) |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

---

## Evaluation Results

*(Fill in after training each model)*

| Model | Train Loss | Perplexity | BLEU | ROUGE-L | Training Time |
|---|---|---|---|---|---|
| TinyLlama-1.1B | — | — | — | — | — |
| Qwen2.5-1.5B | — | — | — | — | — |
| Llama3.1-8B QLoRA | — | — | — | — | — |

See [notebooks/03_results.ipynb](notebooks/03_results.ipynb) for detailed analysis.

---

## CI/CD

GitHub Actions runs automatically on every push and pull request:

| Workflow | Trigger | What it does |
|---|---|---|
| `tests.yml` | push / PR | pytest + coverage (target >70%) |
| `lint.yml` | push / PR | black, isort, flake8 checks |
| `evaluate.yml` | PR to main | Evaluation pipeline smoke test |

---

## Project Structure Rationale

- **`src/` layout**: Prevents import path issues; package is installable via `pip install -e .`
- **YAML configs**: Separates code from hyperparameters — swap models without touching Python files
- **Synthetic data**: Demonstrates data engineering skills; no copyright or privacy concerns
- **Progressive models**: Shows scalability from 1.1B → 1.5B → 8B with same codebase
- **Label masking**: Only the assistant's response contributes to the loss (standard SFT practice)
- **`sshleifer/tiny-gpt2` in CI**: Tests run in <2 minutes without GPU; real models tested locally

---

## License

MIT License. See [LICENSE](LICENSE).
