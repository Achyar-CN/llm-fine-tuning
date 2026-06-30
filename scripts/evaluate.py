"""CLI script: evaluate a fine-tuned model on a test set.

Usage:
    python scripts/evaluate.py \\
        --adapter outputs/final_adapter \\
        --data data/raw/test.jsonl \\
        --output outputs/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from llm_finetuning.config import ModelConfig
from llm_finetuning.data.loader import load_jsonl
from llm_finetuning.evaluation.inference import GenerationConfig, GenerationPipeline, build_prompt_from_messages
from llm_finetuning.evaluation.metrics import evaluate
from llm_finetuning.models.loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLM")
    parser.add_argument("--adapter", type=str, required=True, help="Path to merged model adapter")
    parser.add_argument("--data", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit evaluation to N samples")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model from adapter directory
    cfg = ModelConfig(
        model_name_or_path=args.adapter,
        model_family="llama",
        torch_dtype="float16",
        device_map="auto",
    )
    logger.info(f"Loading model from {args.adapter}...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens)
    pipeline = GenerationPipeline(model, tokenizer, gen_cfg)

    # Load test data
    logger.info(f"Loading test data from {args.data}...")
    samples = load_jsonl(args.data)
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info(f"Evaluating on {len(samples)} samples...")

    predictions, references = [], []
    for sample in samples:
        messages = sample["messages"]
        # Build prompt from all messages except the last assistant turn
        prompt_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        prompt = build_prompt_from_messages(prompt_messages, tokenizer)
        pred = pipeline.generate_one(prompt)
        ref = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
        predictions.append(pred.strip())
        references.append(ref.strip())

    result = evaluate(predictions, references)
    result_dict = result.to_dict()

    logger.info(f"BLEU:      {result.bleu:.2f}")
    logger.info(f"ROUGE-1:   {result.rouge1:.4f}")
    logger.info(f"ROUGE-2:   {result.rouge2:.4f}")
    logger.info(f"ROUGE-L:   {result.rougeL:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
