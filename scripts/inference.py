"""CLI script: interactive chat with a fine-tuned model.

Usage:
    python scripts/inference.py --model outputs/final_adapter
    python scripts/inference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "What is AI?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from llm_finetuning.config import ModelConfig
from llm_finetuning.data.templates import SYSTEM_PROMPT_DEFAULT
from llm_finetuning.evaluation.inference import GenerationConfig, GenerationPipeline, build_prompt_from_messages
from llm_finetuning.models.loader import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned LLM")
    parser.add_argument("--model", type=str, default="outputs/final_adapter",
                        help="Path to model or HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModelConfig(
        model_name_or_path=args.model,
        model_family="llama",
        torch_dtype="float16",
        device_map="auto",
    )
    logger.info(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    pipe = GenerationPipeline(model, tokenizer, gen_cfg)

    def respond(user_input: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
            {"role": "user", "content": user_input},
        ]
        prompt = build_prompt_from_messages(messages, tokenizer)
        return pipe.generate_one(prompt)

    # Single prompt mode
    if args.prompt:
        response = respond(args.prompt)
        print(f"\nAssistant: {response}\n")
        return

    # Interactive chat loop
    print("Fine-tuned LLM Chat — type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        response = respond(user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
