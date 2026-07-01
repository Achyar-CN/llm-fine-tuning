"""GenerationPipeline — wrapper for text generation used in evaluation."""

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase, pipeline


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


class GenerationPipeline:
    """Wraps a HuggingFace pipeline for consistent batch text generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: GenerationConfig | None = None,
    ) -> None:
        self.cfg = cfg or GenerationConfig()
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a list of prompt strings.

        Returns a list of generated strings (without the prompt prefix).
        """
        outputs = self._pipe(
            prompts,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            repetition_penalty=self.cfg.repetition_penalty,
            return_full_text=False,
        )
        return [_extract_text(out) for out in outputs]

    def generate_one(self, prompt: str) -> str:
        """Generate a response for a single prompt string."""
        return self.generate([prompt])[0]


def _extract_text(output: Any) -> str:
    """Extract the generated text string from pipeline output."""
    if isinstance(output, list) and output:
        return output[0].get("generated_text", "")
    if isinstance(output, dict):
        return output.get("generated_text", "")
    return str(output)


def build_prompt_from_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Build a prompt string using the tokenizer's chat template.

    Falls back to a simple 'role: content' format if the tokenizer
    has no chat_template (e.g. base GPT-2 variants).
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = [f"{msg['role']}: {msg['content']}" for msg in messages]
    parts.append("assistant:")
    return "\n".join(parts)
