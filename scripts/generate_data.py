"""CLI script: generate synthetic instruction-following dataset.

Usage:
    python scripts/generate_data.py --config configs/data/synthetic_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from llm_finetuning.config import data_config_from_yaml
from llm_finetuning.data.synthetic_generator import SyntheticGeneratorConfig, generate_dataset, save_jsonl
from llm_finetuning.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning dataset")
    parser.add_argument("--config", type=str, default="configs/data/synthetic_config.yaml",
                        help="Path to data config YAML")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSONL path from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = data_config_from_yaml(args.config)
    set_seed(data_cfg.seed)

    gen_cfg = SyntheticGeneratorConfig(
        num_samples=data_cfg.num_samples,
        seed=data_cfg.seed,
        output_path=args.output or data_cfg.output_path,
        domain_weights=data_cfg.domain_weights,
    )

    logger.info(f"Generating {gen_cfg.num_samples} samples with seed={gen_cfg.seed}...")
    samples = generate_dataset(gen_cfg)

    output_path = Path(gen_cfg.output_path)
    save_jsonl(samples, output_path)
    logger.info(f"Saved {len(samples)} samples to {output_path}")

    # Show domain distribution
    from collections import Counter
    dist = Counter(s["domain"] for s in samples)
    for domain, count in sorted(dist.items()):
        logger.info(f"  {domain}: {count} ({100 * count / len(samples):.1f}%)")


if __name__ == "__main__":
    main()
