.PHONY: install test lint format generate-data train evaluate clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/ --cov-report=term-missing

lint:
	flake8 src/ tests/
	isort --check-only src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

generate-data:
	python scripts/generate_data.py --config configs/data/synthetic_config.yaml

train:
	python scripts/train.py \
		--model configs/model/tinyllama_1b.yaml \
		--training configs/training/sft_lora.yaml \
		--data configs/data/synthetic_config.yaml

evaluate:
	python scripts/evaluate.py \
		--adapter outputs/final_adapter \
		--data data/raw/test.jsonl

inference:
	python scripts/inference.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml
