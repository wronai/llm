# WronAI Development Makefile
# Simplified commands for development workflow

.PHONY: help venv venv-activate install install-dev install-core install-ml install-nlp install-utils clean test lint format docker-build docker-run prepare-data train inference

# Default target
help:
	@echo "🐦‍⬛ WronAI Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  venv            Create Python virtual environment"
	@echo "  install          Install all dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-core     Install core dependencies only"
	@echo "  install-ml       Install ML-related dependencies"
	@echo "  install-nlp      Install NLP-related dependencies"
	@echo "  install-utils    Install utility dependencies"
	@echo "  clean           Clean build artifacts and cache"
	@echo ""
	@echo "Development Commands:"
	@echo "  format          Format code with black and isort"
	@echo "  lint            Run linting with flake8 and mypy"
	@echo "  test            Run test suite"
	@echo "  test-cov        Run tests with coverage report"
	@echo ""
	@echo "Data and Training:"
	@echo "  prepare-data    Download and prepare training data"
	@echo "  train           Start model training"
	@echo "  train-quick     Quick training with minimal data"
	@echo "  inference       Run inference on trained model"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build    Build Docker container"
	@echo "  docker-run      Run training in Docker"
	@echo "  docker-serve    Serve model in Docker"
	@echo ""
	@echo "Utility Commands:"
	@echo "  docs            Generate documentation"
	@echo "  notebook        Start Jupyter notebook server"
	@echo "  tensorboard     Start TensorBoard"

# Virtual environment commands
venv:
	python -m venv wronai-env
	@echo "Virtual environment created. Activate with:"
	@echo "  source wronai-env/bin/activate  # Linux/Mac"
	@echo "  wronai-env\Scripts\activate    # Windows"

# Installation commands
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,docs,inference]"
	pre-commit install

install-core:
	pip install torch transformers accelerate peft datasets evaluate

install-ml:
	pip install bitsandbytes scipy safetensors wandb tensorboard

install-nlp:
	pip install tokenizers sentencepiece regex spacy

install-utils:
	pip install beautifulsoup4 requests aiohttp scrapy pyyaml omegaconf loguru rich

# Clean commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Code quality commands
format:
	black scripts/ wronai/ tests/ || echo "Warning: black formatter had issues"
	isort scripts/ wronai/ tests/ || echo "Warning: isort had issues"

lint:
	flake8 .
	mypy scripts/ wronai/ --ignore-missing-imports
	black --check .
	isort --check-only .

# Testing commands
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=wronai --cov-report=html --cov-report=term

test-fast:
	pytest tests/unit/ -v

# Data preparation
prepare-data:
	python scripts/prepare_data.py --all --output-dir data/processed

prepare-data-minimal:
	python scripts/prepare_data.py --create-instructions --output-dir data/processed

# Training commands
train:
	python scripts/train.py --config configs/default.yaml

train-quick:
	python scripts/train.py --config configs/quick_test.yaml

train-gpu:
	CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/default.yaml

# Inference commands
inference:
	python scripts/inference.py --model checkpoints/wronai-7b --chat

inference-prompt:
	python scripts/inference.py --model checkpoints/wronai-7b --prompt "Opowiedz o Polsce"

# Docker commands
docker-build:
	docker build -t wronai:latest .

docker-run:
	docker-compose up wronai-training

docker-serve:
	docker-compose up wronai-inference

docker-prep:
	docker-compose up wronai-data-prep

docker-down:
	docker-compose down

# Development utilities
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

notebook:
	jupyter notebook notebooks/

tensorboard:
	tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Model evaluation
evaluate:
	python scripts/evaluate.py --model checkpoints/wronai-7b --benchmarks all

benchmark:
	python scripts/benchmark.py --model checkpoints/wronai-7b --output results/

# Release commands
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Environment setup
setup-env:
	python -m venv venv
	@echo "Run: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

setup-cuda:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Quick start for new developers
quickstart: install-dev prepare-data-minimal train-quick
	@echo "🎉 WronAI quickstart completed!"
	@echo "Try: make inference"

# CI/CD simulation
ci: clean lint test
	@echo "✅ CI checks passed!"

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/train.py --config configs/profile.yaml
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	mprof run python scripts/train.py --config configs/memory_test.yaml
	mprof plot

# Model size analysis
model-size:
	python -c "import torch; from transformers import AutoModel; model = AutoModel.from_pretrained('checkpoints/wronai-7b'); total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f'Total parameters: {total_params:,}'); print(f'Trainable parameters: {trainable_params:,}'); print(f'Model size: {total_params * 4 / 1024**3:.2f} GB (fp32)');"

# Download pre-trained models
download-models:
	mkdir -p models/pretrained
	wget -O models/pretrained/wronai-7b.tar.gz "https://softreck.dev/wronai-7b.tar.gz"
	tar -xzf models/pretrained/wronai-7b.tar.gz -C models/pretrained/

# Health check
health-check:
	python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); \\\n\tif torch.cuda.is_available(): \\\n\t    print(f'CUDA version: {torch.version.cuda}'); \\\n\t    print(f'GPU count: {torch.cuda.device_count()}'); \\\n\t    for i in range(torch.cuda.device_count()): \\\n\t        props = torch.cuda.get_device_properties(i); \\\n\t        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)');"

# Package information
info:
	@echo "🐦‍⬛ WronAI Project Information"
	@echo "Version: $(shell python setup.py --version)"
	@echo "Author: $(shell python setup.py --author)"
	@echo "License: $(shell python setup.py --license)"
	@echo ""
	@echo "Dependencies:"
	@pip list | grep -E "(torch|transformers|datasets|accelerate)"

# Security scan
security:
	safety check
	bandit -r wronai/ scripts/

# Update dependencies
update-deps:
	pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U

# Backup important files
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		configs/ scripts/ wronai/ checkpoints/ data/processed/ \
		--exclude="*.pyc" --exclude="__pycache__" --exclude="*.log"