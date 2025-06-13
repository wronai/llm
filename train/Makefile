# WronAI Training Makefile
# Dedicated Makefile for model training processes

# Default shell
SHELL := /bin/bash

# Directories
SCRIPTS_DIR := ../scripts
CONFIGS_DIR := ../configs
DATA_DIR := ../data
CHECKPOINTS_DIR := ../checkpoints
LOGS_DIR := ../logs

# Create directories if they don't exist
$(CHECKPOINTS_DIR):
	mkdir -p $(CHECKPOINTS_DIR)

$(LOGS_DIR):
	mkdir -p $(LOGS_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Training configurations
DEFAULT_CONFIG := $(CONFIGS_DIR)/default.yaml
QUICK_TEST_CONFIG := $(CONFIGS_DIR)/quick_test.yaml

# Default target
.PHONY: help
help:
	@echo "WronAI Training Makefile"
	@echo "========================="
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  prepare-data      - Prepare training data"
	@echo "  train             - Train model with default configuration"
	@echo "  train-quick       - Train model with quick test configuration"
	@echo "  train-custom      - Train model with custom configuration (CONFIG=path/to/config.yaml)"
	@echo "  tensorboard       - Start TensorBoard server"
	@echo "  clean-checkpoints - Remove all checkpoints"
	@echo "  clean-logs        - Remove all logs"
	@echo "  clean-all         - Remove all generated files (checkpoints, logs)"
	@echo "  setup-wandb       - Setup Weights & Biases for experiment tracking"
	@echo "  setup-wandb-local - Setup local Weights & Biases server"
	@echo "  check-gpu         - Check GPU availability"

# Data preparation
.PHONY: prepare-data
prepare-data: $(DATA_DIR)
	./scripts/prepare_data.sh

# Training targets
.PHONY: train
train: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	./scripts/train.sh $(DEFAULT_CONFIG)

.PHONY: train-quick
train-quick: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	./scripts/train.sh $(QUICK_TEST_CONFIG)

.PHONY: train-custom
train-custom: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG variable is not set. Usage: make train-custom CONFIG=path/to/config.yaml"; \
		exit 1; \
	fi
	./scripts/train.sh $(CONFIG)

# TensorBoard
.PHONY: tensorboard
tensorboard:
	./scripts/tensorboard.sh $(CHECKPOINTS_DIR)

# Weights & Biases setup
.PHONY: setup-wandb
setup-wandb:
	./scripts/setup_wandb.sh

# Setup local Weights & Biases server
.PHONY: setup-wandb-local
setup-wandb-local:
	./scripts/setup_wandb_local.sh

# Cleaning targets
.PHONY: clean-checkpoints
clean-checkpoints:
	rm -rf $(CHECKPOINTS_DIR)/*

.PHONY: clean-logs
clean-logs:
	rm -rf $(LOGS_DIR)/*

.PHONY: clean-all
clean-all: clean-checkpoints clean-logs
	@echo "All generated files removed."

# Hardware optimization targets
.PHONY: check-gpu
check-gpu:
	./scripts/check_gpu.sh

# Training with specific hardware configurations
.PHONY: train-cpu-only
train-cpu-only: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	CUDA_VISIBLE_DEVICES="" ./scripts/train.sh $(DEFAULT_CONFIG)

.PHONY: train-specific-gpu
train-specific-gpu: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	@if [ -z "$(GPU)" ]; then \
		echo "Error: GPU variable is not set. Usage: make train-specific-gpu GPU=0"; \
		exit 1; \
	fi
	CUDA_VISIBLE_DEVICES=$(GPU) ./scripts/train.sh $(DEFAULT_CONFIG)

# Multi-GPU training (if supported by the script)
.PHONY: train-multi-gpu
train-multi-gpu: $(CHECKPOINTS_DIR) $(LOGS_DIR)
	./scripts/train_multi_gpu.sh $(DEFAULT_CONFIG)
