#!/bin/bash
# WronAI Multi-GPU Training Script
# Usage: ./train_multi_gpu.sh <config_file>

# Check if config file is provided
if [ -z "$1" ]; then
  echo "Error: No configuration file provided."
  echo "Usage: ./train_multi_gpu.sh <config_file>"
  exit 1
fi

CONFIG_FILE=$1

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
  source ../venv/bin/activate
elif [ -d "../wronai-env" ]; then
  source ../wronai-env/bin/activate
fi

# Run training script with multi-GPU support
echo "Starting multi-GPU training with config: $CONFIG_FILE"
python ../scripts/train.py --config $CONFIG_FILE --multi_gpu

# Exit status
exit $?
