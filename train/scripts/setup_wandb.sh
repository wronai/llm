#!/bin/bash
# WronAI Weights & Biases Setup Script

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
  source ../venv/bin/activate
elif [ -d "../wronai-env" ]; then
  source ../wronai-env/bin/activate
fi

# Setup Weights & Biases
echo "Setting up Weights & Biases..."
pip install wandb
wandb login

# Exit status
exit $?
