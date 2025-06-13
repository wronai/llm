#!/bin/bash
# WronAI TensorBoard Script
# Usage: ./tensorboard.sh <logs_dir>

LOGS_DIR=${1:-"../checkpoints"}

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
  source ../venv/bin/activate
elif [ -d "../wronai-env" ]; then
  source ../wronai-env/bin/activate
fi

# Start TensorBoard
echo "Starting TensorBoard with logdir: $LOGS_DIR"
tensorboard --logdir=$LOGS_DIR

# Exit status
exit $?
