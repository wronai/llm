#!/bin/bash
# WronAI Data Preparation Script

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
  source ../venv/bin/activate
elif [ -d "../wronai-env" ]; then
  source ../wronai-env/bin/activate
fi

# Run data preparation script
echo "Starting data preparation..."
python ../scripts/prepare_data.py

# Exit status
exit $?
