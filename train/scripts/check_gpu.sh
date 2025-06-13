#!/bin/bash
# WronAI GPU Check Script

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
  source ../venv/bin/activate
elif [ -d "../wronai-env" ]; then
  source ../wronai-env/bin/activate
fi

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('No GPU available, training will be slow on CPU only.')
"

# Exit status
exit $?
