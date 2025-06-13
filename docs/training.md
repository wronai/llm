# WronAI Training Guide

This document provides a comprehensive guide to training the WronAI Polish language model using QLoRA fine-tuning and quantization techniques. The training process is designed to be efficient and run on consumer-grade hardware with as little as 8GB of VRAM.

## Table of Contents

- [Overview](#overview)
- [Training Pipeline Visualization](#training-pipeline-visualization)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Step-by-Step Training Process](#step-by-step-training-process)
- [Hardware Optimization](#hardware-optimization)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Overview

WronAI training leverages state-of-the-art techniques to fine-tune large language models for Polish language understanding and generation. The process uses:

- **QLoRA**: Quantized Low-Rank Adaptation for efficient fine-tuning
- **4-bit Quantization**: Memory-efficient model loading
- **Gradient Checkpointing**: Reduced memory footprint during training
- **Polish-specific Tokenization**: Enhanced token representation for Polish language

The primary training script is [`scripts/train.py`](../scripts/train.py) which uses configuration from [`configs/default.yaml`](../configs/default.yaml).

## Training Pipeline Visualization

### ASCII Diagram

```
+--------------------------------------------------------------------------------------------------+
|                                    WronAI LLM Training Pipeline                                   |
+--------------------------------------------------------------------------------------------------+

  +-------------+     +--------------+     +----------------+     +--------------------+
  | Load Config |---->| Setup Logging|---->| Load Tokenizer |---->| Add Polish Tokens  |
  +-------------+     +--------------+     +----------------+     +--------------------+
         |                                                                 |
         v                                                                 v
  +-----------------+                                            +--------------------+
  | Load Base Model |<-------------------------------------------| Configure Model    |
  | (Mistral-7B)    |                                            | Parameters         |
  +-----------------+                                            +--------------------+
         |
         v
  +------------------+     +----------------+     +-------------------+
  | Apply            |---->| Configure LoRA |---->| Prepare for k-bit |
  | Quantization     |     | Parameters     |     | Training          |
  | (4-bit NF4)      |     | (r=16, α=32)   |     |                   |
  +------------------+     +----------------+     +-------------------+
         |
         v
  +------------------+     +----------------+     +-------------------+
  | Load Dataset     |---->| Tokenize &     |---->| Create Data       |
  | (polish-instruct)|     | Preprocess     |     | Collator          |
  +------------------+     | Data           |     |                   |
  +------------------+     +----------------+     +-------------------+
         |
         v
  +------------------+     +----------------+     +-------------------+
  | Configure        |---->| Initialize     |---->| Execute Training  |
  | Training Args    |     | Trainer        |     | Loop              |
  | (epochs, lr, etc)|     |                |     | (with eval steps) |
  +------------------+     +----------------+     +-------------------+
         |
         v
  +------------------+     +----------------+
  | Save Final       |---->| Save Tokenizer |
  | Model            |     |                |
  +------------------+     +----------------+

  Hardware Optimization:
  - Gradient Checkpointing: Enabled
  - CPU Offload: Enabled
  - Memory Limit: 7000MB (for 8GB GPU)
  - Batch Size: 1 with 16 gradient accumulation steps
```

### Mermaid Diagram

```mermaid
flowchart TD
    subgraph "Configuration"
        A[Load Config from YAML] --> B[Setup Logging & WandB]
    end
    
    subgraph "Model Preparation"
        C[Load Tokenizer] --> D[Add Polish-specific Tokens]
        D --> E[Load Base Model: Mistral-7B]
        E --> F[Apply 4-bit Quantization]
        F --> G[Configure LoRA Parameters]
        G --> H[Prepare for k-bit Training]
    end
    
    subgraph "Data Processing"
        I[Load Dataset: polish-instruct] --> J[Split Train/Validation]
        J --> K[Tokenize & Format Data]
        K --> L[Create DataCollator]
    end
    
    subgraph "Training Setup"
        M[Configure Training Arguments] --> N[Initialize Trainer]
    end
    
    subgraph "Training Loop"
        O[Execute Training] --> P[Periodic Evaluation]
        P --> Q[Save Checkpoints]
        Q --> O
    end
    
    subgraph "Finalization"
        R[Save Final Model] --> S[Save Tokenizer]
    end
    
    B --> C
    H --> I
    L --> M
    N --> O
    Q --> R
    
    style Configuration fill:#f9f,stroke:#333,stroke-width:2px
    style "Model Preparation" fill:#bbf,stroke:#333,stroke-width:2px
    style "Data Processing" fill:#bfb,stroke:#333,stroke-width:2px
    style "Training Setup" fill:#fbf,stroke:#333,stroke-width:2px
    style "Training Loop" fill:#fbb,stroke:#333,stroke-width:2px
    style "Finalization" fill:#bff,stroke:#333,stroke-width:2px
```

## Prerequisites

- Python 3.8+ environment with dependencies installed (see [Installation Guide](installation.md))
- GPU with at least 8GB VRAM (recommended: 16GB+)
- Prepared dataset (see [Data Preparation Guide](../scripts/prepare_data.py))
- CUDA 11.8+ with appropriate drivers

## Configuration

The training process is controlled by a YAML configuration file. The default configuration is located at [`configs/default.yaml`](../configs/default.yaml).

Key configuration sections include:

### Model Configuration

```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"  # Base model to fine-tune
  trust_remote_code: true             # Allow execution of remote code
  torch_dtype: "bfloat16"            # Precision for computation
  device_map: "auto"                 # Device allocation strategy
```

### LoRA Configuration

```yaml
lora:
  r: 16                               # Rank of LoRA update matrices
  lora_alpha: 32                      # LoRA scaling factor
  lora_dropout: 0.1                   # Dropout probability for LoRA layers
  bias: "none"                       # Bias configuration
  task_type: "CAUSAL_LM"             # Task type for LoRA
  target_modules:                     # Modules to apply LoRA to
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### Quantization Configuration

```yaml
quantization:
  load_in_4bit: true                  # Enable 4-bit quantization
  bnb_4bit_compute_dtype: "bfloat16"  # Compute dtype for 4-bit quantization
  bnb_4bit_quant_type: "nf4"         # Quantization type (NF4)
  bnb_4bit_use_double_quant: true     # Enable double quantization
```

See the [Advanced Configuration](#advanced-configuration) section for more details on all available parameters.

## Step-by-Step Training Process

### 1. Configuration Loading

The training process begins by loading the configuration from the specified YAML file:

```python
# From scripts/train.py
def load_config(self, config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
```

### 2. Tokenizer Preparation

The tokenizer is loaded and enhanced with Polish-specific tokens:

```python
# From scripts/train.py
def load_tokenizer(self) -> AutoTokenizer:
    """Load and configure tokenizer for Polish language."""
    tokenizer = AutoTokenizer.from_pretrained(
        self.config["model"]["name"],
        trust_remote_code=self.config["model"]["trust_remote_code"],
    )

    # Add Polish-specific tokens if configured
    if self.config["polish"]["add_polish_tokens"]:
        polish_tokens = [
            "<polish>", "</polish>",
            "<formal>", "</formal>",
            "<informal>", "</informal>",
            "<question>", "</question>",
            "<answer>", "</answer>",
        ]
        tokenizer.add_tokens(polish_tokens)
```

### 3. Model Loading and Quantization

The base model is loaded with 4-bit quantization to reduce memory usage:

```python
# From scripts/train.py
def load_model(self, tokenizer: AutoTokenizer):
    """Load model with quantization and LoRA configuration."""
    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=self.config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(
            torch, self.config["quantization"]["bnb_4bit_compute_dtype"]
        ),
        bnb_4bit_quant_type=self.config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=self.config["quantization"][
            "bnb_4bit_use_double_quant"
        ],
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        self.config["model"]["name"],
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, self.config["model"]["torch_dtype"]),
        device_map=self.config["model"]["device_map"],
        trust_remote_code=self.config["model"]["trust_remote_code"],
    )
```

### 4. LoRA Configuration

Low-Rank Adaptation is applied to make fine-tuning efficient:

```python
# From scripts/train.py
# LoRA configuration
lora_config = LoraConfig(
    r=self.config["lora"]["r"],
    lora_alpha=self.config["lora"]["lora_alpha"],
    lora_dropout=self.config["lora"]["lora_dropout"],
    bias=self.config["lora"]["bias"],
    task_type=self.config["lora"]["task_type"],
    target_modules=self.config["lora"]["target_modules"],
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

### 5. Dataset Loading and Processing

The training dataset is loaded and preprocessed:

```python
# From scripts/train.py
def load_dataset(self, tokenizer: AutoTokenizer):
    """Load and preprocess training dataset."""
    # Load dataset
    dataset = load_dataset(
        self.config["data"]["dataset_name"],
        split=self.config["data"]["train_split"],
    )

    # Split into train/eval if needed
    if self.config["data"]["eval_split"] not in dataset:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = load_dataset(
            self.config["data"]["dataset_name"],
            split=self.config["data"]["eval_split"],
        )
```

### 6. Training Execution

The training process is executed using the Hugging Face Trainer:

```python
# From scripts/train.py
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
logger.info("Training started!")
trainer.train()
```

### 7. Model Saving

The final model and tokenizer are saved:

```python
# From scripts/train.py
# Save final model
logger.info("Saving final model...")
trainer.save_model()
tokenizer.save_pretrained(self.config["training"]["output_dir"])
```

## Hardware Optimization

WronAI training is optimized for consumer hardware:

- **4-bit Quantization**: Reduces memory usage by loading model in 4-bit precision
- **Gradient Checkpointing**: Trades computation for memory by recomputing activations
- **Small Batch Size**: Uses batch size of 1 with gradient accumulation of 16 steps
- **CPU Offloading**: Moves some model parameters to CPU when not in use
- **Memory Limit**: Configurable memory limit (default: 7000MB for 8GB GPUs)

These optimizations are configured in the [`configs/default.yaml`](../configs/default.yaml) file.

## Monitoring and Evaluation

Training progress can be monitored using:

- **Weights & Biases**: Real-time metrics and visualizations
- **TensorBoard**: Local visualization of training metrics
- **Logging**: Console output with key information

Evaluation is performed periodically during training using:

- **Loss**: Primary metric for model quality
- **Perplexity**: Measure of language model quality
- **Polish-specific Metrics**: Custom metrics for Polish language evaluation

See the [Benchmarks Guide](benchmarks.md) for more information on evaluation.

## Troubleshooting

Common training issues and solutions:

- **Out of Memory Errors**: Reduce batch size, enable gradient checkpointing, or use more aggressive quantization
- **Slow Training**: Check for CPU bottlenecks, optimize data loading, or use a more powerful GPU
- **Poor Convergence**: Adjust learning rate, increase training epochs, or check data quality

See the [Troubleshooting Guide](troubleshooting.md) for more detailed solutions.

## Advanced Configuration

For advanced users, additional configuration options are available:

- **Custom Tokenizers**: Use a Polish-specific tokenizer for better performance
- **Multi-GPU Training**: Distribute training across multiple GPUs
- **Mixed Precision**: Configure precision for different operations
- **Custom Datasets**: Use your own datasets for fine-tuning

See the [Advanced Training Tutorial](tutorials/02_custom_training.md) for more information.

## Running the Training

To start training with the default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

For quick testing with minimal resources:

```bash
python scripts/train.py --config configs/quick_test.yaml
```

Or use the Makefile for convenience:

```bash
make train
```

## Next Steps

- [Inference Guide](inference.md): How to use your trained model
- [Model Evaluation](benchmarks.md): Evaluate your model's performance
- [Advanced Features](tutorials/04_advanced_features.md): Explore advanced WronAI features