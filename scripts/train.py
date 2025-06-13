#!/usr/bin/env python3
"""
WronAI Training Script
Polish Language Model Training with QLoRA optimization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WronAITrainer:
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.setup_logging()

    def load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Setup logging and monitoring."""
        if "wandb" in self.config["logging"]["report_to"]:
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                name=self.config["logging"]["wandb_run_name"],
                config=self.config,
            )

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer for Polish language."""
        logger.info(f"Loading tokenizer: {self.config['model']['name']}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
        )

        # Add Polish-specific tokens if configured
        if self.config["polish"]["add_polish_tokens"]:
            polish_tokens = [
                "<polish>",
                "</polish>",
                "<formal>",
                "</formal>",
                "<informal>",
                "</informal>",
                "<question>",
                "</question>",
                "<answer>",
                "</answer>",
            ]
            tokenizer.add_tokens(polish_tokens)

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model(self, tokenizer: AutoTokenizer):
        """Load model with quantization and LoRA configuration."""
        logger.info(f"Loading model: {self.config['model']['name']}")

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

        # Resize embeddings if new tokens were added
        if self.config["polish"]["add_polish_tokens"]:
            model.resize_token_embeddings(len(tokenizer))

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

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

        # Print trainable parameters
        model.print_trainable_parameters()

        return model

    def load_dataset(self, tokenizer: AutoTokenizer):
        """Load and preprocess training dataset."""
        logger.info(f"Loading dataset: {self.config['data']['dataset_name']}")

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

        # Tokenization function
        def tokenize_function(examples):
            # Combine instruction and response for causal LM
            texts = []
            for instruction, response in zip(
                examples["instruction"], examples["response"]
            ):
                text = f"<polish><question>{instruction}</question><answer>{response}</answer></polish>"
                texts.append(text)

            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config["data"]["max_seq_length"],
                return_overflowing_tokens=False,
            )

        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config["data"]["preprocessing_num_workers"],
            remove_columns=train_dataset.column_names,
        )

        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config["data"]["preprocessing_num_workers"],
            remove_columns=eval_dataset.column_names,
        )

        return train_dataset, eval_dataset

    def train(self):
        """Main training function."""
        logger.info("Starting WronAI training...")

        # Load components
        tokenizer = self.load_tokenizer()
        model = self.load_model(tokenizer)
        train_dataset, eval_dataset = self.load_dataset(tokenizer)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"][
                "per_device_train_batch_size"
            ],
            per_device_eval_batch_size=self.config["training"][
                "per_device_eval_batch_size"
            ],
            gradient_accumulation_steps=self.config["training"][
                "gradient_accumulation_steps"
            ],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            learning_rate=self.config["training"]["learning_rate"],
            fp16=self.config["training"]["fp16"],
            bf16=self.config["training"]["bf16"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            evaluation_strategy=self.config["training"]["evaluation_strategy"],
            save_strategy=self.config["training"]["save_strategy"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training"]["metric_for_best_model"],
            greater_is_better=self.config["training"]["greater_is_better"],
            save_total_limit=self.config["training"]["save_total_limit"],
            remove_unused_columns=self.config["training"]["remove_unused_columns"],
            dataloader_pin_memory=self.config["training"]["dataloader_pin_memory"],
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            group_by_length=self.config["training"]["group_by_length"],
            optim=self.config["training"]["optim"],
            lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            weight_decay=self.config["training"]["weight_decay"],
            report_to=self.config["logging"]["report_to"],
            run_name=self.config["logging"]["wandb_run_name"],
        )

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

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config["training"]["output_dir"])

        logger.info("Training completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="WronAI Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Initialize and start training
    trainer = WronAITrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
