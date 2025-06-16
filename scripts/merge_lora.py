#!/usr/bin/env python
"""
Merge LoRA adapter with base model to create a standalone model for inference.

This script addresses the vocabulary size mismatch issue by creating a full model
rather than applying adapters at runtime.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_base_model_name(adapter_path):
    """Extract base model name from adapter config."""
    try:
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            logger.error(f"No adapter_config.json found at {adapter_path}")
            return None
            
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            logger.error("No base_model_name_or_path found in adapter_config.json")
            return None
            
        logger.info(f"LoRA was trained on base model: {base_model_name}")
        return base_model_name
    except Exception as e:
        logger.error(f"Error reading adapter_config.json: {e}")
        return None


def load_tokenizer(adapter_path, base_model):
    """Load tokenizer from adapter path or base model."""
    try:
        # Try to load tokenizer from adapter path first
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
        logger.info("Loaded tokenizer from adapter directory")
        return tokenizer
    except Exception as e:
        logger.warning(f"Could not load tokenizer from adapter directory: {e}")
        
        try:
            # Fallback to base model tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
            logger.info(f"Loaded tokenizer from base model: {base_model}")
            return tokenizer
        except Exception as e2:
            logger.error(f"Could not load tokenizer from base model: {e2}")
            return None


def resize_tokenizer_embeddings(model, adapter_tokenizer):
    """Resize model embeddings to match adapter tokenizer size."""
    try:
        logger.info(f"Original model vocab size: {model.get_input_embeddings().weight.shape[0]}")
        logger.info(f"Adapter tokenizer vocab size: {len(adapter_tokenizer)}")
        
        if model.get_input_embeddings().weight.shape[0] != len(adapter_tokenizer):
            logger.info("Resizing model embeddings to match tokenizer")
            model.resize_token_embeddings(len(adapter_tokenizer))
            logger.info(f"New model vocab size: {model.get_input_embeddings().weight.shape[0]}")
        
        return True
    except Exception as e:
        logger.error(f"Error resizing model embeddings: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Override base model detected in adapter_config.json"
    )
    parser.add_argument(
        "--no-resize-embeddings", action="store_true",
        help="Don't resize embeddings to match adapter tokenizer"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force using CPU instead of GPU"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Use fp16 precision to reduce memory usage"
    )
    
    args = parser.parse_args()
    
    # Check if adapter path exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        logger.error(f"No valid LoRA adapter found at {args.adapter_path}")
        sys.exit(1)
    
    # Get base model name if not specified
    base_model = args.base_model
    if not base_model:
        base_model = get_base_model_name(args.adapter_path)
        if not base_model:
            logger.error("Could not determine base model name. Please provide --base-model.")
            sys.exit(1)
    
    # Load tokenizer
    adapter_tokenizer = load_tokenizer(args.adapter_path, base_model)
    if not adapter_tokenizer:
        logger.error("Failed to load tokenizer. Aborting.")
        sys.exit(1)
    
    logger.info(f"Adapter tokenizer vocabulary size: {len(adapter_tokenizer)}")
    
    # Load base model
    try:
        logger.info(f"Loading base model: {base_model}")
        
        # Determine device and dtype
        device_map = "cpu" if args.cpu else "auto"
        torch_dtype = torch.float16 if args.fp16 else torch.float32 if args.cpu else torch.bfloat16
        
        logger.info(f"Using device: {device_map}, dtype: {torch_dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        logger.info("Base model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        sys.exit(1)
    
    # Resize model embeddings if needed
    if not args.no_resize_embeddings:
        try:
            # Move to CPU for resizing if needed
            if not args.cpu and torch.cuda.is_available():
                logger.info("Moving model to CPU for embedding resize to save GPU memory")
                model_device = next(model.parameters()).device
                model = model.cpu()
            
            if not resize_tokenizer_embeddings(model, adapter_tokenizer):
                logger.error("Failed to resize model embeddings. Continuing without resizing.")
                
            # Move back to original device if needed
            if not args.cpu and torch.cuda.is_available():
                logger.info(f"Moving model back to {model_device} after resize")
                model = model.to(model_device)
        except Exception as e:
            logger.error(f"Error during embedding resize: {e}")
            logger.error("Continuing without resizing embeddings.")
    
    # Load and merge LoRA adapter
    try:
        # Make sure we're operating with minimal memory usage
        if not args.cpu and torch.cuda.is_available():
            logger.info("Clearing CUDA cache before adapter loading")
            torch.cuda.empty_cache()
        
        logger.info(f"Loading and merging LoRA adapter from: {args.adapter_path}")
        
        # Move to CPU for adapter loading to save GPU memory
        if not args.cpu and torch.cuda.is_available():
            logger.info("Moving model to CPU for adapter loading to save GPU memory")
            model_device = next(model.parameters()).device
            model = model.cpu()
        
        model = PeftModel.from_pretrained(
            model, 
            args.adapter_path,
            torch_dtype=torch.float32,  # Use float32 precision for CPU operations
        )
        logger.info("LoRA adapter loaded successfully")
        
        logger.info("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        logger.info("Successfully merged LoRA weights!")
        
        # Move back to original device if needed
        if not args.cpu and torch.cuda.is_available():
            logger.info(f"Moving merged model back to {model_device}")
            merged_model = merged_model.to(model_device)
    except Exception as e:
        logger.error(f"Failed to load or merge LoRA adapter: {e}")
        sys.exit(1)
    
    # Save merged model and tokenizer
    try:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged model to: {args.output_path}")
        merged_model.save_pretrained(args.output_path)
        adapter_tokenizer.save_pretrained(args.output_path)
        logger.info("Model and tokenizer saved successfully!")
        
        # Save a README file with info about the merge
        with open(output_path / "README.md", "w") as f:
            f.write(f"# Merged Model\n\n")
            f.write(f"This model was created by merging the LoRA adapter from `{args.adapter_path}` ")
            f.write(f"into the base model `{base_model}`.\n\n")
            f.write(f"## Original Adapter Config\n\n")
            
            # Copy adapter config info for reference
            try:
                with open(adapter_path / "adapter_config.json", "r") as adapter_config_file:
                    adapter_config = json.load(adapter_config_file)
                    f.write(f"```json\n{json.dumps(adapter_config, indent=2)}\n```\n")
            except Exception:
                f.write("(Could not load adapter config)\n")
    except Exception as e:
        logger.error(f"Failed to save merged model: {e}")
        sys.exit(1)
    
    logger.info("✅ Merge completed successfully! You can now use the merged model for inference.")


if __name__ == "__main__":
    main()
