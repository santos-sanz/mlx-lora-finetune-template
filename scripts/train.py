#!/usr/bin/env python3
"""
Training script for MLX LoRA fine-tuning.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --model meta-llama/Llama-3.2-1B --data data/processed
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_utils import load_dataset
from src.model_utils import load_base_model, apply_lora, save_adapters
from src.trainer import LoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with MLX")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--data", type=str, help="Override data directory")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Apply CLI overrides
    if args.model:
        config.model.name = args.model
    if args.data:
        config.data.train_file = f"{args.data}/train.jsonl"
        config.data.valid_file = f"{args.data}/valid.jsonl"
    if args.output:
        config.output.dir = args.output
        config.output.adapters_dir = f"{args.output}/adapters"
        config.output.checkpoints_dir = f"{args.output}/checkpoints"
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.lora_rank:
        config.lora.rank = args.lora_rank
    
    print("=" * 60)
    print("MLX LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"LoRA rank: {config.lora.rank}, alpha: {config.lora.alpha}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}")
    print("=" * 60)
    
    # Ensure output directories exist
    config.output.ensure_dirs()
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_base_model(config.model.name)
    
    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = apply_lora(
        model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    
    # Load data
    print("\nLoading training data...")
    train_data = load_dataset(config.data.train_file)
    print(f"Loaded {len(train_data)} training examples")
    
    val_data = None
    if Path(config.data.valid_file).exists():
        val_data = load_dataset(config.data.valid_file)
        print(f"Loaded {len(val_data)} validation examples")
    
    # Create trainer
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_seq_length=config.model.max_seq_length,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.output.dir,
    )
    
    # Train
    print("\nStarting training...")
    stats = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Final loss: {stats['final_loss']:.4f}")
    if stats['best_val_loss']:
        print(f"Best validation loss: {stats['best_val_loss']:.4f}")
    print(f"Model saved to: {config.output.checkpoints_dir}/final")
    print("=" * 60)


if __name__ == "__main__":
    main()
