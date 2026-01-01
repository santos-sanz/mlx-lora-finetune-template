#!/usr/bin/env python3
"""
Data preparation script for MLX LoRA fine-tuning.

Usage:
    python scripts/prepare_data.py --input data/raw/dataset.json --output data/processed
    python scripts/prepare_data.py --input data/raw/dataset.jsonl --val-split 0.15
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import convert_to_mlx_format, load_dataset, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for MLX LoRA training")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input dataset (json or jsonl)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed files")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--instruction-key", type=str, default="instruction",
                        help="Key for instruction/input field")
    parser.add_argument("--response-key", type=str, default="response",
                        help="Key for response/output field")
    parser.add_argument("--template", type=str, default=None,
                        help="Custom template for formatting (use {instruction}, {response})")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Data Preparation for MLX LoRA Training")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Validation split: {args.val_split * 100:.0f}%")
    print("=" * 60)
    
    # Convert data
    train_path, val_path = convert_to_mlx_format(
        input_path=args.input,
        output_dir=args.output,
        val_ratio=args.val_split,
        instruction_key=args.instruction_key,
        response_key=args.response_key,
        template=args.template,
    )
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Training file: {train_path}")
    print(f"Validation file: {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
