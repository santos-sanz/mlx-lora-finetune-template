"""
Data utilities for preparing training and validation datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import random


def load_dataset(path: Union[str, Path], format: str = "auto") -> List[Dict[str, Any]]:
    """Load dataset from file (json or jsonl format)."""
    path = Path(path)
    
    if format == "auto":
        format = "jsonl" if path.suffix == ".jsonl" else "json"
    
    if format == "jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("data", data) if isinstance(data, dict) else data


def prepare_training_data(
    data: List[Dict[str, Any]],
    instruction_key: str = "instruction",
    response_key: str = "response",
    text_key: Optional[str] = None,
    template: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Prepare data for MLX LoRA training format."""
    prepared = []
    
    for item in data:
        if text_key and text_key in item:
            prepared.append({"text": item[text_key]})
        else:
            instruction = item.get(instruction_key, "")
            response = item.get(response_key, "")
            
            if template:
                text = template.format(instruction=instruction, response=response)
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
            prepared.append({"text": text})
    
    return prepared


def create_train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into training and validation sets."""
    if shuffle:
        random.seed(seed)
        data = data.copy()
        random.shuffle(data)
    
    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save data to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_mlx_format(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.1,
    instruction_key: str = "instruction",
    response_key: str = "response",
    template: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Convert dataset to MLX training format with train/val split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    raw_data = load_dataset(input_path)
    prepared_data = prepare_training_data(
        raw_data,
        instruction_key=instruction_key,
        response_key=response_key,
        template=template,
    )
    
    # Split data
    train_data, val_data = create_train_val_split(prepared_data, val_ratio=val_ratio)
    
    # Save files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "valid.jsonl"
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    return train_path, val_path
