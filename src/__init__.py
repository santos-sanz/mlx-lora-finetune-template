"""
MLX LoRA Fine-tuning Engine

A toolkit for fine-tuning LLM models using LoRA and MLX on Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "Santos Sanz"

from .config import LoRAConfig, TrainingConfig, ModelConfig, HuggingFaceConfig
from .data_utils import load_dataset, prepare_training_data, create_train_val_split, save_jsonl
from .hf_utils import download_model, upload_model, upload_checkpoint
from .model_utils import load_base_model, apply_lora, fuse_lora, save_adapters
from .trainer import LoRATrainer

__all__ = [
    # Config
    "LoRAConfig",
    "TrainingConfig", 
    "ModelConfig",
    "HuggingFaceConfig",
    # Data utilities
    "load_dataset",
    "prepare_training_data",
    "create_train_val_split",
    "save_jsonl",
    # HuggingFace utilities
    "download_model",
    "upload_model",
    "upload_checkpoint",
    # Model utilities
    "load_base_model",
    "apply_lora",
    "fuse_lora",
    "save_adapters",
    # Trainer
    "LoRATrainer",
]
