"""
MLX LoRA Fine-tuning Engine

A toolkit for fine-tuning LLM models using LoRA and MLX on Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "Santos Sanz"

# Core modules (no MLX dependency)
from .config import LoRAConfig, TrainingConfig, ModelConfig, HuggingFaceConfig, Config
from .data_utils import (
    load_dataset, prepare_training_data, create_train_val_split, 
    save_jsonl, create_kfold_splits, get_kfold_data
)
from .hf_utils import get_hf_token, upload_model, upload_checkpoint

# MLX-dependent modules - import lazily to avoid errors when MLX not installed
def __getattr__(name):
    """Lazy import for MLX-dependent modules."""
    if name in ("load_base_model", "apply_lora", "fuse_lora", "save_adapters"):
        from . import model_utils
        return getattr(model_utils, name)
    elif name == "LoRATrainer":
        from .trainer import LoRATrainer
        return LoRATrainer
    elif name == "KFoldTrainer":
        from .trainer import KFoldTrainer
        return KFoldTrainer
    elif name == "download_model":
        from .hf_utils import download_model
        return download_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Config
    "LoRAConfig",
    "TrainingConfig", 
    "ModelConfig",
    "HuggingFaceConfig",
    "Config",
    # Data utilities
    "load_dataset",
    "prepare_training_data",
    "create_train_val_split",
    "create_kfold_splits",
    "get_kfold_data",
    "save_jsonl",
    # HuggingFace utilities
    "get_hf_token",
    "upload_model",
    "upload_checkpoint",
    # Model utilities (lazy)
    "load_base_model",
    "apply_lora",
    "fuse_lora",
    "save_adapters",
    # Trainers (lazy)
    "LoRATrainer",
    "KFoldTrainer",
]
