"""
Configuration classes for MLX LoRA fine-tuning.

This module provides dataclass-based configurations for:
- LoRA parameters
- Training hyperparameters
- Model settings
- Hugging Face integration
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
        }


@dataclass
class ModelConfig:
    """Configuration for base model."""
    
    name: str = "meta-llama/Llama-3.2-1B"
    tokenizer: Optional[str] = None
    max_seq_length: int = 2048
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "tokenizer": self.tokenizer or self.name,
            "max_seq_length": self.max_seq_length,
        }


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face integration."""
    
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    private: bool = True
    token: Optional[str] = None
    
    def __post_init__(self):
        """Load token from environment if not provided."""
        load_dotenv()
        if self.token is None:
            self.token = os.getenv("HF_TOKEN")
        if self.repo_id is None:
            self.repo_id = os.getenv("HF_REPO_ID")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (excluding token for security)."""
        return {
            "push_to_hub": self.push_to_hub,
            "repo_id": self.repo_id,
            "private": self.private,
        }


@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    
    train_file: str = "data/processed/train.jsonl"
    valid_file: str = "data/processed/valid.jsonl"
    prompt_template: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "train_file": self.train_file,
            "valid_file": self.valid_file,
            "prompt_template": self.prompt_template,
        }


@dataclass
class OutputConfig:
    """Configuration for output directories."""
    
    dir: str = "outputs"
    adapters_dir: str = "outputs/adapters"
    checkpoints_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    
    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for path in [self.dir, self.adapters_dir, self.checkpoints_dir, self.logs_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "dir": self.dir,
            "adapters_dir": self.adapters_dir,
            "checkpoints_dir": self.checkpoints_dir,
            "logs_dir": self.logs_dir,
        }


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            output=OutputConfig(**data.get("output", {})),
            huggingface=HuggingFaceConfig(**data.get("huggingface", {})),
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            "model": self.model.to_dict(),
            "lora": self.lora.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "output": self.output.to_dict(),
            "huggingface": self.huggingface.to_dict(),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
