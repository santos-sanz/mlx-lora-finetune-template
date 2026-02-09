"""
Configuration classes for MLX LoRA fine-tuning.

This module provides dataclass-based configurations for:
- LoRA parameters
- Training hyperparameters
- Model settings
- Hugging Face integration
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
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
    
    # RL data paths (GRPO)
    rl_train_file: str = "data/processed/rl_train.jsonl"
    rl_valid_file: str = "data/processed/rl_valid.jsonl"
    rl_eval_file: str = "data/processed/rl_eval.jsonl"

    # Training method
    training_method: str = "basic"  # "basic", "kfold", "grpo", or "grpo_kfold"

    # K-Fold Cross-Validation settings
    kfold_splits: int = 5  # Number of folds (3-10)
    kfold_seed: int = 42  # Seed for reproducibility
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "train_file": self.train_file,
            "valid_file": self.valid_file,
            "rl_train_file": self.rl_train_file,
            "rl_valid_file": self.rl_valid_file,
            "rl_eval_file": self.rl_eval_file,
            "prompt_template": self.prompt_template,
            "training_method": self.training_method,
            "kfold_splits": self.kfold_splits,
            "kfold_seed": self.kfold_seed,
        }


@dataclass
class GRPOConfig:
    """Configuration for GRPO (Group Relative Policy Optimization) training."""

    group_size: int = 4
    clip_epsilon: float = 0.2
    beta_kl: float = 0.02
    advantage_epsilon: float = 1e-8

    # Generation settings
    max_generation_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 1.0

    # Training loop behavior
    warmup_epochs: int = 1
    preflight_sample_size: int = 128
    min_reward_std: float = 1e-6
    eval_steps: int = 50
    logging_steps: int = 10

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "group_size": self.group_size,
            "clip_epsilon": self.clip_epsilon,
            "beta_kl": self.beta_kl,
            "advantage_epsilon": self.advantage_epsilon,
            "max_generation_tokens": self.max_generation_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "warmup_epochs": self.warmup_epochs,
            "preflight_sample_size": self.preflight_sample_size,
            "min_reward_std": self.min_reward_std,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
        }


@dataclass
class RewardConfig:
    """Configuration for rule-based reward computation in GRPO."""

    function: str = "weighted_rules"
    weights: Dict[str, float] = field(default_factory=lambda: {
        "exact_match": 0.4,
        "keyword_coverage": 0.3,
        "json_format": 0.2,
        "length_band": 0.1,
    })
    pass_threshold: float = 0.6
    metadata_keyword_field: str = "keywords"
    min_response_length: int = 16
    max_response_length: int = 512
    require_nonzero_variance: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "function": self.function,
            "weights": self.weights,
            "pass_threshold": self.pass_threshold,
            "metadata_keyword_field": self.metadata_keyword_field,
            "min_response_length": self.min_response_length,
            "max_response_length": self.max_response_length,
            "require_nonzero_variance": self.require_nonzero_variance,
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
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
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
            grpo=GRPOConfig(**data.get("grpo", {})),
            reward=RewardConfig(**data.get("reward", {})),
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
            "grpo": self.grpo.to_dict(),
            "reward": self.reward.to_dict(),
            "data": self.data.to_dict(),
            "output": self.output.to_dict(),
            "huggingface": self.huggingface.to_dict(),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
