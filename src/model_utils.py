"""
Model utilities for loading, applying LoRA, and saving models.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import json


def load_base_model(
    model_name: str,
    tokenizer_name: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load base model and tokenizer using mlx-lm.
    
    Args:
        model_name: Model identifier or path
        tokenizer_name: Tokenizer identifier (defaults to model_name)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from mlx_lm import load
    
    tokenizer_name = tokenizer_name or model_name
    model, tokenizer = load(model_name)
    
    print(f"Loaded model: {model_name}")
    return model, tokenizer


def apply_lora(
    model: Any,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[list] = None,
    num_layers: int = -1,
) -> Any:
    """
    Apply LoRA adapters to model.
    
    Args:
        model: Base model
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to (keys in config)
        num_layers: Number of layers to apply LoRA to (-1 for all layers)
    
    Returns:
        Model with LoRA adapters applied
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers
    
    if target_modules is None:
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Build LoRA config for the new API
    lora_config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "scale": alpha / rank,
        "keys": target_modules,
    }
    
    # Get number of layers from model if not specified
    if num_layers == -1:
        # Try to get from model architecture
        if hasattr(model, 'layers'):
            num_layers = len(model.layers)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        else:
            num_layers = 32  # Default fallback
    
    linear_to_lora_layers(model, num_layers, lora_config)
    
    # Use mlx_lm's helper to count trainable parameters
    from mlx_lm.tuner.utils import print_trainable_parameters
    print(f"Applied LoRA with rank={rank}, alpha={alpha} to {num_layers} layers")
    print_trainable_parameters(model)
    
    return model


def fuse_lora(
    model: Any,
    adapter_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Fuse LoRA adapters into base model weights.
    
    Args:
        model: Base model with LoRA adapters
        adapter_path: Path to saved adapter weights
        output_path: Path to save fused model
    
    Returns:
        Path to fused model
    """
    from mlx_lm.tuner.utils import fuse_lora_layers
    import mlx.core as mx
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load adapter weights
    adapter_path = Path(adapter_path)
    adapters = mx.load(str(adapter_path / "adapters.safetensors"))
    
    # Fuse weights
    fused_model = fuse_lora_layers(model)
    
    # Save fused model
    weights = dict(fused_model.parameters())
    mx.save_safetensors(str(output_path / "model.safetensors"), weights)
    
    print(f"Fused model saved to {output_path}")
    return output_path


def save_adapters(
    model: Any,
    output_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save LoRA adapter weights.
    
    Args:
        model: Model with LoRA adapters
        output_path: Path to save adapters
        config: Optional config to save alongside
    
    Returns:
        Path to saved adapters
    """
    import mlx.core as mx
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get trainable (LoRA) parameters only
    trainable_params = dict(model.trainable_parameters())
    
    # Save adapters
    mx.save_safetensors(str(output_path / "adapters.safetensors"), trainable_params)
    
    # Save config if provided
    if config:
        with open(output_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Saved {len(trainable_params)} adapter layers to {output_path}")
    return output_path


def load_adapters(
    model: Any,
    adapter_path: Union[str, Path],
) -> Any:
    """
    Load LoRA adapter weights into model.
    
    Args:
        model: Model with LoRA layers applied
        adapter_path: Path to saved adapters
    
    Returns:
        Model with loaded adapter weights
    """
    import mlx.core as mx
    
    adapter_path = Path(adapter_path)
    adapters = mx.load(str(adapter_path / "adapters.safetensors"))
    
    model.load_weights(list(adapters.items()))
    
    print(f"Loaded adapters from {adapter_path}")
    return model
