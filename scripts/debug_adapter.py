
import sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config  # noqa: E402
from src.model_utils import apply_lora  # noqa: E402

def debug_adapters():
    # Load config
    config_path = project_root / "configs" / "current.yaml"
    config = Config.from_yaml(config_path)
    
    checkpoint_path = project_root / "outputs" / "checkpoints" / "best"
    adapter_file = checkpoint_path / "adapters.safetensors"
    
    if not adapter_file.exists():
        print(f"Error: Adapter file not found at {adapter_file}")
        return

    print(f"Loading adapters from {adapter_file}")
    adapters = mx.load(str(adapter_file))
    adapter_keys = set(adapters.keys())
    print(f"Found {len(adapter_keys)} adapter keys")
    
    # Extract target modules logic from app.py
    target_modules = set()
    lora_rank = None
    
    for key, value in adapters.items():
        if key.endswith('.lora_a') and lora_rank is None:
            print(f"Sample lora_a shape: {value.shape}")
            # In MLX, Linear is (in, out). 
            # LoRA A is usually (in, rank).
            # LoRA B is usually (rank, out).
            lora_rank = value.shape[1] 
        
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                module_parts = parts[i+2:-1]
                if module_parts:
                    module_name = '.'.join(module_parts)
                    target_modules.add(module_name)
                break
    
    target_modules_list = list(target_modules) if target_modules else None
    print(f"Extracted target modules: {target_modules_list}")
    print(f"Extracted rank: {lora_rank}")

    print(f"Loading base model: {config.model.name}")
    model, _ = load(config.model.name)
    
    print("Applying LoRA...")
    apply_lora(model, rank=lora_rank, alpha=config.lora.alpha, target_modules=target_modules_list)
    
    print("Checking model parameters...")
    
    def flatten_params(container, parent_key="", sep="."):
        items = []
        iterator = container.items() if isinstance(container, dict) else enumerate(container)
        
        for k, v in iterator:
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                items.extend(flatten_params(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    model_params = flatten_params(model.parameters())
    model_keys = set(model_params.keys())
    
    print(f"Model keys sample: {list(model_keys)[:5]}")
    
    print("Comparing keys...")
    missing_in_model = adapter_keys - model_keys
    
    if missing_in_model:
        print(f"CRITICAL: {len(missing_in_model)} keys from adapters are NOT in the model!")
        print(f"Example missing keys: {list(missing_in_model)[:5]}")
        
        # Check if it's a prefix issue
        adapter_sample = list(missing_in_model)[0]
        print(f"Checking if {adapter_sample} can be found with prefix stripping...")
        if adapter_sample.startswith("model."):
            stripped = adapter_sample[6:]
            if stripped in model_keys:
                print(f"FOUND {stripped} in model! It's a prefix issue.")
    else:
        print("SUCCESS: All adapter keys exist in the model.")
        
    print("Attempting load_weights with strict=True...")
    try:
        model.load_weights(list(adapters.items()), strict=False) # Keep strict=False to match app.py but we want to see if it works
        print("load_weights(strict=False) passed.")
    except Exception as e:
        print(f"load_weights(strict=False) failed: {e}")

    # Re-fetch model parameters AFTER loading
    model_params = flatten_params(model.parameters())
    
    try:
        # Pick one weight
        test_key = list(adapter_keys)[0]
        print(f"Verifying weight update for {test_key}")
        
        # Get current value
        current_val = model_params[test_key]
        adapter_val = adapters[test_key]
        
        # Check if adapter weight is non-zero
        if mx.max(mx.abs(adapter_val)) < 1e-6:
             print("WARNING: Adapter weight is effectively zero!")
        else:
             print(f"Adapter weight max abs value: {mx.max(mx.abs(adapter_val))}")

        # They should be equal now
        if mx.array_equal(current_val, adapter_val):
            print("Verification PASSED: Model weight matches adapter weight.")
        else:
            print("Verification FAILED: Model weight does not match adapter weight after loading!")
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    debug_adapters()
