
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

def main():
    print("Loading model...")
    model, tokenizer = load("Qwen/Qwen3-0.6B")
    
    print("\nFreezing model...")
    model.freeze()

    print("\nApplying LoRA...")
    
    # Correct keys for Qwen structure
    correct_keys = ["self_attn.q_proj", "self_attn.v_proj"]
    lora_config = {"rank": 4, "alpha": 8, "dropout": 0.0, "scale": 2.0, "keys": correct_keys}
    linear_to_lora_layers(model, 32, lora_config)
    
    print("\ninspecting model structure after LoRA...")
    # Inspect one layer
    if hasattr(model, "layers"):
        layer0 = model.layers[0]
        # Check attention
        if hasattr(layer0, "self_attn"):
            attn = layer0.self_attn
            if hasattr(attn, "q_proj"):
                print(f"q_proj type: {type(attn.q_proj)}")

    print("\nTraversing parameters...")
    count = 0
    for name, p in model.parameters().items():
        count += 1
    print(f"Total parameter arrays: {count}")
    
    print("\nTrainable parameters:")
    trainable = model.trainable_parameters()
    print(f"Trainable count: {len(trainable)}")
    for k, v in trainable.items():
        # print first few
        if count < 5:
            print(f"  {k}: {type(v)}")
        count += 1

    print("\nTraversing parameters...")
    count = 0
    for name, p in model.parameters().items():
        # parameters() returns flat dict of arrays
        count += 1
    print(f"Total parameter arrays: {count}")
    
    print("\nTrainable parameters:")
    trainable = model.trainable_parameters()
    print(f"Trainable count: {len(trainable)}")
    for k, v in trainable.items():
        print(f"  {k}: {type(v)}")

if __name__ == "__main__":
    main()
