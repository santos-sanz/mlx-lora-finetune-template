import mlx.nn as nn
from mlx_lm import load

model, _ = load("LiquidAI/LFM2-2.6B-Exp")
print(model)

def print_modules(m, prefix=""):
    for name, submodule in m.named_modules():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(submodule, nn.Linear):
            print(f"Linear: {path}")

print_modules(model)
