"""Tests for trainer/model utility behavior that is critical for training correctness."""

from __future__ import annotations

import importlib
import sys
import types


class _FakeLoss:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


def _make_fake_tree_map():
    def tree_map(fn, *trees):
        if len(trees) == 1:
            tree = trees[0]
            if isinstance(tree, dict):
                return {k: tree_map(fn, v) for k, v in tree.items()}
            return fn(tree)

        left, right = trees
        if isinstance(left, dict):
            return {k: tree_map(fn, left[k], right[k]) for k in left}
        return fn(left, right)

    return tree_map


def _install_fake_mlx_for_trainer(monkeypatch):
    """Install fake mlx modules so src.trainer can be imported without MLX."""
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.eval = lambda *args, **kwargs: None
    fake_mx_core.array = lambda value: value
    fake_mx_core.save_safetensors = lambda *args, **kwargs: None

    fake_nn = types.ModuleType("mlx.nn")
    fake_nn.Module = object
    fake_nn.losses = types.SimpleNamespace(cross_entropy=lambda *args, **kwargs: _FakeLoss(1.0))

    def value_and_grad(model, loss_fn):
        def wrapped(current_model, batch):
            return _FakeLoss(1.0), {"w": 1.0}

        return wrapped

    fake_nn.value_and_grad = value_and_grad

    fake_optimizers = types.ModuleType("mlx.optimizers")

    class FakeAdamW:
        instances = []

        def __init__(self, learning_rate: float, weight_decay: float):
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.state = {}
            self.update_calls = []
            FakeAdamW.instances.append(self)

        def update(self, model, grads):
            self.update_calls.append(grads)

    fake_optimizers.AdamW = FakeAdamW

    fake_utils = types.ModuleType("mlx.utils")
    fake_utils.tree_map = _make_fake_tree_map()

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx_core
    fake_mlx.nn = fake_nn
    fake_mlx.optimizers = fake_optimizers
    fake_mlx.utils = fake_utils

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", fake_optimizers)
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_utils)

    return FakeAdamW


def _install_fake_mlx_core(monkeypatch, fake_mx_core: types.ModuleType) -> None:
    """Install a fake mlx package exposing mlx.core for function-local imports."""
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)


def test_trainer_uses_gradient_accumulation_and_tracks_tokens(monkeypatch, tmp_path):
    """When accumulation is enabled, optimizer updates should happen per effective step."""
    fake_adamw = _install_fake_mlx_for_trainer(monkeypatch)
    trainer_module = importlib.reload(importlib.import_module("src.trainer"))

    class DummyModel:
        def parameters(self):
            return {"w": 1.0}

        def trainable_parameters(self):
            return {"w": 1.0}

    class DummyTokenizer:
        def encode(self, text: str):
            return [1, 2, 3, 4]

    trainer = trainer_module.LoRATrainer(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        train_data=[{"text": "a"}, {"text": "b"}],
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_steps=1000,
        eval_steps=1000,
        output_dir=tmp_path,
    )
    trainer._save_checkpoint = lambda *_: None
    trainer.train()

    # Two micro-batches with grad_accum=2 must produce one optimizer update.
    assert len(fake_adamw.instances) == 1
    assert len(fake_adamw.instances[0].update_calls) == 1
    assert trainer.global_step == 1
    assert trainer.training_log
    assert trainer.training_log[0]["tps"] > 0


def test_trainer_handles_zero_warmup_steps(monkeypatch):
    """Warmup=0 should not cause division errors and should use base LR directly."""
    _install_fake_mlx_for_trainer(monkeypatch)
    trainer_module = importlib.reload(importlib.import_module("src.trainer"))

    class DummyModel:
        pass

    class DummyTokenizer:
        def encode(self, text: str):
            return [1, 2]

    trainer = trainer_module.LoRATrainer(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        train_data=[],
        warmup_steps=0,
    )
    assert trainer._get_lr(0) == trainer.learning_rate


def test_fuse_lora_loads_adapters_before_fusing(monkeypatch, tmp_path):
    """fuse_lora should load adapter weights into model before calling fuse."""
    from src import model_utils

    captured = {"save_path": None}

    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.load = lambda path: {"adapter.weight": 1.0}
    fake_mx_core.save_safetensors = lambda path, weights: captured.update(save_path=path, weights=weights)

    fake_tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    fake_tuner_utils.fuse_lora_layers = lambda model: model

    _install_fake_mlx_core(monkeypatch, fake_mx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_tuner_utils)

    class DummyModel:
        def __init__(self):
            self.loaded = None

        def load_weights(self, items):
            self.loaded = list(items)

        def parameters(self):
            return {"model.weight": 1.0}.items()

    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "fused"
    adapter_dir.mkdir()

    model = DummyModel()
    result_path = model_utils.fuse_lora(model, adapter_dir, output_dir)

    assert model.loaded is not None
    assert str(result_path) == str(output_dir)
    assert captured["save_path"] == str(output_dir / "model.safetensors")


def test_save_adapters_flattens_nested_parameters(monkeypatch, tmp_path):
    """save_adapters should flatten nested trainable parameters before saving."""
    from src import model_utils

    captured = {}

    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.save_safetensors = lambda path, weights: captured.update(path=path, weights=weights)
    _install_fake_mlx_core(monkeypatch, fake_mx_core)

    class DummyModel:
        def trainable_parameters(self):
            return {"layer": {"weight": 1.0, "bias": 0.5}}

    output_path = model_utils.save_adapters(DummyModel(), tmp_path)

    assert str(output_path) == str(tmp_path)
    assert "layer.weight" in captured["weights"]
    assert "layer.bias" in captured["weights"]
