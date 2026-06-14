"""Tests for GRPO trainer math helpers."""

from __future__ import annotations

import importlib
import sys
import types


def _install_fake_mlx(monkeypatch):
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.array = lambda x: x
    fake_mx_core.exp = lambda x: x
    fake_mx_core.clip = lambda x, *_: x
    fake_mx_core.minimum = min
    fake_mx_core.logsumexp = lambda x, **kwargs: x
    fake_mx_core.save_safetensors = lambda *args, **kwargs: None
    fake_mx_core.load = lambda *args, **kwargs: {}
    fake_mx_core.eval = lambda *args, **kwargs: None

    fake_nn = types.ModuleType("mlx.nn")
    fake_nn.Module = object
    fake_nn.losses = types.SimpleNamespace(cross_entropy=lambda *args, **kwargs: 0.0)
    fake_nn.value_and_grad = lambda *args, **kwargs: None

    fake_optim = types.ModuleType("mlx.optimizers")
    fake_optim.AdamW = object
    fake_utils = types.ModuleType("mlx.utils")
    fake_utils.tree_map = lambda fn, *trees: trees[0] if trees else None

    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx_core
    fake_mlx.nn = fake_nn
    fake_mlx.optimizers = fake_optim
    fake_mlx.utils = fake_utils

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", fake_optim)
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_utils)


def test_group_advantages_have_zero_mean(monkeypatch):
    _install_fake_mlx(monkeypatch)
    mod = importlib.reload(importlib.import_module("src.grpo_trainer"))

    rewards = [1.0, 0.0, 0.5, 0.5]
    adv = mod.GRPOTrainer.compute_group_advantages(rewards)

    assert len(adv) == len(rewards)
    assert abs(sum(adv)) < 1e-6


def test_clipped_policy_objective_respects_clip(monkeypatch):
    _install_fake_mlx(monkeypatch)
    mod = importlib.reload(importlib.import_module("src.grpo_trainer"))

    # ratio above clip should clip to 1 + eps for positive advantages.
    obj = mod.GRPOTrainer.clipped_policy_objective(ratio=3.0, advantage=1.0, clip_epsilon=0.2)
    assert abs(obj - 1.2) < 1e-9

    # ratio below clip should clip to 1 - eps for negative advantages.
    obj_neg = mod.GRPOTrainer.clipped_policy_objective(ratio=0.1, advantage=-1.0, clip_epsilon=0.2)
    assert abs(obj_neg + 0.8) < 1e-9
