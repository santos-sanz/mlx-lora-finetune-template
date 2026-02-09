"""Smoke tests for train CLI dispatch with GRPO modes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

from src.config import Config


def _install_fake_mlx(monkeypatch):
    fake_mx_core = types.ModuleType("mlx.core")
    fake_nn = types.ModuleType("mlx.nn")
    fake_nn.Module = object
    fake_optim = types.ModuleType("mlx.optimizers")
    fake_utils = types.ModuleType("mlx.utils")
    fake_utils.tree_map = lambda fn, *trees: trees[0] if trees else None
    fake_mx = types.ModuleType("mlx")
    fake_mx.core = fake_mx_core
    fake_mx.nn = fake_nn
    fake_mx.optimizers = fake_optim
    fake_mx.utils = fake_utils
    monkeypatch.setitem(sys.modules, "mlx", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", fake_optim)
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_utils)


def _load_train_module(monkeypatch):
    _install_fake_mlx(monkeypatch)
    train_path = Path(__file__).parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script_module", str(train_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_cli_dispatches_grpo(monkeypatch, tmp_path):
    module = _load_train_module(monkeypatch)

    cfg = Config()
    cfg_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(cfg_path))

    called = {"grpo": 0}

    def fake_grpo(config, config_path):
        called["grpo"] += 1
        return {"mode": "grpo"}

    monkeypatch.setattr(module, "run_grpo", fake_grpo)
    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo_kfold", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--training-method", "grpo"])
    module.main()

    assert called["grpo"] == 1


def test_cli_dispatches_grpo_kfold(monkeypatch, tmp_path):
    module = _load_train_module(monkeypatch)

    cfg = Config()
    cfg_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(cfg_path))

    called = {"grpo_kfold": 0}

    def fake_grpo_kfold(config, config_path):
        called["grpo_kfold"] += 1
        return {"mode": "grpo_kfold"}

    monkeypatch.setattr(module, "run_grpo_kfold", fake_grpo_kfold)
    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--training-method", "grpo_kfold"])
    module.main()

    assert called["grpo_kfold"] == 1
