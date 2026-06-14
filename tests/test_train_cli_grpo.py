"""Smoke tests for train CLI dispatch with GRPO modes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


def _install_fake_mlx(monkeypatch):
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.array = object
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

    def fake_grpo(config, config_path, **kwargs):
        called["grpo"] += 1
        return {"mode": "grpo"}

    monkeypatch.setattr(module, "run_grpo", fake_grpo)
    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--training-method", "grpo"])
    module.main()

    assert called["grpo"] == 1


def test_cli_dispatches_grpo_kfold(monkeypatch, tmp_path):
    module = _load_train_module(monkeypatch)

    cfg = Config()
    cfg_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(cfg_path))

    called = {"grpo_kfold": 0}

    def fake_grpo_kfold(config, config_path, **kwargs):
        called["grpo_kfold"] += 1
        return {"mode": "grpo_kfold"}

    monkeypatch.setattr(module, "run_grpo_kfold", fake_grpo_kfold)
    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--training-method", "grpo_kfold"])
    module.main()

    assert called["grpo_kfold"] == 1


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_cli_dry_run_reports_limited_sft_plan_without_training(monkeypatch, tmp_path, capsys):
    module = _load_train_module(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    _write_jsonl(train_path, [{"text": f"train {i}"} for i in range(3)])
    _write_jsonl(valid_path, [{"text": f"valid {i}"} for i in range(2)])

    cfg = Config()
    cfg.data.train_file = str(train_path)
    cfg.data.valid_file = str(valid_path)
    cfg.output.dir = str(tmp_path / "outputs")
    cfg_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(cfg_path))

    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_path),
            "--dry-run",
            "--max-train-samples",
            "2",
            "--max-val-samples",
            "1",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert "Dry run complete" in output
    assert '"training_method": "basic"' in output
    assert '"loaded_samples": 3' in output
    assert '"used_samples": 2' in output
    assert '"loaded_samples": 2' in output
    assert '"used_samples": 1' in output


def test_cli_passes_sample_limits_to_grpo(monkeypatch, tmp_path):
    module = _load_train_module(monkeypatch)

    cfg = Config()
    cfg_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(cfg_path))

    captured = {}

    def fake_grpo(config, config_path, **kwargs):
        captured.update(kwargs)
        return {"mode": "grpo"}

    monkeypatch.setattr(module, "run_grpo", fake_grpo)
    monkeypatch.setattr(module, "run_basic_or_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(module, "run_grpo_kfold", lambda *_, **__: (_ for _ in ()).throw(AssertionError("unexpected")))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(cfg_path),
            "--training-method",
            "grpo",
            "--max-train-samples",
            "4",
            "--max-val-samples",
            "2",
        ],
    )
    module.main()

    assert captured["max_train_samples"] == 4
    assert captured["max_val_samples"] == 2
