"""Tests for guided navigation, readiness, and training presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config
from src.ui.presets import apply_training_preset, infer_training_preset
from src.ui.workflow import (
    PAGE_LABELS,
    build_workflow_readiness,
    config_signature,
    normalize_page,
    page_label,
    set_page,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("home", "home"),
        ("data", "data"),
        ("config", "train"),
        ("🚀 Train", "train"),
        ("unknown", "home"),
        (None, "home"),
    ],
)
def test_normalize_page_supports_ids_labels_and_legacy_values(value, expected):
    assert normalize_page(value) == expected


def test_page_label_and_state_update_share_the_same_page_ids():
    state = {}

    selected = set_page(state, "config")

    assert selected == "train"
    assert state["page"] == "train"
    assert page_label(selected) == PAGE_LABELS["train"]


def _write_training_file(project_root: Path, relative_path: str) -> Path:
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"text": "sample"}\n', encoding="utf-8")
    return path


def test_basic_readiness_routes_missing_data_to_preparation(tmp_path):
    config = Config()

    readiness = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=False,
    )

    assert readiness.data_kind == "SFT"
    assert not readiness.data_ready
    assert not readiness.training_ready
    assert readiness.next_page == "data"
    assert "Prepare training data" in readiness.next_action_label


def test_public_or_local_model_is_ready_without_hf_token(tmp_path):
    config = Config()
    _write_training_file(tmp_path, config.data.train_file)

    readiness = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=False,
    )

    assert readiness.model_ready
    assert readiness.data_ready
    assert readiness.training_ready
    assert not readiness.hf_token_configured
    assert readiness.next_page == "train"


def test_grpo_readiness_uses_rl_training_file(tmp_path):
    config = Config()
    config.data.training_method = "grpo"
    _write_training_file(tmp_path, config.data.train_file)

    missing_rl = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=True,
    )
    assert missing_rl.data_kind == "GRPO"
    assert not missing_rl.data_ready

    rl_path = _write_training_file(tmp_path, config.data.rl_train_file)
    ready_rl = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=True,
    )
    assert ready_rl.data_path == rl_path
    assert ready_rl.training_ready


def test_available_output_routes_to_model_testing(tmp_path):
    config = Config()
    _write_training_file(tmp_path, config.data.train_file)

    readiness = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=False,
        has_checkpoint=True,
    )

    assert readiness.next_page == "test"
    assert "Test your model" in readiness.next_action_label


def test_missing_model_routes_ready_data_to_training_configuration(tmp_path):
    config = Config()
    config.model.name = ""
    _write_training_file(tmp_path, config.data.train_file)

    readiness = build_workflow_readiness(
        config,
        tmp_path,
        hf_token_configured=True,
    )

    assert readiness.data_ready
    assert not readiness.model_ready
    assert not readiness.training_ready
    assert readiness.next_page == "train"


def test_config_signature_changes_with_visible_training_settings():
    config = Config()
    original = config_signature(config)

    config.training.batch_size += 1

    assert config_signature(config) != original


def test_presets_are_inferred_applied_and_mark_custom_settings():
    config = Config()
    assert infer_training_preset(config) == "⚖️ Balanced"

    apply_training_preset(config, "🚀 Quick Test")
    assert infer_training_preset(config) == "🚀 Quick Test"
    assert config.training.num_epochs == 1
    assert config.lora.rank == 4

    config.training.batch_size = 7
    assert infer_training_preset(config) is None


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown training preset"):
        apply_training_preset(Config(), "Not a preset")
