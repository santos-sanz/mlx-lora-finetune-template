"""Training preset helpers shared by the Streamlit UI and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.config import Config


@dataclass(frozen=True)
class TrainingPreset:
    """Beginner-friendly training defaults presented as one selectable preset."""

    description: str
    lora_rank: int
    lora_alpha: int
    batch_size: int
    epochs: int
    learning_rate: float


TRAINING_PRESETS: Mapping[str, TrainingPreset] = {
    "🚀 Quick Test": TrainingPreset(
        description="Fast training to verify your setup before a longer run.",
        lora_rank=4,
        lora_alpha=8,
        batch_size=2,
        epochs=1,
        learning_rate=2e-4,
    ),
    "⚖️ Balanced": TrainingPreset(
        description="Recommended for most users: a practical quality and time balance.",
        lora_rank=8,
        lora_alpha=16,
        batch_size=4,
        epochs=3,
        learning_rate=1e-4,
    ),
    "🎯 High Quality": TrainingPreset(
        description="A longer run for datasets that already passed a quick test.",
        lora_rank=16,
        lora_alpha=32,
        batch_size=4,
        epochs=5,
        learning_rate=5e-5,
    ),
    "🔬 Maximum Quality": TrainingPreset(
        description="An extended run. Monitor validation metrics to avoid overfitting.",
        lora_rank=32,
        lora_alpha=64,
        batch_size=2,
        epochs=10,
        learning_rate=2e-5,
    ),
}


def preset_matches(config: Config, preset: TrainingPreset) -> bool:
    """Return whether the editable config still matches a preset exactly."""
    return (
        config.lora.rank == preset.lora_rank
        and config.lora.alpha == preset.lora_alpha
        and config.training.batch_size == preset.batch_size
        and config.training.num_epochs == preset.epochs
        and config.training.learning_rate == preset.learning_rate
    )


def infer_training_preset(config: Config) -> str | None:
    """Return the matching preset name, or None for custom settings."""
    for preset_name, preset in TRAINING_PRESETS.items():
        if preset_matches(config, preset):
            return preset_name
    return None


def apply_training_preset(config: Config, preset_name: str) -> None:
    """Apply a named preset to the editable config."""
    try:
        preset = TRAINING_PRESETS[preset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown training preset: {preset_name}") from exc

    config.lora.rank = preset.lora_rank
    config.lora.alpha = preset.lora_alpha
    config.training.batch_size = preset.batch_size
    config.training.num_epochs = preset.epochs
    config.training.learning_rate = preset.learning_rate
