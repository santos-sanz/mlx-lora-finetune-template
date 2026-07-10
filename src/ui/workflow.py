"""Shared workflow state for the Streamlit experience."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, MutableMapping, cast

from src.config import Config


PageId = Literal["home", "data", "train", "test", "upload"]

PAGE_LABELS: dict[PageId, str] = {
    "home": "🏠 Home",
    "data": "📊 Prepare Data",
    "train": "🚀 Train",
    "test": "🧪 Test Model",
    "upload": "☁️ HuggingFace",
}

_LEGACY_PAGE_IDS = {
    "config": "train",
}


def normalize_page(value: str | None) -> PageId:
    """Normalize page ids and legacy labels to a supported page id."""
    if value in PAGE_LABELS:
        return cast(PageId, value)

    if value in _LEGACY_PAGE_IDS:
        return cast(PageId, _LEGACY_PAGE_IDS[value])

    for page_id, label in PAGE_LABELS.items():
        if value == label:
            return page_id

    return "home"


def page_label(page_id: str) -> str:
    """Return the display label for a normalized page id."""
    return PAGE_LABELS[normalize_page(page_id)]


def set_page(state: MutableMapping[str, Any], page_id: str) -> PageId:
    """Update a Streamlit-like state mapping with a normalized page id."""
    normalized = normalize_page(page_id)
    state["page"] = normalized
    return normalized


def _resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class WorkflowReadiness:
    """Resolved readiness and next action for the guided workflow."""

    training_method: str
    model_ready: bool
    data_ready: bool
    data_path: Path
    hf_token_configured: bool
    has_checkpoint: bool
    has_adapter: bool

    @property
    def training_ready(self) -> bool:
        """Whether the current model and method-specific data can train."""
        return self.model_ready and self.data_ready

    @property
    def next_page(self) -> PageId:
        """Return the most useful next page for the current project state."""
        if not self.data_ready:
            return "data"
        if not self.model_ready or not (self.has_checkpoint or self.has_adapter):
            return "train"
        return "test"

    @property
    def next_action_label(self) -> str:
        """Return concise copy for the guided primary action."""
        return {
            "data": "📊 Prepare training data",
            "train": "🚀 Configure and train",
            "test": "🧪 Test your model",
        }[self.next_page]

    @property
    def next_action_description(self) -> str:
        """Explain why the recommended next action matters."""
        return {
            "data": f"Create the required dataset at `{self.data_path}` before training.",
            "train": "Review the model, preset, and training method, then save the command.",
            "test": "A training output is available. Compare it with the base model.",
        }[self.next_page]

    @property
    def data_kind(self) -> str:
        """Return the dataset family expected by the active method."""
        return "GRPO" if self.training_method in {"grpo", "grpo_kfold"} else "SFT"


def build_workflow_readiness(
    config: Config,
    project_root: Path,
    *,
    hf_token_configured: bool,
    has_checkpoint: bool = False,
    has_adapter: bool = False,
) -> WorkflowReadiness:
    """Resolve the current training prerequisites without mutating config."""
    method = getattr(config.data, "training_method", "basic")
    configured_path = (
        config.data.rl_train_file
        if method in {"grpo", "grpo_kfold"}
        else config.data.train_file
    )
    data_path = _resolve_path(project_root, configured_path)

    return WorkflowReadiness(
        training_method=method,
        model_ready=bool(config.model.name.strip()),
        data_ready=data_path.is_file(),
        data_path=data_path,
        hf_token_configured=hf_token_configured,
        has_checkpoint=has_checkpoint,
        has_adapter=has_adapter,
    )


def config_signature(config: Config) -> str:
    """Return a stable signature for the user-visible training configuration."""
    payload = {
        "model": config.model.to_dict(),
        "lora": config.lora.to_dict(),
        "training": config.training.to_dict(),
        "grpo": config.grpo.to_dict(),
        "reward": config.reward.to_dict(),
        "data": config.data.to_dict(),
        "output": config.output.to_dict(),
        "huggingface": config.huggingface.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
