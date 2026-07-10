"""Session-state initialization helpers for Streamlit."""

from pathlib import Path
from typing import Dict, Any

import streamlit as st
import yaml

from src.config import Config


def _load_config_with_fallback(project_root: Path) -> Config:
    """Load current config when possible, otherwise fall back to defaults."""
    current_config_path = project_root / "configs" / "current.yaml"
    default_config_path = project_root / "configs" / "default.yaml"

    candidates = [current_config_path, default_config_path]
    st.session_state["config_load_error"] = None

    for config_path in candidates:
        if not config_path.exists():
            continue
        try:
            return Config.from_yaml(str(config_path))
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            st.session_state["config_load_error"] = (
                f"Failed loading {config_path.name}: {exc}"
            )

    return Config()


def init_session_state(project_root: Path) -> None:
    """Initialize session state and load config from disk if available."""
    defaults: Dict[str, Any] = {
        "config": None,
        "training_process": None,
        "training_logs": [],
        "training_running": False,
        "log_queue": None,
        "theme": "dark",
        "test_base_model": None,
        "test_finetuned_model": None,
        "test_tokenizer": None,
        "test_chat_history": [],
        "selected_checkpoint": None,
        "test_models_loaded": False,
        "config_load_error": None,
        "page": "home",
        "saved_config_signature": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.config is None:
        st.session_state.config = _load_config_with_fallback(project_root)
