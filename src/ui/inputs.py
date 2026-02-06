"""Reusable Streamlit file/folder input components."""

from pathlib import Path
from typing import Optional, Sequence

import streamlit as st


def file_picker(
    label: str,
    default_path: str = "",
    file_types: Optional[Sequence[str]] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
) -> str:
    """Render a path input for files with lightweight validation feedback."""
    picker_key = key or f"fp_{label}"

    if picker_key not in st.session_state:
        st.session_state[picker_key] = default_path

    help_text = help
    if not help_text and file_types:
        help_text = f"Supported: {', '.join(file_types)}"

    path = st.text_input(
        label,
        value=st.session_state[picker_key],
        key=f"{picker_key}_input",
        help=help_text,
    )

    st.session_state[picker_key] = path

    if path:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_file():
                st.caption(f"✅ File exists ({path_obj.stat().st_size:,} bytes)")
            else:
                st.caption("⚠️ Path is a directory, not a file")
        else:
            st.caption("⚠️ File not found")

    return path


def folder_picker(
    label: str,
    default_path: str = "",
    key: Optional[str] = None,
    help: Optional[str] = None,
) -> str:
    """Render a path input for folders with lightweight validation feedback."""
    picker_key = key or f"dp_{label}"

    if picker_key not in st.session_state:
        st.session_state[picker_key] = default_path

    path = st.text_input(
        label,
        value=st.session_state[picker_key],
        key=f"{picker_key}_input",
        help=help or "Enter folder path",
    )

    st.session_state[picker_key] = path

    if path:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                try:
                    count = len(list(path_obj.iterdir()))
                    st.caption(f"✅ Folder exists ({count} items)")
                except OSError:
                    st.caption("✅ Folder exists")
            else:
                st.caption("⚠️ Path is a file, not a folder")
        else:
            st.caption("⚠️ Folder not found (will be created)")

    return path

