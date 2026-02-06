"""Small UI-agnostic status helpers."""

from pathlib import Path


def get_status_badge(condition: bool, success_text: str, fail_text: str) -> str:
    """Generate HTML for a success/error status badge."""
    if condition:
        return f'<span class="badge badge-success">✓ {success_text}</span>'
    return f'<span class="badge badge-error">✗ {fail_text}</span>'


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching a pattern in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

