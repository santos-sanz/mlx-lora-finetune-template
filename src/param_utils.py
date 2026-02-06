"""Utilities for serializing nested MLX parameter containers."""

from typing import Any


def flatten_params(container: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dict/list structures to dotted-key dictionaries."""
    items: list[tuple[str, Any]] = []
    iterator = container.items() if isinstance(container, dict) else enumerate(container)

    for key, value in iterator:
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, (dict, list)):
            items.extend(flatten_params(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

