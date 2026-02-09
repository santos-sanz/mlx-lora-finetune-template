"""Rule-based reward functions for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Dict, Mapping


@dataclass
class RewardInput:
    """Input bundle used for reward computation."""

    sample: Dict[str, Any]
    response: str


@dataclass
class RewardOutput:
    """Structured reward output with per-component details."""

    total_score: float
    pass_threshold: float
    passed: bool
    components: Dict[str, float]


RewardFn = Callable[[Dict[str, Any], str, Mapping[str, Any]], float]


def _get_config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def reward_exact_match(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> float:
    """Binary exact match against the sample reference."""
    reference = str(sample.get("reference", "")).strip()
    if not reference:
        return 0.0

    normalize = bool(_get_config_value(config, "normalize_text", True))
    if normalize:
        return 1.0 if _normalize_text(reference) == _normalize_text(response) else 0.0
    return 1.0 if reference.strip() == response.strip() else 0.0


def reward_keyword_coverage(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> float:
    """Coverage of required keywords from metadata or reference text."""
    metadata = sample.get("metadata") or {}
    keyword_field = str(_get_config_value(config, "metadata_keyword_field", "keywords"))

    keywords = metadata.get(keyword_field)
    if keywords is None:
        reference = str(sample.get("reference", "")).strip()
        if not reference:
            return 0.0
        # Fallback: extract a lightweight keyword set from reference tokens.
        tokens = [tok for tok in re.findall(r"[A-Za-z0-9_]+", reference.lower()) if len(tok) > 3]
        keywords = sorted(set(tokens[:10]))

    if not isinstance(keywords, list) or not keywords:
        return 0.0

    resp = _normalize_text(response)
    hits = 0
    for kw in keywords:
        kw_norm = _normalize_text(str(kw))
        if kw_norm and kw_norm in resp:
            hits += 1
    return hits / len(keywords)


def reward_json_format(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> float:
    """Checks whether response is valid JSON."""
    text = response.strip()
    if not text:
        return 0.0
    try:
        json.loads(text)
        return 1.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def reward_length_band(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> float:
    """Rewards responses that stay inside a configured length band."""
    min_len = int(_get_config_value(config, "min_response_length", 16))
    max_len = int(_get_config_value(config, "max_response_length", 512))

    n = len(response.strip())
    if min_len <= n <= max_len:
        return 1.0
    if n < min_len:
        return max(0.0, n / max(min_len, 1))
    # n > max_len
    overflow = n - max_len
    return max(0.0, 1.0 - overflow / max(max_len, 1))


def _weighted_rules(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> Dict[str, float]:
    weights = dict(_get_config_value(config, "weights", {}))
    if not weights:
        weights = {
            "exact_match": 0.4,
            "keyword_coverage": 0.3,
            "json_format": 0.2,
            "length_band": 0.1,
        }

    total_w = sum(max(0.0, float(w)) for w in weights.values())
    if total_w <= 0:
        total_w = 1.0

    components: Dict[str, float] = {}
    for name, weight in weights.items():
        fn = REWARD_REGISTRY.get(name)
        if fn is None:
            components[name] = 0.0
            continue
        value = fn(sample, response, config)
        if math.isnan(value) or math.isinf(value):
            value = 0.0
        components[name] = max(0.0, min(1.0, float(value))) * max(0.0, float(weight)) / total_w

    return components


REWARD_REGISTRY: Dict[str, RewardFn] = {
    "exact_match": reward_exact_match,
    "keyword_coverage": reward_keyword_coverage,
    "json_format": reward_json_format,
    "length_band": reward_length_band,
}


def compute_reward(sample: Dict[str, Any], response: str, config: Mapping[str, Any]) -> RewardOutput:
    """Compute reward for a sample/response pair using configured rule set."""
    reward_function = str(_get_config_value(config, "function", "weighted_rules"))
    pass_threshold = float(_get_config_value(config, "pass_threshold", 0.6))

    if reward_function == "weighted_rules":
        components = _weighted_rules(sample, response, config)
        total = float(sum(components.values()))
    else:
        fn = REWARD_REGISTRY.get(reward_function)
        if fn is None:
            raise ValueError(f"Unknown reward function: {reward_function}")
        raw = fn(sample, response, config)
        if math.isnan(raw) or math.isinf(raw):
            raw = 0.0
        total = max(0.0, min(1.0, float(raw)))
        components = {reward_function: total}

    return RewardOutput(
        total_score=total,
        pass_threshold=pass_threshold,
        passed=total >= pass_threshold,
        components=components,
    )
