"""RL dataset utilities and preflight validation for GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from src.data_utils import create_train_val_split, load_dataset, save_jsonl
from src.rewards import compute_reward


@dataclass
class RLSchemaValidation:
    """Validation summary for RL dataset schema quality."""

    total_samples: int
    valid_samples: int
    invalid_samples: int
    duplicate_samples: int
    missing_prompt: int
    missing_reference: int
    min_prompt_length: int
    min_reference_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "duplicate_samples": self.duplicate_samples,
            "missing_prompt": self.missing_prompt,
            "missing_reference": self.missing_reference,
            "min_prompt_length": self.min_prompt_length,
            "min_reference_length": self.min_reference_length,
        }


@dataclass
class RewardAuditReport:
    """Reward coverage and numeric stability report."""

    sample_size: int
    total_reward_mean: float
    total_reward_std: float
    total_reward_min: float
    total_reward_max: float
    component_stats: Dict[str, Dict[str, float]]
    has_nan_or_inf: bool
    has_nonzero_variance: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "total_reward_mean": self.total_reward_mean,
            "total_reward_std": self.total_reward_std,
            "total_reward_min": self.total_reward_min,
            "total_reward_max": self.total_reward_max,
            "component_stats": self.component_stats,
            "has_nan_or_inf": self.has_nan_or_inf,
            "has_nonzero_variance": self.has_nonzero_variance,
        }


def load_rl_dataset(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load RL dataset from json/jsonl and enforce list semantics."""
    data = load_dataset(path)
    if not isinstance(data, list):
        raise ValueError("RL dataset must be a list of samples")
    return data


def normalize_to_rl_sample(
    item: Mapping[str, Any],
    instruction_key: str = "instruction",
    response_key: str = "response",
) -> Dict[str, Any]:
    """Normalize mixed schemas into a strict prompt/reference sample."""
    prompt = item.get("prompt")
    reference = item.get("reference")

    if prompt is None:
        prompt = item.get(instruction_key) or item.get("question") or item.get("input")
    if reference is None:
        reference = item.get(response_key) or item.get("answer") or item.get("output")

    metadata = item.get("metadata")
    if metadata is None:
        metadata = {}

    return {
        "prompt": "" if prompt is None else str(prompt),
        "reference": "" if reference is None else str(reference),
        "metadata": metadata if isinstance(metadata, dict) else {"raw_metadata": metadata},
    }


def validate_rl_schema(
    data: Iterable[Mapping[str, Any]],
    min_prompt_length: int = 3,
    min_reference_length: int = 1,
) -> Tuple[List[Dict[str, Any]], RLSchemaValidation]:
    """Validate prompt/reference schema and return normalized valid samples."""
    data_list = list(data)
    normalized: List[Dict[str, Any]] = []
    seen_pairs = set()
    duplicate_samples = 0
    missing_prompt = 0
    missing_reference = 0

    for raw in data_list:
        sample = normalize_to_rl_sample(raw)
        prompt = sample["prompt"].strip()
        reference = sample["reference"].strip()

        if len(prompt) < min_prompt_length:
            missing_prompt += 1
            continue
        if len(reference) < min_reference_length:
            missing_reference += 1
            continue

        key = (prompt, reference)
        if key in seen_pairs:
            duplicate_samples += 1
            continue
        seen_pairs.add(key)
        sample["prompt"] = prompt
        sample["reference"] = reference
        normalized.append(sample)

    total = len(data_list)
    valid = len(normalized)
    invalid = total - valid

    report = RLSchemaValidation(
        total_samples=total,
        valid_samples=valid,
        invalid_samples=invalid,
        duplicate_samples=duplicate_samples,
        missing_prompt=missing_prompt,
        missing_reference=missing_reference,
        min_prompt_length=min_prompt_length,
        min_reference_length=min_reference_length,
    )
    return normalized, report


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float], mean_value: Optional[float] = None) -> float:
    if not values:
        return 0.0
    m = _mean(values) if mean_value is None else mean_value
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def audit_reward_coverage(
    data: List[Dict[str, Any]],
    reward_config: Mapping[str, Any],
    sample_size: int = 128,
    seed: int = 42,
) -> RewardAuditReport:
    """Audit reward distribution and component-level numeric stability."""
    if not data:
        raise ValueError("Cannot audit reward coverage on empty dataset")

    rng = random.Random(seed)
    if len(data) <= sample_size:
        probe = data[:]
    else:
        probe = rng.sample(data, sample_size)

    total_scores: List[float] = []
    component_values: Dict[str, List[float]] = {}
    has_nan_or_inf = False

    for idx, sample in enumerate(probe):
        # Alternate between near-gold and degraded responses to stress reward dynamic range.
        if idx % 2 == 0:
            response = sample.get("reference", "")
        else:
            response = ""

        out = compute_reward(sample, response, reward_config)
        total = float(out.total_score)
        total_scores.append(total)

        if math.isnan(total) or math.isinf(total):
            has_nan_or_inf = True

        for name, value in out.components.items():
            component_values.setdefault(name, []).append(float(value))
            if math.isnan(value) or math.isinf(value):
                has_nan_or_inf = True

    total_mean = _mean(total_scores)
    total_std = _std(total_scores, total_mean)

    component_stats: Dict[str, Dict[str, float]] = {}
    for name, values in component_values.items():
        mean_v = _mean(values)
        component_stats[name] = {
            "mean": mean_v,
            "std": _std(values, mean_v),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    return RewardAuditReport(
        sample_size=len(probe),
        total_reward_mean=total_mean,
        total_reward_std=total_std,
        total_reward_min=min(total_scores) if total_scores else 0.0,
        total_reward_max=max(total_scores) if total_scores else 0.0,
        component_stats=component_stats,
        has_nan_or_inf=has_nan_or_inf,
        has_nonzero_variance=total_std > 0.0,
    )


def run_grpo_preflight(
    data: List[Dict[str, Any]],
    reward_config: Mapping[str, Any],
    min_prompt_length: int,
    min_reference_length: int,
    sample_size: int,
    seed: int,
    require_nonzero_variance: bool,
    min_reward_std: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate RL schema and reward quality before GRPO training."""
    normalized, schema_report = validate_rl_schema(
        data,
        min_prompt_length=min_prompt_length,
        min_reference_length=min_reference_length,
    )
    if not normalized:
        raise ValueError("GRPO preflight failed: no valid prompt/reference samples")

    audit = audit_reward_coverage(
        normalized,
        reward_config=reward_config,
        sample_size=sample_size,
        seed=seed,
    )

    if audit.has_nan_or_inf:
        raise ValueError("GRPO preflight failed: reward produced NaN/Inf values")

    if require_nonzero_variance and audit.total_reward_std < min_reward_std:
        raise ValueError(
            "GRPO preflight failed: reward variance too low "
            f"(std={audit.total_reward_std:.6f}, required>={min_reward_std:.6f})"
        )

    report = {
        "schema": schema_report.to_dict(),
        "reward_audit": audit.to_dict(),
    }
    return normalized, report


def convert_to_grpo_format(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    instruction_key: str = "instruction",
    response_key: str = "response",
) -> Tuple[Path, Path, Path]:
    """Convert mixed instruction/response or prompt/reference into GRPO files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset(input_path)
    normalized = [
        normalize_to_rl_sample(item, instruction_key=instruction_key, response_key=response_key)
        for item in raw_data
    ]

    # First split train+valid from eval.
    train_valid, eval_data = create_train_val_split(normalized, val_ratio=eval_ratio)
    train_data, valid_data = create_train_val_split(train_valid, val_ratio=val_ratio)

    train_path = output_dir / "rl_train.jsonl"
    valid_path = output_dir / "rl_valid.jsonl"
    eval_path = output_dir / "rl_eval.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(valid_data, valid_path)
    save_jsonl(eval_data, eval_path)

    print(f"Saved {len(train_data)} RL train samples to {train_path}")
    print(f"Saved {len(valid_data)} RL valid samples to {valid_path}")
    print(f"Saved {len(eval_data)} RL eval samples to {eval_path}")

    return train_path, valid_path, eval_path


def save_preflight_report(report: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Persist preflight report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
