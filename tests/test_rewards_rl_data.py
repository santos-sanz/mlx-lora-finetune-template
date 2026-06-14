"""Tests for reward functions and GRPO data preflight helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.rewards import compute_reward
from src.rl_data import (
    audit_reward_coverage,
    convert_to_grpo_format,
    run_grpo_preflight,
    validate_rl_schema,
)


def test_compute_reward_exact_match():
    sample = {"prompt": "2+2", "reference": "4", "metadata": {}}
    config = {"function": "exact_match", "pass_threshold": 0.5}
    out = compute_reward(sample, "4", config)
    assert out.total_score == 1.0
    assert out.passed


def test_compute_reward_weighted_rules_with_keywords():
    sample = {
        "prompt": "Give two words",
        "reference": "alpha beta",
        "metadata": {"keywords": ["alpha", "beta"]},
    }
    config = {
        "function": "weighted_rules",
        "weights": {
            "keyword_coverage": 1.0,
        },
        "metadata_keyword_field": "keywords",
        "pass_threshold": 0.5,
    }
    out = compute_reward(sample, "alpha beta", config)
    assert out.total_score == 1.0
    assert out.components["keyword_coverage"] == 1.0


def test_compute_reward_json_format():
    sample = {"prompt": "Return JSON", "reference": "{}", "metadata": {}}
    out_good = compute_reward(sample, '{"a": 1}', {"function": "json_format", "pass_threshold": 0.5})
    out_bad = compute_reward(sample, "not-json", {"function": "json_format", "pass_threshold": 0.5})
    assert out_good.total_score == 1.0
    assert out_bad.total_score == 0.0


def test_validate_rl_schema_filters_invalid_and_duplicates():
    data = [
        {"prompt": "prompt-1", "reference": "r1"},
        {"prompt": "prompt-1", "reference": "r1"},  # duplicate
        {"prompt": "", "reference": "r2"},    # invalid
    ]
    normalized, report = validate_rl_schema(data)
    assert len(normalized) == 1
    assert report.duplicate_samples == 1
    assert report.invalid_samples == 2


def test_audit_reward_coverage_has_variance():
    data = [
        {"prompt": f"p{i}", "reference": f"answer {i}", "metadata": {"keywords": ["answer"]}}
        for i in range(20)
    ]
    audit = audit_reward_coverage(data, reward_config={"function": "weighted_rules"}, sample_size=10)
    assert audit.sample_size == 10
    assert audit.total_reward_std >= 0.0


def test_run_grpo_preflight_returns_report():
    data = [
        {"prompt": f"prompt {i}", "reference": f"reference {i}", "metadata": {"keywords": [f"reference {i}"]}}
        for i in range(20)
    ]
    normalized, report = run_grpo_preflight(
        data=data,
        reward_config={
            "function": "weighted_rules",
            "weights": {"exact_match": 1.0},
        },
        min_prompt_length=3,
        min_reference_length=1,
        sample_size=10,
        seed=42,
        require_nonzero_variance=False,
        min_reward_std=1e-8,
    )
    assert len(normalized) == 20
    assert "schema" in report
    assert "reward_audit" in report


def test_convert_to_grpo_format_creates_three_files():
    data = [{"instruction": f"Q{i}", "response": f"A{i}"} for i in range(40)]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.json"
        output_dir = Path(tmpdir) / "out"

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        train_path, valid_path, eval_path = convert_to_grpo_format(input_path, output_dir)

        assert train_path.exists()
        assert valid_path.exists()
        assert eval_path.exists()
