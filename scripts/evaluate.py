#!/usr/bin/env python3
"""Post-training evaluation for GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.config import Config
from src.model_utils import apply_lora, load_base_model
from src.rewards import compute_reward
from src.rl_data import load_rl_dataset, validate_rl_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/pretrain/grpo models with reward metrics")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--eval-file", type=str, default=None, help="Optional override for RL eval file")
    parser.add_argument("--output-dir", type=str, default="outputs/evals", help="Directory for evaluation artifacts")
    parser.add_argument("--pretrain-checkpoint", type=str, default="outputs/checkpoints/pretrain", help="Pretrain checkpoint dir")
    parser.add_argument("--grpo-checkpoint", type=str, default="outputs/checkpoints/final", help="GRPO final checkpoint dir")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override generation max tokens")
    return parser.parse_args()


def _format_prompt(prompt: str, template: Optional[str]) -> str:
    if not template:
        return prompt
    partial = template
    if "{response}" in partial:
        partial = partial.split("{response}")[0]
    try:
        return partial.format(instruction=prompt)
    except Exception:
        return prompt


def _safe_generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    from mlx_lm import generate

    kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "verbose": False,
    }
    try:
        return generate(model, tokenizer, temperature=temperature, top_p=top_p, **kwargs)
    except TypeError:
        return generate(model, tokenizer, **kwargs)


def _load_variant_model(config: Config, checkpoint_dir: Optional[Path]) -> tuple[Any, Any]:
    model, tokenizer = load_base_model(config.model.name)
    if checkpoint_dir is None:
        return model, tokenizer

    adapter_file = checkpoint_dir / "adapters.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Missing adapters file: {adapter_file}")

    model = apply_lora(
        model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=0.0,
        target_modules=config.lora.target_modules,
    )

    adapters = mx.load(str(adapter_file))
    model.load_weights(list(adapters.items()), strict=False)
    return model, tokenizer


def evaluate_variant(
    variant_name: str,
    config: Config,
    dataset: List[Dict[str, Any]],
    reward_config: Mapping[str, Any],
    checkpoint_dir: Optional[Path],
    max_tokens: int,
) -> Dict[str, Any]:
    model, tokenizer = _load_variant_model(config, checkpoint_dir)

    scores: List[float] = []
    component_values: Dict[str, List[float]] = {}
    empty_responses = 0
    json_failures = 0
    samples: List[Dict[str, Any]] = []

    for sample in dataset:
        prompt = sample["prompt"]
        prompt_text = _format_prompt(prompt, config.data.prompt_template)
        response = _safe_generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            temperature=config.grpo.temperature,
            top_p=config.grpo.top_p,
        )

        reward = compute_reward(sample, response, reward_config)
        scores.append(reward.total_score)

        if not response.strip():
            empty_responses += 1

        json_component = reward.components.get("json_format")
        if json_component is not None and json_component < 1.0:
            json_failures += 1

        for name, value in reward.components.items():
            component_values.setdefault(name, []).append(float(value))

        samples.append(
            {
                "variant": variant_name,
                "prompt": prompt,
                "reference": sample.get("reference", ""),
                "response": response,
                "reward": reward.total_score,
                "reward_components": reward.components,
                "passed": reward.passed,
            }
        )

    avg = sum(scores) / len(scores) if scores else 0.0
    std = (sum((s - avg) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0.0

    component_means = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in component_values.items()
    }

    return {
        "variant": variant_name,
        "num_samples": len(dataset),
        "avg_reward": avg,
        "std_reward": std,
        "pass_rate": (sum(1 for s in scores if s >= reward_config.get("pass_threshold", 0.6)) / len(scores)) if scores else 0.0,
        "empty_response_rate": (empty_responses / len(dataset)) if dataset else 0.0,
        "json_error_rate": (json_failures / len(dataset)) if dataset else 0.0,
        "component_means": component_means,
        "samples": samples,
    }


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    config = Config.from_yaml(args.config)
    eval_file = Path(args.eval_file) if args.eval_file else Path(config.data.rl_eval_file)

    raw_eval = load_rl_dataset(eval_file)
    eval_data, schema_report = validate_rl_schema(raw_eval)
    if not eval_data:
        raise ValueError("Evaluation dataset contains no valid prompt/reference samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_config = config.reward.to_dict()
    max_tokens = args.max_tokens or config.grpo.max_generation_tokens

    base_result = evaluate_variant(
        "base",
        config,
        eval_data,
        reward_config,
        checkpoint_dir=None,
        max_tokens=max_tokens,
    )

    pretrain_cp = Path(args.pretrain_checkpoint)
    pretrain_result = None
    if pretrain_cp.exists():
        pretrain_result = evaluate_variant(
            "pretrain",
            config,
            eval_data,
            reward_config,
            checkpoint_dir=pretrain_cp,
            max_tokens=max_tokens,
        )

    grpo_cp = Path(args.grpo_checkpoint)
    grpo_result = None
    if grpo_cp.exists():
        grpo_result = evaluate_variant(
            "grpo_final",
            config,
            eval_data,
            reward_config,
            checkpoint_dir=grpo_cp,
            max_tokens=max_tokens,
        )

    all_variants = [r for r in [base_result, pretrain_result, grpo_result] if r is not None]

    def _win_rate(target: Dict[str, Any], baseline: Dict[str, Any]) -> float:
        t = [s["reward"] for s in target["samples"]]
        b = [s["reward"] for s in baseline["samples"]]
        if not t or not b:
            return 0.0
        wins = sum(1 for x, y in zip(t, b) if x > y)
        return wins / min(len(t), len(b))

    summary = {
        "schema": schema_report.to_dict(),
        "variants": [
            {k: v for k, v in variant.items() if k != "samples"}
            for variant in all_variants
        ],
    }

    if grpo_result is not None:
        summary["grpo_vs_base_win_rate"] = _win_rate(grpo_result, base_result)
        if pretrain_result is not None:
            summary["grpo_vs_pretrain_win_rate"] = _win_rate(grpo_result, pretrain_result)

    summary_path = output_dir / "grpo_eval_summary.json"
    samples_path = output_dir / "grpo_eval_samples.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(samples_path, "w", encoding="utf-8") as f:
        for variant in all_variants:
            for sample in variant["samples"]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Evaluation summary saved to: {summary_path}")
    print(f"Evaluation samples saved to: {samples_path}")

    return summary


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
