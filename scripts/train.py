#!/usr/bin/env python3
"""Training script for MLX LoRA fine-tuning and GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.config import Config
from src.data_utils import load_dataset
from src.grpo_trainer import GRPOKFoldTrainer, GRPOTrainer
from src.model_utils import apply_lora, load_base_model, save_adapters
from src.rl_data import load_rl_dataset, run_grpo_preflight, save_preflight_report
from src.trainer import KFoldTrainer, LoRATrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters with MLX (SFT/KFold/GRPO)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--data", type=str, help="Override SFT data directory")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")
    parser.add_argument(
        "--training-method",
        type=str,
        choices=["basic", "kfold", "grpo", "grpo_kfold"],
        help="Override training method",
    )
    return parser.parse_args()


def create_model_loader(config: Config):
    """Create a function that loads a fresh model with LoRA applied."""

    def load_model():
        model, _ = load_base_model(config.model.name)
        model = apply_lora(
            model,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )
        return model

    return load_model


def _to_sft_examples(samples: List[Dict[str, Any]], template: Optional[str]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for item in samples:
        prompt = str(item.get("prompt", "")).strip()
        reference = str(item.get("reference", "")).strip()
        if not prompt or not reference:
            continue
        if template:
            try:
                text = template.format(instruction=prompt, response=reference)
            except Exception:
                text = f"### Instruction:\n{prompt}\n\n### Response:\n{reference}"
        else:
            text = f"### Instruction:\n{prompt}\n\n### Response:\n{reference}"
        examples.append({"text": text})
    return examples


def _run_post_eval(
    config_path: str,
    eval_file: str,
    pretrain_checkpoint: Path,
    grpo_checkpoint: Path,
    output_dir: Path,
) -> None:
    """Run post-training GRPO evaluation script."""
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        config_path,
        "--eval-file",
        eval_file,
        "--output-dir",
        str(output_dir),
        "--pretrain-checkpoint",
        str(pretrain_checkpoint),
        "--grpo-checkpoint",
        str(grpo_checkpoint),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: post-training evaluation failed ({exc})")


def run_basic_or_kfold(config: Config, training_method: str) -> Dict[str, Any]:
    """Run original SFT basic/kfold training methods."""
    print("\nLoading training data...")
    train_data = load_dataset(config.data.train_file)
    print(f"Loaded {len(train_data)} training examples")

    if training_method == "kfold":
        print(f"\n🔄 Using K-Fold Cross-Validation with {config.data.kfold_splits} folds")

        print("\nLoading tokenizer...")
        _, tokenizer = load_base_model(config.model.name)

        model_loader = create_model_loader(config)

        full_data = train_data
        if Path(config.data.valid_file).exists():
            val_data = load_dataset(config.data.valid_file)
            full_data = train_data + val_data
            print(f"Combined train + val data: {len(full_data)} total examples")

        trainer = KFoldTrainer(
            model_loader=model_loader,
            tokenizer=tokenizer,
            full_data=full_data,
            k=config.data.kfold_splits,
            seed=config.data.kfold_seed,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            num_epochs=config.training.num_epochs,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            max_seq_length=config.model.max_seq_length,
            save_steps=config.training.save_steps,
            eval_steps=config.training.eval_steps,
            logging_steps=config.training.logging_steps,
            output_dir=config.output.dir,
            model_name=config.model.name,
            lora_config={
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
            },
        )
        return trainer.train()

    print("\nLoading model...")
    model, tokenizer = load_base_model(config.model.name)
    print("\nApplying LoRA adapters...")
    model = apply_lora(
        model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    val_data = None
    if Path(config.data.valid_file).exists():
        val_data = load_dataset(config.data.valid_file)
        print(f"Loaded {len(val_data)} validation examples")

    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_seq_length=config.model.max_seq_length,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.output.dir,
        model_name=config.model.name,
        lora_config={
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
        },
    )
    return trainer.train()


def run_grpo(config: Config, config_path: str) -> Dict[str, Any]:
    """Run GRPO single-run training with mandatory preflight + warmup + post-eval."""
    rl_train = load_rl_dataset(config.data.rl_train_file)
    rl_valid = load_rl_dataset(config.data.rl_valid_file) if Path(config.data.rl_valid_file).exists() else []

    preflight_data, preflight_report = run_grpo_preflight(
        data=rl_train,
        reward_config=config.reward.to_dict(),
        min_prompt_length=3,
        min_reference_length=1,
        sample_size=config.grpo.preflight_sample_size,
        seed=config.data.kfold_seed,
        require_nonzero_variance=config.reward.require_nonzero_variance,
        min_reward_std=config.grpo.min_reward_std,
    )
    logs_dir = Path(config.output.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_preflight_report(preflight_report, logs_dir / "grpo_preflight.json")
    print("✅ GRPO preflight passed")

    print("\nLoading model for GRPO...")
    model, tokenizer = load_base_model(config.model.name)
    model = apply_lora(
        model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    # Warmup SFT stage.
    if config.grpo.warmup_epochs > 0:
        print(f"\n🔥 Running SFT warmup for {config.grpo.warmup_epochs} epoch(s)")
        warmup_trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=_to_sft_examples(preflight_data, config.data.prompt_template),
            val_data=_to_sft_examples(rl_valid, config.data.prompt_template),
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            num_epochs=config.grpo.warmup_epochs,
            warmup_steps=0,
            weight_decay=0.0,
            max_seq_length=config.model.max_seq_length,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            save_steps=10_000_000,
            eval_steps=10_000_000,
            logging_steps=10_000_000,
            output_dir=config.output.dir,
            model_name=config.model.name,
            lora_config={"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout},
        )
        warmup_trainer.train()
        warmup_trainer._save_checkpoint("pretrain")
    else:
        pretrain_dir = Path(config.output.dir) / "checkpoints" / "pretrain"
        save_adapters(model, pretrain_dir)

    pretrain_checkpoint = Path(config.output.dir) / "checkpoints" / "pretrain"

    # Build frozen reference policy from warmup checkpoint.
    reference_model, _ = load_base_model(config.model.name)
    reference_model = apply_lora(
        reference_model,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )
    pretrain_adapters = mx.load(str(pretrain_checkpoint / "adapters.safetensors"))
    reference_model.load_weights(list(pretrain_adapters.items()), strict=False)
    model.load_weights(list(pretrain_adapters.items()), strict=False)

    trainer = GRPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        train_data=preflight_data,
        val_data=rl_valid,
        reward_config=config.reward.to_dict(),
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        group_size=config.grpo.group_size,
        clip_epsilon=config.grpo.clip_epsilon,
        beta_kl=config.grpo.beta_kl,
        advantage_epsilon=config.grpo.advantage_epsilon,
        max_seq_length=config.model.max_seq_length,
        max_generation_tokens=config.grpo.max_generation_tokens,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        save_steps=config.training.save_steps,
        eval_steps=config.grpo.eval_steps,
        logging_steps=config.grpo.logging_steps,
        output_dir=config.output.dir,
        prompt_template=config.data.prompt_template,
        model_name=config.model.name,
        lora_config={"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout},
    )

    stats = trainer.train()

    eval_file = config.data.rl_eval_file if Path(config.data.rl_eval_file).exists() else config.data.rl_train_file
    _run_post_eval(
        config_path=config_path,
        eval_file=eval_file,
        pretrain_checkpoint=pretrain_checkpoint,
        grpo_checkpoint=Path(config.output.dir) / "checkpoints" / "final",
        output_dir=Path(config.output.dir) / "evals",
    )
    return stats


def run_grpo_kfold(config: Config, config_path: str) -> Dict[str, Any]:
    """Run GRPO K-Fold training with warmup per fold and automatic post-eval."""
    rl_train = load_rl_dataset(config.data.rl_train_file)
    rl_valid = load_rl_dataset(config.data.rl_valid_file) if Path(config.data.rl_valid_file).exists() else []
    full_data = rl_train + rl_valid

    normalized_data, preflight_report = run_grpo_preflight(
        data=full_data,
        reward_config=config.reward.to_dict(),
        min_prompt_length=3,
        min_reference_length=1,
        sample_size=config.grpo.preflight_sample_size,
        seed=config.data.kfold_seed,
        require_nonzero_variance=config.reward.require_nonzero_variance,
        min_reward_std=config.grpo.min_reward_std,
    )
    logs_dir = Path(config.output.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_preflight_report(preflight_report, logs_dir / "grpo_kfold_preflight.json")
    print("✅ GRPO K-Fold preflight passed")

    # Load tokenizer once.
    _, tokenizer = load_base_model(config.model.name)

    model_loader = create_model_loader(config)
    reference_model_loader = create_model_loader(config)

    trainer = GRPOKFoldTrainer(
        model_loader=model_loader,
        reference_model_loader=reference_model_loader,
        tokenizer=tokenizer,
        full_data=normalized_data,
        reward_config=config.reward.to_dict(),
        k=config.data.kfold_splits,
        seed=config.data.kfold_seed,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        group_size=config.grpo.group_size,
        clip_epsilon=config.grpo.clip_epsilon,
        beta_kl=config.grpo.beta_kl,
        advantage_epsilon=config.grpo.advantage_epsilon,
        max_seq_length=config.model.max_seq_length,
        max_generation_tokens=config.grpo.max_generation_tokens,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        save_steps=config.training.save_steps,
        eval_steps=config.grpo.eval_steps,
        logging_steps=config.grpo.logging_steps,
        output_dir=config.output.dir,
        prompt_template=config.data.prompt_template,
        model_name=config.model.name,
        lora_config={"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout},
        warmup_epochs=config.grpo.warmup_epochs,
        warmup_learning_rate=config.training.learning_rate,
    )

    summary = trainer.train()

    best_fold = summary["best_fold"]
    fold_dir = Path(config.output.dir) / f"fold_{best_fold}" / "checkpoints"
    eval_file = config.data.rl_eval_file if Path(config.data.rl_eval_file).exists() else config.data.rl_train_file

    _run_post_eval(
        config_path=config_path,
        eval_file=eval_file,
        pretrain_checkpoint=fold_dir / "pretrain",
        grpo_checkpoint=fold_dir / "final",
        output_dir=Path(config.output.dir) / "evals",
    )
    return summary


def main() -> None:
    args = parse_args()

    config = Config.from_yaml(args.config)

    if args.model:
        config.model.name = args.model
    if args.data:
        config.data.train_file = f"{args.data}/train.jsonl"
        config.data.valid_file = f"{args.data}/valid.jsonl"
        config.data.rl_train_file = f"{args.data}/rl_train.jsonl"
        config.data.rl_valid_file = f"{args.data}/rl_valid.jsonl"
        config.data.rl_eval_file = f"{args.data}/rl_eval.jsonl"
    if args.output:
        config.output.dir = args.output
        config.output.adapters_dir = f"{args.output}/adapters"
        config.output.checkpoints_dir = f"{args.output}/checkpoints"
        config.output.logs_dir = f"{args.output}/logs"
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.lora_rank:
        config.lora.rank = args.lora_rank
    if args.training_method:
        config.data.training_method = args.training_method

    training_method = getattr(config.data, "training_method", "basic")

    print("=" * 70)
    print("MLX LoRA + GRPO Training")
    print("=" * 70)
    print(f"Model: {config.model.name}")
    print(f"LoRA rank: {config.lora.rank}, alpha: {config.lora.alpha}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Training method: {training_method}")
    print("=" * 70)

    config.output.ensure_dirs()

    if training_method in {"basic", "kfold"}:
        stats = run_basic_or_kfold(config, training_method)
    elif training_method == "grpo":
        stats = run_grpo(config, args.config)
    elif training_method == "grpo_kfold":
        stats = run_grpo_kfold(config, args.config)
    else:
        raise ValueError(f"Unknown training method: {training_method}")

    print("\n" + "=" * 70)
    print("Training complete")
    print(json.dumps(stats, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
