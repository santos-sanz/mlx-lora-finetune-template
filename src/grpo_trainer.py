"""GRPO training engine built on top of MLX LoRA models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.data_utils import create_kfold_splits, get_kfold_data
from src.param_utils import flatten_params
from src.rewards import compute_reward
from src.trainer import LoRATrainer


@dataclass
class GRPOStepMetrics:
    """Container for GRPO step metrics."""

    step: int
    policy_loss: float
    kl_loss: float
    total_loss: float
    mean_reward: float
    std_reward: float
    learning_rate: float
    tokens_per_second: float
    elapsed_time: float


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for LoRA models."""

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        tokenizer: Any,
        train_data: List[Dict[str, Any]],
        reward_config: Mapping[str, Any],
        val_data: Optional[List[Dict[str, Any]]] = None,
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        num_epochs: int = 1,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.02,
        advantage_epsilon: float = 1e-8,
        max_seq_length: int = 2048,
        max_generation_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 50,
        logging_steps: int = 10,
        output_dir: Union[str, Path] = "outputs",
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data or []
        self.reward_config = dict(reward_config)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl
        self.advantage_epsilon = advantage_epsilon

        self.max_seq_length = max_seq_length
        self.max_generation_tokens = max_generation_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

        self.prompt_template = prompt_template
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.callbacks = callbacks or {}

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "training_log.jsonl"

        self.global_step = 0
        self.epoch = 0
        self.best_eval_reward = float("-inf")

    @staticmethod
    def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
        """Compute normalized relative advantages for a reward group."""
        if not rewards:
            return []
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        denom = std + eps
        return [(r - mean) / denom for r in rewards]

    @staticmethod
    def clipped_policy_objective(ratio: float, advantage: float, clip_epsilon: float) -> float:
        """PPO-style clipped policy objective used by GRPO."""
        clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
        return min(ratio * advantage, clipped * advantage)

    def _write_log_entry(self, entry: Dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _batch_iterate(self, data: List[Dict[str, Any]]):
        for i in range(0, len(data), self.batch_size):
            yield data[i:i + self.batch_size]

    def _format_prompt(self, prompt: str) -> str:
        if not self.prompt_template:
            return prompt
        template = self.prompt_template
        if "{response}" in template:
            template = template.split("{response}")[0]
        try:
            return template.format(instruction=prompt)
        except Exception:
            return prompt

    def _safe_generate(self, prompt: str) -> str:
        from mlx_lm import generate

        kwargs = {
            "prompt": prompt,
            "max_tokens": self.max_generation_tokens,
            "verbose": False,
        }
        try:
            return generate(
                self.model,
                self.tokenizer,
                temperature=self.temperature,
                top_p=self.top_p,
                **kwargs,
            )
        except TypeError:
            return generate(self.model, self.tokenizer, **kwargs)

    def _tokenize_completion_window(self, prompt: str, response: str) -> Tuple[List[int], int]:
        formatted_prompt = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        full_tokens = self.tokenizer.encode(formatted_prompt + response)

        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[: self.max_seq_length]
        prompt_len = min(len(prompt_tokens), max(0, len(full_tokens) - 1))
        return full_tokens, prompt_len

    def _sequence_logprob_and_kl(
        self,
        model: nn.Module,
        prompt: str,
        response: str,
    ) -> Tuple[mx.array, mx.array]:
        full_tokens, prompt_len = self._tokenize_completion_window(prompt, response)
        if len(full_tokens) < 2 or prompt_len >= len(full_tokens) - 1:
            return mx.array(0.0), mx.array(0.0)

        input_ids = mx.array(full_tokens[:-1])[None, :]
        targets = mx.array(full_tokens[1:])[None, :]

        logits = model(input_ids)

        start = max(prompt_len - 1, 0)
        completion_logits = logits[:, start:, :]
        completion_targets = targets[:, start:]

        if completion_targets.size == 0:
            return mx.array(0.0), mx.array(0.0)

        vocab_size = completion_logits.shape[-1]
        ce = nn.losses.cross_entropy(
            completion_logits.reshape(-1, vocab_size),
            completion_targets.reshape(-1),
        )
        logprob = -ce.sum()

        kl = mx.array(0.0)
        if self.reference_model is not None:
            ref_logits = self.reference_model(input_ids)[:, start:, :]
            current_log_probs = completion_logits - mx.logsumexp(completion_logits, axis=-1, keepdims=True)
            ref_log_probs = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
            p_current = mx.exp(current_log_probs)
            kl = (p_current * (current_log_probs - ref_log_probs)).sum(axis=-1).mean()

        return logprob, kl

    def _prepare_trajectories(self, batch: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float, float, int]:
        trajectories: List[Dict[str, Any]] = []
        all_rewards: List[float] = []
        total_tokens = 0

        for sample in batch:
            prompt = str(sample.get("prompt", "")).strip()
            if not prompt:
                continue

            responses = [self._safe_generate(self._format_prompt(prompt)) for _ in range(self.group_size)]
            rewards = [compute_reward(sample, r, self.reward_config).total_score for r in responses]
            advantages = self.compute_group_advantages(rewards, eps=self.advantage_epsilon)

            for response, reward, advantage in zip(responses, rewards, advantages):
                old_logprob, _ = self._sequence_logprob_and_kl(self.model, prompt, response)
                old_logprob_value = float(old_logprob.item())
                trajectories.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "old_logprob": old_logprob_value,
                        "advantage": float(advantage),
                        "reward": float(reward),
                    }
                )
                all_rewards.append(float(reward))

                full_tokens, prompt_len = self._tokenize_completion_window(prompt, response)
                completion_tokens = max(0, len(full_tokens) - prompt_len)
                total_tokens += completion_tokens

        if not all_rewards:
            return trajectories, 0.0, 0.0, total_tokens

        mean_r = sum(all_rewards) / len(all_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
        return trajectories, mean_r, std_r, total_tokens

    def _evaluate_mean_reward(self, data: List[Dict[str, Any]]) -> float:
        if not data:
            return 0.0

        scores: List[float] = []
        for sample in data:
            prompt = str(sample.get("prompt", "")).strip()
            if not prompt:
                continue
            response = self._safe_generate(self._format_prompt(prompt))
            out = compute_reward(sample, response, self.reward_config)
            scores.append(out.total_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _log_step(self, metrics: GRPOStepMetrics) -> None:
        self._write_log_entry(
            {
                "type": "grpo_step",
                "step": metrics.step,
                "epoch": self.epoch,
                "policy_loss": metrics.policy_loss,
                "kl_loss": metrics.kl_loss,
                "total_loss": metrics.total_loss,
                "mean_reward": metrics.mean_reward,
                "std_reward": metrics.std_reward,
                "learning_rate": metrics.learning_rate,
                "tokens_per_second": metrics.tokens_per_second,
                "elapsed_time": metrics.elapsed_time,
            }
        )

    def _save_checkpoint(self, name: str) -> None:
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainable_params = flatten_params(self.model.trainable_parameters())
        mx.save_safetensors(str(checkpoint_dir / "adapters.safetensors"), trainable_params)

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_reward": self.best_eval_reward,
            "beta_kl": self.beta_kl,
            "group_size": self.group_size,
        }
        with open(checkpoint_dir / "grpo_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def train(self) -> Dict[str, Any]:
        """Run GRPO training loop."""
        if not self.train_data:
            raise ValueError("GRPO training data is empty")

        self._write_log_entry(
            {
                "type": "grpo_start",
                "model_name": self.model_name,
                "lora_config": self.lora_config,
                "grpo_config": {
                    "group_size": self.group_size,
                    "clip_epsilon": self.clip_epsilon,
                    "beta_kl": self.beta_kl,
                    "max_generation_tokens": self.max_generation_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
                "reward_config": self.reward_config,
                "train_samples": len(self.train_data),
                "val_samples": len(self.val_data),
            }
        )

        optimizer = optim.AdamW(learning_rate=self.learning_rate, weight_decay=0.0)

        def loss_components(model, trajectories):
            policy_total = mx.array(0.0)
            kl_total = mx.array(0.0)

            for traj in trajectories:
                current_logprob, kl_value = self._sequence_logprob_and_kl(
                    model,
                    traj["prompt"],
                    traj["response"],
                )
                old_logprob = mx.array(traj["old_logprob"])
                advantage = mx.array(traj["advantage"])

                ratio = mx.exp(current_logprob - old_logprob)
                clipped_ratio = mx.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                policy_obj = mx.minimum(ratio * advantage, clipped_ratio * advantage)

                policy_total = policy_total - policy_obj
                kl_total = kl_total + kl_value

            denom = max(1, len(trajectories))
            policy_mean = policy_total / denom
            kl_mean = kl_total / denom
            total = policy_mean + self.beta_kl * kl_mean
            return policy_mean, kl_mean, total

        def loss_fn(model, trajectories):
            _, _, total = loss_components(model, trajectories)
            return total

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        start_time = time.time()
        total_tokens = 0
        last_policy_loss = 0.0
        last_kl_loss = 0.0
        last_total_loss = 0.0

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            for batch in self._batch_iterate(self.train_data):
                trajectories, mean_reward, std_reward, batch_tokens = self._prepare_trajectories(batch)
                if not trajectories:
                    continue

                loss, grads = loss_and_grad(self.model, trajectories)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)

                self.global_step += 1
                total_tokens += batch_tokens

                # Compute detailed metrics for observability.
                policy_loss, kl_loss, total_loss = loss_components(self.model, trajectories)
                last_policy_loss = float(policy_loss.item())
                last_kl_loss = float(kl_loss.item())
                last_total_loss = float(total_loss.item())

                if self.global_step % self.logging_steps == 0:
                    elapsed = time.time() - start_time
                    metrics = GRPOStepMetrics(
                        step=self.global_step,
                        policy_loss=last_policy_loss,
                        kl_loss=last_kl_loss,
                        total_loss=last_total_loss,
                        mean_reward=mean_reward,
                        std_reward=std_reward,
                        learning_rate=float(optimizer.learning_rate),
                        tokens_per_second=(total_tokens / elapsed) if elapsed > 0 else 0.0,
                        elapsed_time=elapsed,
                    )
                    self._log_step(metrics)

                if self.val_data and self.global_step % self.eval_steps == 0:
                    eval_reward = self._evaluate_mean_reward(self.val_data)
                    self._write_log_entry(
                        {
                            "type": "grpo_eval",
                            "step": self.global_step,
                            "epoch": self.epoch,
                            "mean_reward": eval_reward,
                            "elapsed_time": time.time() - start_time,
                        }
                    )
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        self._save_checkpoint("best")

                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"step-{self.global_step}")

        self._save_checkpoint("final")

        total_time = time.time() - start_time
        final_eval_reward = self._evaluate_mean_reward(self.val_data) if self.val_data else None
        if final_eval_reward is not None and final_eval_reward > self.best_eval_reward:
            self.best_eval_reward = final_eval_reward

        summary = {
            "total_steps": self.global_step,
            "total_time": total_time,
            "final_total_loss": last_total_loss,
            "final_policy_loss": last_policy_loss,
            "final_kl_loss": last_kl_loss,
            "best_eval_reward": None if self.best_eval_reward == float("-inf") else self.best_eval_reward,
        }

        self._write_log_entry({"type": "grpo_end", **summary})
        return summary


class GRPOKFoldTrainer:
    """K-Fold wrapper that trains a fresh GRPO model per fold."""

    def __init__(
        self,
        model_loader: Callable[[], nn.Module],
        reference_model_loader: Callable[[], nn.Module],
        tokenizer: Any,
        full_data: List[Dict[str, Any]],
        reward_config: Mapping[str, Any],
        k: int = 5,
        seed: int = 42,
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        num_epochs: int = 1,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.02,
        advantage_epsilon: float = 1e-8,
        max_seq_length: int = 2048,
        max_generation_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 50,
        logging_steps: int = 10,
        output_dir: Union[str, Path] = "outputs",
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 1,
        warmup_learning_rate: Optional[float] = None,
    ):
        self.model_loader = model_loader
        self.reference_model_loader = reference_model_loader
        self.tokenizer = tokenizer
        self.full_data = full_data
        self.reward_config = dict(reward_config)

        self.k = k
        self.seed = seed

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl
        self.advantage_epsilon = advantage_epsilon

        self.max_seq_length = max_seq_length
        self.max_generation_tokens = max_generation_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps

        self.prompt_template = prompt_template
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.warmup_epochs = warmup_epochs
        self.warmup_learning_rate = warmup_learning_rate

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "training_log.jsonl"

    def _to_sft_examples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert RL prompt/reference samples into SFT text examples."""
        examples: List[Dict[str, str]] = []
        for item in samples:
            prompt = str(item.get("prompt", "")).strip()
            reference = str(item.get("reference", "")).strip()
            if not prompt or not reference:
                continue
            if self.prompt_template:
                try:
                    text = self.prompt_template.format(instruction=prompt, response=reference)
                except Exception:
                    text = f"### Instruction:\n{prompt}\n\n### Response:\n{reference}"
            else:
                text = f"### Instruction:\n{prompt}\n\n### Response:\n{reference}"
            examples.append({"text": text})
        return examples

    def _write_log_entry(self, entry: Dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def train(self) -> Dict[str, Any]:
        if not self.full_data:
            raise ValueError("GRPO K-Fold requires non-empty data")

        splits = create_kfold_splits(self.full_data, k=self.k, seed=self.seed)
        self._write_log_entry(
            {
                "type": "grpo_kfold_start",
                "k": self.k,
                "total_samples": len(self.full_data),
                "reward_config": self.reward_config,
            }
        )

        fold_results: List[Dict[str, Any]] = []
        best_fold = 0
        best_reward = float("-inf")
        start_time = time.time()

        for fold_idx in range(self.k):
            train_data, val_data = get_kfold_data(self.full_data, splits, fold_idx)
            self._write_log_entry(
                {
                    "type": "grpo_fold_start",
                    "fold": fold_idx,
                    "total_folds": self.k,
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                }
            )

            model = self.model_loader()
            reference_model = self.reference_model_loader()

            fold_output_dir = self.output_dir / f"fold_{fold_idx}"

            if self.warmup_epochs > 0:
                warmup_train = self._to_sft_examples(train_data)
                warmup_val = self._to_sft_examples(val_data)
                warmup_trainer = LoRATrainer(
                    model=model,
                    tokenizer=self.tokenizer,
                    train_data=warmup_train,
                    val_data=warmup_val,
                    learning_rate=self.warmup_learning_rate or self.learning_rate,
                    batch_size=self.batch_size,
                    num_epochs=self.warmup_epochs,
                    warmup_steps=0,
                    weight_decay=0.0,
                    max_seq_length=self.max_seq_length,
                    save_steps=10_000_000,
                    eval_steps=10_000_000,
                    logging_steps=10_000_000,
                    output_dir=fold_output_dir,
                    model_name=self.model_name,
                    lora_config=self.lora_config,
                )
                warmup_trainer.log_file = self.log_file
                warmup_trainer.train()
                warmup_trainer._save_checkpoint("pretrain")

                pretrain_adapters = mx.load(str(fold_output_dir / "checkpoints" / "pretrain" / "adapters.safetensors"))
                reference_model.load_weights(list(pretrain_adapters.items()), strict=False)
                model.load_weights(list(pretrain_adapters.items()), strict=False)

            trainer = GRPOTrainer(
                model=model,
                reference_model=reference_model,
                tokenizer=self.tokenizer,
                train_data=train_data,
                val_data=val_data,
                reward_config=self.reward_config,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                group_size=self.group_size,
                clip_epsilon=self.clip_epsilon,
                beta_kl=self.beta_kl,
                advantage_epsilon=self.advantage_epsilon,
                max_seq_length=self.max_seq_length,
                max_generation_tokens=self.max_generation_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                save_steps=self.save_steps,
                eval_steps=self.eval_steps,
                logging_steps=self.logging_steps,
                output_dir=fold_output_dir,
                prompt_template=self.prompt_template,
                model_name=self.model_name,
                lora_config=self.lora_config,
            )
            trainer.log_file = self.log_file

            result = trainer.train()
            result["fold"] = fold_idx
            result["train_samples"] = len(train_data)
            result["val_samples"] = len(val_data)
            fold_results.append(result)

            fold_reward = result.get("best_eval_reward")
            if fold_reward is not None and fold_reward > best_reward:
                best_reward = fold_reward
                best_fold = fold_idx

            self._write_log_entry(
                {
                    "type": "grpo_fold_end",
                    "fold": fold_idx,
                    "total_folds": self.k,
                    "fold_result": result,
                }
            )

        rewards = [r.get("best_eval_reward", 0.0) or 0.0 for r in fold_results]
        avg_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

        summary = {
            "k": self.k,
            "total_time": time.time() - start_time,
            "avg_best_reward": avg_reward,
            "std_best_reward": std_reward,
            "best_fold": best_fold,
            "best_reward": best_reward,
            "fold_results": fold_results,
        }

        with open(self.output_dir / "grpo_kfold_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._write_log_entry({"type": "grpo_kfold_end", **summary})
        return summary
