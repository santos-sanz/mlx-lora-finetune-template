"""
Training engine for LoRA fine-tuning with MLX.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import json
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    learning_rate: float
    tokens_per_second: float
    elapsed_time: float


class LoRATrainer:
    """
    Main trainer class for LoRA fine-tuning with MLX.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_data: list,
        val_data: Optional[list] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        output_dir: Union[str, Path] = "outputs",
        callbacks: Optional[Dict[str, Callable]] = None,
        model_name: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logs directory
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "training_log.jsonl"
        
        self.callbacks = callbacks or {}
        
        # Model info for logging
        self.model_name = model_name
        self.lora_config = lora_config or {}
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_log = []
    
    def _get_lr(self, step: int) -> float:
        """Calculate learning rate with warmup."""
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        return self.learning_rate
    
    def _batch_iterate(self, data: list):
        """Iterate over data in batches."""
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            yield batch
    
    def _compute_loss(self, batch: list) -> mx.array:
        """Compute loss for a batch."""
        total_loss = mx.array(0.0)
        num_tokens = 0
        
        for item in batch:
            text = item.get("text", "")
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            
            if len(tokens) < 2:
                continue
            
            input_ids = mx.array(tokens[:-1])[None, :]
            targets = mx.array(tokens[1:])[None, :]
            
            logits = self.model(input_ids)
            
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            total_loss = total_loss + loss.sum()
            num_tokens += targets.size
        
        if num_tokens == 0:
            return mx.array(0.0)
        
        return total_loss / num_tokens
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Returns:
            Training statistics
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Training samples: {len(self.train_data)}")
        if self.val_data:
            print(f"Validation samples: {len(self.val_data)}")
        
        # Log training start with model info
        self._write_log_entry({
            "type": "train_start",
            "model_name": self.model_name,
            "lora_config": self.lora_config,
            "training_config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.weight_decay,
                "max_seq_length": self.max_seq_length,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "train_samples": len(self.train_data),
            "val_samples": len(self.val_data) if self.val_data else 0,
        })
        
        # Setup optimizer
        optimizer = optim.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Get trainable parameters
        trainable = self.model.trainable_parameters()
        
        # Loss and grad function
        def loss_fn(model, batch):
            self.model = model
            return self._compute_loss(batch)
        
        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        
        start_time = time.time()
        total_tokens = 0
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self._batch_iterate(self.train_data):
                # Update learning rate
                lr = self._get_lr(self.global_step)
                optimizer.learning_rate = lr
                
                # Forward and backward pass
                loss, grads = loss_and_grad(self.model, batch)
                
                # Update weights
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    elapsed = time.time() - start_time
                    metrics = TrainingMetrics(
                        step=self.global_step,
                        loss=loss.item(),
                        learning_rate=lr,
                        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0,
                        elapsed_time=elapsed,
                    )
                    self._log_metrics(metrics)
                
                # Evaluation
                if self.val_data and self.global_step % self.eval_steps == 0:
                    val_loss = self._evaluate()
                    print(f"Step {self.global_step} - Validation loss: {val_loss:.4f}")
                    
                    # Log validation result
                    self._write_log_entry({
                        "type": "eval",
                        "step": self.global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "elapsed_time": time.time() - start_time,
                    })
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"step-{self.global_step}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average loss: {avg_epoch_loss:.4f}")
            
            # Log epoch end
            self._write_log_entry({
                "type": "epoch_end",
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "global_step": self.global_step,
                "elapsed_time": time.time() - start_time,
            })
        
        # Save final model
        self._save_checkpoint("final")
        
        total_time = time.time() - start_time
        stats = {
            "total_steps": self.global_step,
            "total_time": total_time,
            "final_loss": avg_epoch_loss,
            "best_val_loss": self.best_val_loss if self.val_data else None,
        }
        
        # Log training end
        self._write_log_entry({
            "type": "train_end",
            "total_steps": self.global_step,
            "total_time": total_time,
            "final_loss": avg_epoch_loss,
            "best_val_loss": self.best_val_loss if self.val_data else None,
        })
        
        print(f"Training complete in {total_time:.2f}s")
        return stats
    
    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        if not self.val_data:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self._batch_iterate(self.val_data):
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        log_entry = {
            "step": metrics.step,
            "loss": metrics.loss,
            "lr": metrics.learning_rate,
            "tps": metrics.tokens_per_second,
            "time": metrics.elapsed_time,
        }
        self.training_log.append(log_entry)
        
        # Write step entry to log file
        self._write_log_entry({
            "type": "step",
            "step": metrics.step,
            "epoch": self.epoch,
            "loss": metrics.loss,
            "learning_rate": metrics.learning_rate,
            "tokens_per_second": metrics.tokens_per_second,
            "elapsed_time": metrics.elapsed_time,
        })
        
        print(f"Step {metrics.step} - Loss: {metrics.loss:.4f}, LR: {metrics.learning_rate:.2e}")
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the training log file."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # trainable_parameters() returns a nested dictionary which save_safetensors cannot handle.
        # We must flatten it to a keys-dot-notation dictionary.
        def flatten_params(container, parent_key="", sep="."):
            items = []
            iterator = container.items() if isinstance(container, dict) else enumerate(container)
            
            for k, v in iterator:
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                if isinstance(v, (dict, list)):
                    items.extend(flatten_params(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        trainable_params = flatten_params(self.model.trainable_parameters())
        
        mx.save_safetensors(str(checkpoint_dir / "adapters.safetensors"), trainable_params)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load a training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load adapter weights
        adapters = mx.load(str(checkpoint_path / "adapters.safetensors"))
        self.model.load_weights(list(adapters.items()))
        
        # Load training state
        with open(checkpoint_path / "trainer_state.json", "r") as f:
            state = json.load(f)
        
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_val_loss = state["best_val_loss"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")


class KFoldTrainer:
    """
    K-Fold Cross-Validation trainer for LoRA fine-tuning.
    
    Trains multiple models on different data splits and aggregates results.
    """
    
    def __init__(
        self,
        model_loader: Callable,
        tokenizer: Any,
        full_data: list,
        k: int = 5,
        seed: int = 42,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        save_steps: int = 500,
        eval_steps: int = 100,
        logging_steps: int = 10,
        output_dir: Union[str, Path] = "outputs",
        model_name: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        fold_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize KFoldTrainer.
        
        Args:
            model_loader: Callable that returns a fresh model instance with LoRA applied.
                         This is needed because we need a fresh model for each fold.
            tokenizer: The tokenizer to use
            full_data: Complete training dataset (will be split into k folds)
            k: Number of folds
            seed: Random seed for reproducibility
            fold_callback: Optional callback(fold_idx, total_folds, fold_stats) called after each fold
        """
        self.model_loader = model_loader
        self.tokenizer = tokenizer
        self.full_data = full_data
        self.k = k
        self.seed = seed
        
        # Training params
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logs directory
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "training_log.jsonl"
        
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.fold_callback = fold_callback
        
        # Results storage
        self.fold_results = []
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a log entry to the training log file."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def _create_kfold_splits(self) -> list:
        """Create k-fold splits using indices."""
        import random
        
        n_samples = len(self.full_data)
        indices = list(range(n_samples))
        
        random.seed(self.seed)
        random.shuffle(indices)
        
        fold_size = n_samples // self.k
        remainder = n_samples % self.k
        
        folds = []
        start = 0
        for i in range(self.k):
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end = start + current_fold_size
            folds.append(indices[start:end])
            start = end
        
        splits = []
        for i in range(self.k):
            val_indices = folds[i]
            train_indices = []
            for j in range(self.k):
                if j != i:
                    train_indices.extend(folds[j])
            splits.append((train_indices, val_indices))
        
        return splits
    
    def train(self) -> Dict[str, Any]:
        """
        Run k-fold cross-validation training.
        
        Returns:
            Dictionary with aggregated results across all folds
        """
        print(f"Starting {self.k}-Fold Cross-Validation Training")
        print(f"Total samples: {len(self.full_data)}")
        
        # Log kfold training start
        self._write_log_entry({
            "type": "kfold_start",
            "k": self.k,
            "total_samples": len(self.full_data),
            "model_name": self.model_name,
            "lora_config": self.lora_config,
            "training_config": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
            },
        })
        
        # Create splits
        splits = self._create_kfold_splits()
        
        start_time = time.time()
        self.fold_results = []
        best_fold_idx = 0
        best_fold_val_loss = float("inf")
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx + 1}/{self.k}")
            print(f"{'='*50}")
            print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
            
            # Get data for this fold
            train_data = [self.full_data[i] for i in train_indices]
            val_data = [self.full_data[i] for i in val_indices]
            
            # Log fold start
            self._write_log_entry({
                "type": "fold_start",
                "fold": fold_idx,
                "total_folds": self.k,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
            })
            
            # Load fresh model for this fold
            model = self.model_loader()
            
            # Create fold-specific output directory
            fold_output_dir = self.output_dir / f"fold_{fold_idx}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trainer for this fold
            trainer = LoRATrainer(
                model=model,
                tokenizer=self.tokenizer,
                train_data=train_data,
                val_data=val_data,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                max_seq_length=self.max_seq_length,
                save_steps=self.save_steps,
                eval_steps=self.eval_steps,
                logging_steps=self.logging_steps,
                output_dir=str(fold_output_dir),
                model_name=self.model_name,
                lora_config=self.lora_config,
            )
            
            # Redirect trainer's log to main kfold log
            trainer.log_file = self.log_file
            
            # Train this fold
            fold_stats = trainer.train()
            fold_stats["fold"] = fold_idx
            fold_stats["train_samples"] = len(train_data)
            fold_stats["val_samples"] = len(val_data)
            
            self.fold_results.append(fold_stats)
            
            # Track best fold
            if fold_stats.get("best_val_loss", float("inf")) < best_fold_val_loss:
                best_fold_val_loss = fold_stats["best_val_loss"]
                best_fold_idx = fold_idx
            
            # Log fold end
            self._write_log_entry({
                "type": "fold_end",
                "fold": fold_idx,
                "total_folds": self.k,
                "fold_stats": fold_stats,
            })
            
            # Call fold callback if provided
            if self.fold_callback:
                self.fold_callback(fold_idx, self.k, fold_stats)
            
            print(f"Fold {fold_idx + 1} complete - Final loss: {fold_stats['final_loss']:.4f}, Best val loss: {fold_stats.get('best_val_loss', 'N/A')}")
        
        total_time = time.time() - start_time
        
        # Calculate aggregated metrics
        avg_final_loss = sum(r["final_loss"] for r in self.fold_results) / self.k
        avg_val_loss = sum(r.get("best_val_loss", 0) for r in self.fold_results) / self.k
        
        # Calculate standard deviation
        import math
        loss_variance = sum((r["final_loss"] - avg_final_loss) ** 2 for r in self.fold_results) / self.k
        loss_std = math.sqrt(loss_variance)
        
        val_variance = sum((r.get("best_val_loss", 0) - avg_val_loss) ** 2 for r in self.fold_results) / self.k
        val_std = math.sqrt(val_variance)
        
        summary = {
            "k": self.k,
            "total_time": total_time,
            "avg_final_loss": avg_final_loss,
            "std_final_loss": loss_std,
            "avg_val_loss": avg_val_loss,
            "std_val_loss": val_std,
            "best_fold": best_fold_idx,
            "best_val_loss": best_fold_val_loss,
            "fold_results": self.fold_results,
        }
        
        # Save summary
        summary_path = self.output_dir / "kfold_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log kfold end
        self._write_log_entry({
            "type": "kfold_end",
            "k": self.k,
            "total_time": total_time,
            "avg_final_loss": avg_final_loss,
            "std_final_loss": loss_std,
            "avg_val_loss": avg_val_loss,
            "std_val_loss": val_std,
            "best_fold": best_fold_idx,
            "best_val_loss": best_fold_val_loss,
        })
        
        print(f"\n{'='*50}")
        print(f"K-FOLD TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average final loss: {avg_final_loss:.4f} ± {loss_std:.4f}")
        print(f"Average val loss: {avg_val_loss:.4f} ± {val_std:.4f}")
        print(f"Best fold: {best_fold_idx + 1} (val loss: {best_fold_val_loss:.4f})")
        print(f"Summary saved to: {summary_path}")
        
        return summary
