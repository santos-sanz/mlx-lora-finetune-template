"""
Training engine for LoRA fine-tuning with MLX.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
import json

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
        
        self.callbacks = callbacks or {}
        
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
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"step-{self.global_step}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self._save_checkpoint("final")
        
        total_time = time.time() - start_time
        stats = {
            "total_steps": self.global_step,
            "total_time": total_time,
            "final_loss": avg_epoch_loss,
            "best_val_loss": self.best_val_loss if self.val_data else None,
        }
        
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
        self.training_log.append({
            "step": metrics.step,
            "loss": metrics.loss,
            "lr": metrics.learning_rate,
            "tps": metrics.tokens_per_second,
            "time": metrics.elapsed_time,
        })
        print(f"Step {metrics.step} - Loss: {metrics.loss:.4f}, LR: {metrics.learning_rate:.2e}")
    
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
