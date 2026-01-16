"""
LTC Module Trainer

Responsible for training the LTC module while keeping LLM parameters frozen.
"""

import os
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .losses import LTCLoss
from .dataset import LTCTrainingDataset, collate_fn
from ..models.ltc import LatentThoughtCondenser
from ..models.model_loader import load_model, get_model_config


class LTCTrainer:
    """
    LTC module trainer.

    Training workflow:
    1. Process training samples with LLM to obtain KV Cache
    2. Compress KV Cache using LTC
    3. Compute loss and update LTC parameters
    4. Keep LLM parameters frozen
    """

    def __init__(
        self,
        model_name_or_path: str,
        ltc_config: Dict,
        training_config: Dict,
        output_dir: str
    ):
        """
        Initialize trainer.

        Args:
            model_name_or_path: HuggingFace model path
            ltc_config: LTC module configuration
            training_config: Training configuration
            output_dir: Output directory
        """
        self.model_name_or_path = model_name_or_path
        self.ltc_config = ltc_config
        self.training_config = training_config
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self._setup_model()

        # Initialize LTC
        self._setup_ltc()

        # Initialize loss function
        self._setup_loss()

        # Initialize optimizer
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')

    def _setup_model(self):
        """Setup LLM model"""
        print(f"Loading model: {self.model_name_or_path}")

        self.model, self.tokenizer = load_model(
            self.model_name_or_path,
            torch_dtype="bfloat16",
            device_map="auto"
        )

        # Freeze LLM parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_config = get_model_config(self.model_name_or_path)

    def _setup_ltc(self):
        """Setup LTC module"""
        self.ltc = LatentThoughtCondenser(
            num_layers=self.ltc_config.get("num_layers", self.model_config["num_layers"]),
            kv_dim=self.ltc_config.get("kv_dim", self.model_config["kv_dim"]),
            compression_dim=self.ltc_config.get("compression_dim", 64),
            init_std=self.ltc_config.get("init_std", 0.02)
        )

        self.ltc = self.ltc.to(self.device).to(torch.bfloat16)

    def _setup_loss(self):
        """Setup loss function"""
        self.loss_fn = LTCLoss(
            lambda_coverage=self.training_config.get("lambda_coverage", 0.1),
            lambda_orthogonality=self.training_config.get("lambda_orthogonality", 0.01),
            num_sampled_queries=self.training_config.get("num_sampled_queries", 128)
        )

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.ltc.parameters(),
            lr=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01)
        )

        total_steps = self.training_config.get("total_steps", 50000)
        warmup_steps = self.training_config.get("warmup_steps", 1000)

        # Warmup + Cosine scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    def train(
        self,
        train_dataset: LTCTrainingDataset,
        eval_dataset: Optional[LTCTrainingDataset] = None
    ):
        """
        Execute training loop.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        batch_size = self.training_config.get("batch_size", 64)
        total_steps = self.training_config.get("total_steps", 50000)
        logging_steps = self.training_config.get("logging_steps", 100)
        eval_steps = self.training_config.get("eval_steps", 1000)
        save_steps = self.training_config.get("save_steps", 5000)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        self.ltc.train()
        train_iter = iter(train_loader)

        running_loss = 0.0
        running_recon = 0.0
        running_coverage = 0.0
        running_orthogonality = 0.0

        pbar = tqdm(range(total_steps), desc="Training")

        for step in pbar:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Training step
            losses = self.training_step(batch)

            # Update statistics
            running_loss += losses["total"].item()
            running_recon += losses["recon"].item()
            running_coverage += losses["coverage"].item()
            running_orthogonality += losses["orthogonality"].item()

            self.global_step += 1

            # Logging
            if self.global_step % logging_steps == 0:
                avg_loss = running_loss / logging_steps
                avg_recon = running_recon / logging_steps
                avg_coverage = running_coverage / logging_steps
                avg_ortho = running_orthogonality / logging_steps

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "recon": f"{avg_recon:.4f}",
                    "cov": f"{avg_coverage:.4f}",
                    "ortho": f"{avg_ortho:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                self._log_metrics({
                    "train/loss": avg_loss,
                    "train/recon_loss": avg_recon,
                    "train/coverage_loss": avg_coverage,
                    "train/orthogonality_loss": avg_ortho,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                })

                running_loss = 0.0
                running_recon = 0.0
                running_coverage = 0.0
                running_orthogonality = 0.0

            # Evaluation
            if eval_dataset and self.global_step % eval_steps == 0:
                eval_results = self.evaluate(eval_dataset)
                print(f"\nEval at step {self.global_step}: {eval_results}")

                if eval_results["loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_results["loss"]
                    self.save_checkpoint(os.path.join(self.output_dir, "best_checkpoint"))

            # Save
            if self.global_step % save_steps == 0:
                self.save_checkpoint(
                    os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                )

        # Final save
        self.save_checkpoint(os.path.join(self.output_dir, "final_checkpoint"))

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            batch: Data batch

        Returns:
            Loss dictionary
        """
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Get KV Cache using LLM
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values

        # Convert KV Cache format
        original_kv = self._convert_kv_cache(past_key_values)

        # LTC compression
        compressed_kv, attention_weights = self.ltc(original_kv, attention_mask)

        # Compute loss
        losses = self.loss_fn(
            original_kv=original_kv,
            compressed_kv=compressed_kv,
            attention_weights=attention_weights,
            probe_matrix=self.ltc.get_probe_matrix()
        )

        # Backward
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ltc.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        return losses

    def _convert_kv_cache(self, past_key_values) -> Dict:
        """Convert HuggingFace format KV Cache to LTC format"""
        kv_cache = {}

        for layer_idx, (key, value) in enumerate(past_key_values):
            batch_size, num_heads, seq_len, head_dim = key.shape
            key_reshaped = key.transpose(1, 2).reshape(batch_size, seq_len, -1)
            value_reshaped = value.transpose(1, 2).reshape(batch_size, seq_len, -1)
            kv_cache[layer_idx] = (key_reshaped, value_reshaped)

        return kv_cache

    @torch.no_grad()
    def evaluate(self, eval_dataset: LTCTrainingDataset) -> Dict[str, float]:
        """
        Evaluate current model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics dictionary
        """
        self.ltc.eval()

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.training_config.get("batch_size", 64),
            shuffle=False,
            collate_fn=collate_fn
        )

        total_loss = 0.0
        total_recon = 0.0
        total_coverage = 0.0
        total_orthogonality = 0.0
        num_batches = 0

        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )

            original_kv = self._convert_kv_cache(outputs.past_key_values)
            compressed_kv, attention_weights = self.ltc(original_kv, attention_mask)

            losses = self.loss_fn(
                original_kv=original_kv,
                compressed_kv=compressed_kv,
                attention_weights=attention_weights,
                probe_matrix=self.ltc.get_probe_matrix()
            )

            total_loss += losses["total"].item()
            total_recon += losses["recon"].item()
            total_coverage += losses["coverage"].item()
            total_orthogonality += losses["orthogonality"].item()
            num_batches += 1

        self.ltc.train()

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "coverage_loss": total_coverage / num_batches,
            "orthogonality_loss": total_orthogonality / num_batches,
        }

    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        os.makedirs(path, exist_ok=True)

        # Save LTC parameters
        torch.save(self.ltc.state_dict(), os.path.join(path, "ltc_model.pt"))

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(path, "training_state.pt"))

        # Save configuration
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "ltc_config": self.ltc_config,
                "training_config": self.training_config,
                "model_name_or_path": self.model_name_or_path,
            }, f, indent=2)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        # Load LTC parameters
        self.ltc.load_state_dict(
            torch.load(os.path.join(path, "ltc_model.pt"), map_location=self.device)
        )

        # Load training state
        state = torch.load(os.path.join(path, "training_state.pt"), map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.best_eval_loss = state["best_eval_loss"]

        print(f"Checkpoint loaded from {path}, step {self.global_step}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics (extensible to wandb, etc.)"""
        pass
