"""
LTC模块训练器

负责训练LTC模块，LLM参数保持冻结。
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
    LTC模块训练器。

    训练流程:
    1. 使用LLM处理训练样本，获取KV Cache
    2. 使用LTC压缩KV Cache
    3. 计算损失并更新LTC参数
    4. LLM参数保持冻结
    """

    def __init__(
        self,
        model_name_or_path: str,
        ltc_config: Dict,
        training_config: Dict,
        output_dir: str
    ):
        """
        初始化训练器。

        Args:
            model_name_or_path: HuggingFace模型路径
            ltc_config: LTC模块配置
            training_config: 训练配置
            output_dir: 输出目录
        """
        self.model_name_or_path = model_name_or_path
        self.ltc_config = ltc_config
        self.training_config = training_config
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self._setup_model()

        # 初始化LTC
        self._setup_ltc()

        # 初始化损失函数
        self._setup_loss()

        # 初始化优化器
        self._setup_optimizer()

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')

    def _setup_model(self):
        """设置LLM模型"""
        print(f"Loading model: {self.model_name_or_path}")

        self.model, self.tokenizer = load_model(
            self.model_name_or_path,
            torch_dtype="bfloat16",
            device_map="auto"
        )

        # 冻结LLM参数
        for param in self.model.parameters():
            param.requires_grad = False

        self.model_config = get_model_config(self.model_name_or_path)

    def _setup_ltc(self):
        """设置LTC模块"""
        self.ltc = LatentThoughtCondenser(
            num_layers=self.ltc_config.get("num_layers", self.model_config["num_layers"]),
            kv_dim=self.ltc_config.get("kv_dim", self.model_config["kv_dim"]),
            compression_dim=self.ltc_config.get("compression_dim", 64),
            init_std=self.ltc_config.get("init_std", 0.02)
        )

        self.ltc = self.ltc.to(self.device).to(torch.bfloat16)

    def _setup_loss(self):
        """设置损失函数"""
        self.loss_fn = LTCLoss(
            lambda_coverage=self.training_config.get("lambda_coverage", 0.1),
            lambda_orthogonality=self.training_config.get("lambda_orthogonality", 0.01),
            num_sampled_queries=self.training_config.get("num_sampled_queries", 128)
        )

    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = AdamW(
            self.ltc.parameters(),
            lr=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01)
        )

        total_steps = self.training_config.get("total_steps", 50000)
        warmup_steps = self.training_config.get("warmup_steps", 1000)

        # Warmup + Cosine调度
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
        执行训练循环。

        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集（可选）
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
            # 获取batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # 训练步骤
            losses = self.training_step(batch)

            # 更新统计
            running_loss += losses["total"].item()
            running_recon += losses["recon"].item()
            running_coverage += losses["coverage"].item()
            running_orthogonality += losses["orthogonality"].item()

            self.global_step += 1

            # 日志
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

            # 评估
            if eval_dataset and self.global_step % eval_steps == 0:
                eval_results = self.evaluate(eval_dataset)
                print(f"\nEval at step {self.global_step}: {eval_results}")

                if eval_results["loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_results["loss"]
                    self.save_checkpoint(os.path.join(self.output_dir, "best_checkpoint"))

            # 保存
            if self.global_step % save_steps == 0:
                self.save_checkpoint(
                    os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                )

        # 最终保存
        self.save_checkpoint(os.path.join(self.output_dir, "final_checkpoint"))

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        单步训练。

        Args:
            batch: 数据batch

        Returns:
            损失字典
        """
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # 使用LLM获取KV Cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values

        # 转换KV Cache格式
        original_kv = self._convert_kv_cache(past_key_values)

        # LTC压缩
        compressed_kv, attention_weights = self.ltc(original_kv, attention_mask)

        # 计算损失
        losses = self.loss_fn(
            original_kv=original_kv,
            compressed_kv=compressed_kv,
            attention_weights=attention_weights,
            probe_matrix=self.ltc.get_probe_matrix()
        )

        # 反向传播
        losses["total"].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.ltc.parameters(), max_norm=1.0)

        # 更新参数
        self.optimizer.step()
        self.scheduler.step()

        return losses

    def _convert_kv_cache(self, past_key_values) -> Dict:
        """将HuggingFace格式的KV Cache转换为LTC格式"""
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
        评估当前模型。

        Args:
            eval_dataset: 评估数据集

        Returns:
            评估指标字典
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
        """保存checkpoint"""
        os.makedirs(path, exist_ok=True)

        # 保存LTC参数
        torch.save(self.ltc.state_dict(), os.path.join(path, "ltc_model.pt"))

        # 保存优化器状态
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(path, "training_state.pt"))

        # 保存配置
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "ltc_config": self.ltc_config,
                "training_config": self.training_config,
                "model_name_or_path": self.model_name_or_path,
            }, f, indent=2)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载checkpoint"""
        # 加载LTC参数
        self.ltc.load_state_dict(
            torch.load(os.path.join(path, "ltc_model.pt"), map_location=self.device)
        )

        # 加载训练状态
        state = torch.load(os.path.join(path, "training_state.pt"), map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.best_eval_loss = state["best_eval_loss"]

        print(f"Checkpoint loaded from {path}, step {self.global_step}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """记录指标（可扩展为wandb等）"""
        pass
