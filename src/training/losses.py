"""
LTC训练的损失函数

包含重建损失、覆盖正则化和正交正则化。
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCLoss(nn.Module):
    """
    LTC训练的复合损失函数。

    组成:
    - L_recon: 重建损失，确保压缩Cache产生相近的注意力输出
    - L_coverage: 覆盖正则化，防止探针坍缩
    - L_orthogonality: 正交正则化，鼓励探针捕获互补特征
    """

    def __init__(
        self,
        lambda_coverage: float = 0.1,
        lambda_orthogonality: float = 0.01,
        num_sampled_queries: int = 128
    ):
        """
        初始化损失函数。

        Args:
            lambda_coverage: 覆盖损失权重
            lambda_orthogonality: 正交损失权重
            num_sampled_queries: 用于计算重建损失的采样query数量
        """
        super().__init__()

        self.lambda_coverage = lambda_coverage
        self.lambda_orthogonality = lambda_orthogonality
        self.num_sampled_queries = num_sampled_queries

    def forward(
        self,
        original_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        compressed_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        attention_weights: Dict[int, torch.Tensor],
        probe_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失及各分量。

        Args:
            original_kv: 原始KV Cache，层索引 -> (K, V)
            compressed_kv: 压缩后的KV Cache
            attention_weights: LTC计算的注意力权重
            probe_matrix: 语义探针矩阵 [compression_dim, kv_dim]

        Returns:
            包含各损失分量的字典:
            - total: 总损失
            - recon: 重建损失
            - coverage: 覆盖损失
            - orthogonality: 正交损失
        """
        # 获取设备和数据类型
        device = probe_matrix.device
        dtype = probe_matrix.dtype

        # 采样query向量用于重建损失
        kv_dim = probe_matrix.shape[1]
        sampled_queries = torch.randn(
            self.num_sampled_queries, kv_dim,
            device=device, dtype=dtype
        )
        sampled_queries = F.normalize(sampled_queries, dim=-1)

        # 计算各损失分量
        recon_loss = self.reconstruction_loss(original_kv, compressed_kv, sampled_queries)
        coverage_loss = self.coverage_loss(attention_weights)
        orthogonality_loss = self.orthogonality_loss(probe_matrix)

        # 总损失
        total_loss = (
            recon_loss +
            self.lambda_coverage * coverage_loss +
            self.lambda_orthogonality * orthogonality_loss
        )

        return {
            "total": total_loss,
            "recon": recon_loss,
            "coverage": coverage_loss,
            "orthogonality": orthogonality_loss,
        }

    def reconstruction_loss(
        self,
        original_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        compressed_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        sampled_queries: torch.Tensor
    ) -> torch.Tensor:
        """
        计算重建损失。

        对于采样的query向量，比较:
        - Attn(Q_s, K_orig, V_orig)
        - Attn(Q_s, K_compressed, V_compressed)

        Args:
            original_kv: 原始KV Cache
            compressed_kv: 压缩后的KV Cache
            sampled_queries: 采样的query向量 [num_queries, kv_dim]

        Returns:
            重建损失
        """
        total_loss = 0.0
        num_layers = len(original_kv)

        for layer_idx in original_kv.keys():
            orig_k, orig_v = original_kv[layer_idx]
            comp_k, comp_v = compressed_kv[layer_idx]

            # orig_k, orig_v: [batch, seq_len, kv_dim]
            # comp_k, comp_v: [batch, compression_dim, kv_dim]
            batch_size = orig_k.shape[0]

            # 扩展sampled_queries到batch维度
            # queries: [batch, num_queries, kv_dim]
            queries = sampled_queries.unsqueeze(0).expand(batch_size, -1, -1)

            # 计算原始注意力输出
            orig_output = self._compute_attention_output(queries, orig_k, orig_v)

            # 计算压缩后的注意力输出
            comp_output = self._compute_attention_output(queries, comp_k, comp_v)

            # MSE损失
            layer_loss = F.mse_loss(comp_output, orig_output)
            total_loss = total_loss + layer_loss

        return total_loss / num_layers

    def _compute_attention_output(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力输出。

        Args:
            queries: [batch, num_queries, dim]
            keys: [batch, seq_len, dim]
            values: [batch, seq_len, dim]

        Returns:
            attention_output: [batch, num_queries, dim]
        """
        dim = queries.shape[-1]
        scale = math.sqrt(dim)

        # 计算注意力分数
        # scores: [batch, num_queries, seq_len]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / scale

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        # output: [batch, num_queries, dim]
        output = torch.bmm(attn_weights, values)

        return output

    def coverage_loss(
        self,
        attention_weights: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算覆盖损失: -H(mean_attention)

        最大化平均注意力分布的熵，防止探针只关注少数位置。

        Args:
            attention_weights: 每层的注意力权重
                层索引 -> [batch, compression_dim, seq_len]

        Returns:
            覆盖损失（负熵）
        """
        total_entropy = 0.0
        num_layers = len(attention_weights)

        for layer_idx, weights in attention_weights.items():
            # weights: [batch, compression_dim, seq_len]

            # 计算每个位置被关注的平均程度
            # mean_attention: [batch, seq_len]
            mean_attention = weights.mean(dim=1)

            # 归一化确保是有效的概率分布
            mean_attention = mean_attention / (mean_attention.sum(dim=-1, keepdim=True) + 1e-10)

            # 计算熵
            entropy = -torch.sum(mean_attention * torch.log(mean_attention + 1e-10), dim=-1)

            # 取batch平均
            total_entropy = total_entropy + entropy.mean()

        # 返回负熵（因为我们要最大化熵，即最小化负熵）
        return -total_entropy / num_layers

    def orthogonality_loss(
        self,
        probe_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        计算正交损失: ||Q_c @ Q_c^T - I||_F^2

        鼓励探针之间正交，捕获互补特征。

        Args:
            probe_matrix: 语义探针矩阵 [compression_dim, kv_dim]

        Returns:
            正交损失
        """
        compression_dim = probe_matrix.shape[0]

        # 归一化探针
        normalized_probes = F.normalize(probe_matrix, dim=-1)

        # 计算探针之间的相似度矩阵
        # similarity: [compression_dim, compression_dim]
        similarity = torch.mm(normalized_probes, normalized_probes.t())

        # 目标是单位矩阵
        identity = torch.eye(compression_dim, device=probe_matrix.device, dtype=probe_matrix.dtype)

        # Frobenius范数
        loss = torch.norm(similarity - identity, p='fro') ** 2

        return loss / (compression_dim ** 2)


class ReconstructionLoss(nn.Module):
    """单独的重建损失模块"""

    def __init__(self, num_sampled_queries: int = 128):
        super().__init__()
        self.num_sampled_queries = num_sampled_queries

    def forward(
        self,
        original_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        compressed_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        # 获取维度信息
        first_layer = list(original_kv.keys())[0]
        kv_dim = original_kv[first_layer][0].shape[-1]
        device = original_kv[first_layer][0].device
        dtype = original_kv[first_layer][0].dtype

        # 采样query
        sampled_queries = torch.randn(
            self.num_sampled_queries, kv_dim,
            device=device, dtype=dtype
        )
        sampled_queries = F.normalize(sampled_queries, dim=-1)

        total_loss = 0.0
        num_layers = len(original_kv)

        for layer_idx in original_kv.keys():
            orig_k, orig_v = original_kv[layer_idx]
            comp_k, comp_v = compressed_kv[layer_idx]

            batch_size = orig_k.shape[0]
            queries = sampled_queries.unsqueeze(0).expand(batch_size, -1, -1)

            orig_output = self._attention(queries, orig_k, orig_v)
            comp_output = self._attention(queries, comp_k, comp_v)

            total_loss = total_loss + F.mse_loss(comp_output, orig_output)

        return total_loss / num_layers

    def _attention(self, q, k, v):
        scale = math.sqrt(q.shape[-1])
        scores = torch.bmm(q, k.transpose(1, 2)) / scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v)
