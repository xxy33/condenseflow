"""
Loss Functions for LTC Training

Contains reconstruction loss, coverage regularization, and orthogonality regularization.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCLoss(nn.Module):
    """
    Composite loss function for LTC training.

    Components:
    - L_recon: Reconstruction loss, ensures compressed Cache produces similar attention outputs
    - L_coverage: Coverage regularization, prevents probe collapse
    - L_orthogonality: Orthogonality regularization, encourages probes to capture complementary features
    """

    def __init__(
        self,
        lambda_coverage: float = 0.1,
        lambda_orthogonality: float = 0.01,
        num_sampled_queries: int = 128
    ):
        """
        Initialize loss function.

        Args:
            lambda_coverage: Coverage loss weight
            lambda_orthogonality: Orthogonality loss weight
            num_sampled_queries: Number of sampled queries for reconstruction loss
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
        Compute total loss and components.

        Args:
            original_kv: Original KV Cache, layer_idx -> (K, V)
            compressed_kv: Compressed KV Cache
            attention_weights: Attention weights computed by LTC
            probe_matrix: Semantic probe matrix [compression_dim, kv_dim]

        Returns:
            Dictionary containing loss components:
            - total: Total loss
            - recon: Reconstruction loss
            - coverage: Coverage loss
            - orthogonality: Orthogonality loss
        """
        # Get device and dtype
        device = probe_matrix.device
        dtype = probe_matrix.dtype

        # Sample query vectors for reconstruction loss
        kv_dim = probe_matrix.shape[1]
        sampled_queries = torch.randn(
            self.num_sampled_queries, kv_dim,
            device=device, dtype=dtype
        )
        sampled_queries = F.normalize(sampled_queries, dim=-1)

        # Compute loss components
        recon_loss = self.reconstruction_loss(original_kv, compressed_kv, sampled_queries)
        coverage_loss = self.coverage_loss(attention_weights)
        orthogonality_loss = self.orthogonality_loss(probe_matrix)

        # Total loss
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
        Compute reconstruction loss.

        For sampled query vectors, compare:
        - Attn(Q_s, K_orig, V_orig)
        - Attn(Q_s, K_compressed, V_compressed)

        Args:
            original_kv: Original KV Cache
            compressed_kv: Compressed KV Cache
            sampled_queries: Sampled query vectors [num_queries, kv_dim]

        Returns:
            Reconstruction loss
        """
        total_loss = 0.0
        num_layers = len(original_kv)

        for layer_idx in original_kv.keys():
            orig_k, orig_v = original_kv[layer_idx]
            comp_k, comp_v = compressed_kv[layer_idx]

            # orig_k, orig_v: [batch, seq_len, kv_dim]
            # comp_k, comp_v: [batch, compression_dim, kv_dim]
            batch_size = orig_k.shape[0]

            # Expand sampled_queries to batch dimension
            # queries: [batch, num_queries, kv_dim]
            queries = sampled_queries.unsqueeze(0).expand(batch_size, -1, -1)

            # Compute original attention output
            orig_output = self._compute_attention_output(queries, orig_k, orig_v)

            # Compute compressed attention output
            comp_output = self._compute_attention_output(queries, comp_k, comp_v)

            # MSE loss
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
        Compute attention output.

        Args:
            queries: [batch, num_queries, dim]
            keys: [batch, seq_len, dim]
            values: [batch, seq_len, dim]

        Returns:
            attention_output: [batch, num_queries, dim]
        """
        dim = queries.shape[-1]
        scale = math.sqrt(dim)

        # Compute attention scores
        # scores: [batch, num_queries, seq_len]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / scale

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        # output: [batch, num_queries, dim]
        output = torch.bmm(attn_weights, values)

        return output

    def coverage_loss(
        self,
        attention_weights: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute coverage loss: -H(mean_attention)

        Maximize entropy of mean attention distribution, preventing probes from focusing on only a few positions.

        Args:
            attention_weights: Attention weights for each layer
                layer_idx -> [batch, compression_dim, seq_len]

        Returns:
            Coverage loss (negative entropy)
        """
        total_entropy = 0.0
        num_layers = len(attention_weights)

        for layer_idx, weights in attention_weights.items():
            # weights: [batch, compression_dim, seq_len]

            # Compute average attention for each position
            # mean_attention: [batch, seq_len]
            mean_attention = weights.mean(dim=1)

            # Normalize to ensure valid probability distribution
            mean_attention = mean_attention / (mean_attention.sum(dim=-1, keepdim=True) + 1e-10)

            # Compute entropy
            entropy = -torch.sum(mean_attention * torch.log(mean_attention + 1e-10), dim=-1)

            # Take batch average
            total_entropy = total_entropy + entropy.mean()

        # Return negative entropy (we want to maximize entropy, i.e., minimize negative entropy)
        return -total_entropy / num_layers

    def orthogonality_loss(
        self,
        probe_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute orthogonality loss: ||Q_c @ Q_c^T - I||_F^2

        Encourage probes to be orthogonal, capturing complementary features.

        Args:
            probe_matrix: Semantic probe matrix [compression_dim, kv_dim]

        Returns:
            Orthogonality loss
        """
        compression_dim = probe_matrix.shape[0]

        # Normalize probes
        normalized_probes = F.normalize(probe_matrix, dim=-1)

        # Compute similarity matrix between probes
        # similarity: [compression_dim, compression_dim]
        similarity = torch.mm(normalized_probes, normalized_probes.t())

        # Target is identity matrix
        identity = torch.eye(compression_dim, device=probe_matrix.device, dtype=probe_matrix.dtype)

        # Frobenius norm
        loss = torch.norm(similarity - identity, p='fro') ** 2

        return loss / (compression_dim ** 2)


class ReconstructionLoss(nn.Module):
    """Standalone reconstruction loss module"""

    def __init__(self, num_sampled_queries: int = 128):
        super().__init__()
        self.num_sampled_queries = num_sampled_queries

    def forward(
        self,
        original_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        compressed_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        # Get dimension info
        first_layer = list(original_kv.keys())[0]
        kv_dim = original_kv[first_layer][0].shape[-1]
        device = original_kv[first_layer][0].device
        dtype = original_kv[first_layer][0].dtype

        # Sample queries
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
