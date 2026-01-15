"""
Latent Thought Condenser (LTC) Core Module

Compresses variable-length KV Cache into fixed-size semantic representations,
achieving O(1) communication complexity.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentThoughtCondenser(nn.Module):
    """
    Core compression module that compresses variable-length KV Cache into fixed-size representations.

    Architecture:
    - Learnable semantic probes: Q_c ∈ R^{K × d_k}, where K is the compression dimension
    - Cross-layer parameter sharing: All Transformer layers share the same probes
    - Layer normalization: Independent LayerNorm parameters for each layer

    Input:
    - kv_cache: Dict[int, Tuple[Tensor, Tensor]]
      layer_idx -> (K, V), where K, V ∈ R^{batch, T, d_k}

    Output:
    - compressed_cache: Dict[int, Tuple[Tensor, Tensor]]
      layer_idx -> (K_compressed, V_compressed), where K_compressed, V_compressed ∈ R^{batch, K, d_k}
    """

    def __init__(
        self,
        num_layers: int,
        kv_dim: int,
        compression_dim: int = 64,
        init_std: float = 0.02
    ):
        """
        Initialize LTC module.

        Args:
            num_layers: Number of Transformer layers
            kv_dim: KV Cache dimension (n_kv_heads * head_dim)
            compression_dim: Compressed sequence length K
            init_std: Probe initialization standard deviation
        """
        super().__init__()

        self.num_layers = num_layers
        self.kv_dim = kv_dim
        self.compression_dim = compression_dim
        self.init_std = init_std

        # Learnable semantic probes Q_c ∈ R^{K × d_k}
        # All layers share the same set of probes
        self.semantic_probes = nn.Parameter(
            torch.randn(compression_dim, kv_dim) * init_std
        )

        # Independent LayerNorm parameters for each layer
        self.key_layer_norms = nn.ModuleList([
            nn.LayerNorm(kv_dim) for _ in range(num_layers)
        ])
        self.value_layer_norms = nn.ModuleList([
            nn.LayerNorm(kv_dim) for _ in range(num_layers)
        ])

        # Scale factor
        self.scale = math.sqrt(kv_dim)

    def forward(
        self,
        kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, torch.Tensor]]:
        """
        Perform KV Cache compression.

        Args:
            kv_cache: layer_idx -> (K, V), where K, V ∈ R^{batch, seq_len, kv_dim}
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            compressed_cache: Compressed KV Cache
            attention_weights: Attention weights for each layer, used for loss computation
        """
        compressed_cache = {}
        attention_weights = {}

        for layer_idx, (keys, values) in kv_cache.items():
            # keys, values: [batch, seq_len, kv_dim]

            # Compute attention weights
            # semantic_probes: [compression_dim, kv_dim]
            # keys: [batch, seq_len, kv_dim]
            # attn_weights: [batch, compression_dim, seq_len]
            attn_weights = self.compute_attention_weights(keys, attention_mask)
            attention_weights[layer_idx] = attn_weights

            # Aggregate Key and Value using attention weights
            # compressed_keys: [batch, compression_dim, kv_dim]
            compressed_keys = torch.bmm(attn_weights, keys)
            compressed_values = torch.bmm(attn_weights, values)

            # Apply layer normalization
            compressed_keys = self.key_layer_norms[layer_idx](compressed_keys)
            compressed_values = self.value_layer_norms[layer_idx](compressed_values)

            compressed_cache[layer_idx] = (compressed_keys, compressed_values)

        return compressed_cache, attention_weights

    def compute_attention_weights(
        self,
        keys: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights between semantic probes and Keys.

        Args:
            keys: [batch, seq_len, kv_dim]
            attention_mask: [batch, seq_len], 1 for valid positions, 0 for padding

        Returns:
            attention_weights: [batch, compression_dim, seq_len]
        """
        batch_size = keys.size(0)

        # Expand probes to batch dimension
        # probes: [batch, compression_dim, kv_dim]
        probes = self.semantic_probes.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention scores
        # scores: [batch, compression_dim, seq_len]
        scores = torch.bmm(probes, keys.transpose(1, 2)) / self.scale

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len]
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights

    def get_compression_ratio(self, original_seq_len: int) -> float:
        """
        Calculate compression ratio.

        Args:
            original_seq_len: Original sequence length

        Returns:
            Compression ratio (compression_dim / original_seq_len)
        """
        return self.compression_dim / original_seq_len

    def get_memory_reduction(self, original_seq_len: int) -> float:
        """
        Calculate memory reduction ratio.

        Args:
            original_seq_len: Original sequence length

        Returns:
            Memory reduction ratio (1 - compression_ratio)
        """
        return 1.0 - self.get_compression_ratio(original_seq_len)

    def compress_single_layer(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV Cache for a single layer.

        Args:
            keys: [batch, seq_len, kv_dim]
            values: [batch, seq_len, kv_dim]
            layer_idx: Layer index
            attention_mask: [batch, seq_len]

        Returns:
            compressed_keys: [batch, compression_dim, kv_dim]
            compressed_values: [batch, compression_dim, kv_dim]
            attention_weights: [batch, compression_dim, seq_len]
        """
        attn_weights = self.compute_attention_weights(keys, attention_mask)

        compressed_keys = torch.bmm(attn_weights, keys)
        compressed_values = torch.bmm(attn_weights, values)

        compressed_keys = self.key_layer_norms[layer_idx](compressed_keys)
        compressed_values = self.value_layer_norms[layer_idx](compressed_values)

        return compressed_keys, compressed_values, attn_weights

    def get_probe_matrix(self) -> torch.Tensor:
        """Get semantic probe matrix for computing orthogonality loss"""
        return self.semantic_probes

    @torch.no_grad()
    def analyze_attention_distribution(
        self,
        kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze attention distribution for visualization and debugging.

        Returns:
            Dictionary containing various statistics
        """
        _, attention_weights = self.forward(kv_cache, attention_mask)

        # Compute mean attention distribution
        all_weights = torch.stack([w for w in attention_weights.values()])
        mean_weights = all_weights.mean(dim=0)  # [batch, compression_dim, seq_len]

        # Compute entropy
        entropy = -torch.sum(mean_weights * torch.log(mean_weights + 1e-10), dim=-1)

        # Compute effective rank
        # Using singular value decomposition
        U, S, V = torch.svd(mean_weights.mean(dim=0))  # [compression_dim, seq_len]
        normalized_S = S / S.sum()
        effective_rank = torch.exp(-torch.sum(normalized_S * torch.log(normalized_S + 1e-10)))

        return {
            "mean_attention": mean_weights,
            "entropy": entropy,
            "effective_rank": effective_rank,
            "singular_values": S,
        }
