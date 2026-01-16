"""
KV Cache Operation Utilities

Provides utility functions for KV Cache format conversion and operations.
"""

from typing import Dict, Tuple

import torch


def reshape_kv_for_ltc(
    past_key_values: Tuple,
    num_kv_heads: int,
    head_dim: int
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert HuggingFace format KV Cache to LTC input format.

    HuggingFace format: tuple of (key, value) per layer
    key/value shape: [batch, num_kv_heads, seq_len, head_dim]

    LTC format: Dict[layer_idx, (key, value)]
    key/value shape: [batch, seq_len, kv_dim]

    Args:
        past_key_values: HuggingFace format past_key_values
        num_kv_heads: number of KV heads
        head_dim: dimension per head

    Returns:
        LTC format KV Cache dictionary
    """
    kv_cache = {}

    for layer_idx, (key, value) in enumerate(past_key_values):
        batch_size, n_heads, seq_len, h_dim = key.shape

        # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads * head_dim]
        key_reshaped = key.transpose(1, 2).reshape(batch_size, seq_len, -1)
        value_reshaped = value.transpose(1, 2).reshape(batch_size, seq_len, -1)

        kv_cache[layer_idx] = (key_reshaped, value_reshaped)

    return kv_cache


def reshape_kv_from_ltc(
    compressed_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    num_kv_heads: int,
    head_dim: int
) -> Tuple:
    """
    Convert LTC format compressed cache back to HuggingFace format.

    LTC format: Dict[layer_idx, (key, value)]
    key/value shape: [batch, compression_dim, kv_dim]

    HuggingFace format: tuple of (key, value) per layer
    key/value shape: [batch, num_kv_heads, compression_dim, head_dim]

    Args:
        compressed_cache: LTC format compressed KV Cache
        num_kv_heads: number of KV heads
        head_dim: dimension per head

    Returns:
        HuggingFace format past_key_values
    """
    num_layers = len(compressed_cache)
    past_key_values = []

    for layer_idx in range(num_layers):
        key, value = compressed_cache[layer_idx]
        batch_size, comp_dim, kv_dim = key.shape

        # Reshape: [batch, comp_dim, kv_dim] -> [batch, num_kv_heads, comp_dim, head_dim]
        key_reshaped = key.view(batch_size, comp_dim, num_kv_heads, head_dim)
        key_reshaped = key_reshaped.transpose(1, 2)

        value_reshaped = value.view(batch_size, comp_dim, num_kv_heads, head_dim)
        value_reshaped = value_reshaped.transpose(1, 2)

        past_key_values.append((key_reshaped, value_reshaped))

    return tuple(past_key_values)


def concatenate_kv_caches(
    cache1: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    cache2: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Concatenate two KV Caches along the sequence dimension.

    Args:
        cache1: first KV Cache
        cache2: second KV Cache

    Returns:
        concatenated KV Cache
    """
    result = {}

    for layer_idx in cache1.keys():
        k1, v1 = cache1[layer_idx]
        k2, v2 = cache2[layer_idx]

        # Concatenate along seq_len dimension
        k_concat = torch.cat([k1, k2], dim=1)
        v_concat = torch.cat([v1, v2], dim=1)

        result[layer_idx] = (k_concat, v_concat)

    return result


def slice_kv_cache(
    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    start: int,
    end: int
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Slice KV Cache.

    Args:
        kv_cache: KV Cache
        start: start position
        end: end position

    Returns:
        sliced KV Cache
    """
    result = {}

    for layer_idx, (key, value) in kv_cache.items():
        result[layer_idx] = (key[:, start:end, :], value[:, start:end, :])

    return result


def get_kv_cache_size(
    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, float]:
    """
    Calculate KV Cache size.

    Args:
        kv_cache: KV Cache

    Returns:
        size information dictionary
    """
    total_bytes = 0
    total_elements = 0

    for layer_idx, (key, value) in kv_cache.items():
        total_bytes += key.numel() * key.element_size()
        total_bytes += value.numel() * value.element_size()
        total_elements += key.numel() + value.numel()

    return {
        "bytes": total_bytes,
        "mb": total_bytes / 1024 / 1024,
        "gb": total_bytes / 1024 / 1024 / 1024,
        "elements": total_elements,
        "num_layers": len(kv_cache),
    }


def move_kv_cache_to_device(
    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Move KV Cache to specified device.

    Args:
        kv_cache: KV Cache
        device: target device

    Returns:
        moved KV Cache
    """
    result = {}

    for layer_idx, (key, value) in kv_cache.items():
        result[layer_idx] = (key.to(device), value.to(device))

    return result
