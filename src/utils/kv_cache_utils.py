"""
KV Cache操作工具

提供KV Cache格式转换和操作的工具函数。
"""

from typing import Dict, Tuple

import torch


def reshape_kv_for_ltc(
    past_key_values: Tuple,
    num_kv_heads: int,
    head_dim: int
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    将HuggingFace格式的KV Cache转换为LTC输入格式。

    HuggingFace格式: tuple of (key, value) per layer
    key/value shape: [batch, num_kv_heads, seq_len, head_dim]

    LTC格式: Dict[layer_idx, (key, value)]
    key/value shape: [batch, seq_len, kv_dim]

    Args:
        past_key_values: HuggingFace格式的past_key_values
        num_kv_heads: KV头数量
        head_dim: 每个头的维度

    Returns:
        LTC格式的KV Cache字典
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
    将LTC格式的压缩Cache转换回HuggingFace格式。

    LTC格式: Dict[layer_idx, (key, value)]
    key/value shape: [batch, compression_dim, kv_dim]

    HuggingFace格式: tuple of (key, value) per layer
    key/value shape: [batch, num_kv_heads, compression_dim, head_dim]

    Args:
        compressed_cache: LTC格式的压缩KV Cache
        num_kv_heads: KV头数量
        head_dim: 每个头的维度

    Returns:
        HuggingFace格式的past_key_values
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
    沿序列维度拼接两个KV Cache。

    Args:
        cache1: 第一个KV Cache
        cache2: 第二个KV Cache

    Returns:
        拼接后的KV Cache
    """
    result = {}

    for layer_idx in cache1.keys():
        k1, v1 = cache1[layer_idx]
        k2, v2 = cache2[layer_idx]

        # 沿seq_len维度拼接
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
    切片KV Cache。

    Args:
        kv_cache: KV Cache
        start: 起始位置
        end: 结束位置

    Returns:
        切片后的KV Cache
    """
    result = {}

    for layer_idx, (key, value) in kv_cache.items():
        result[layer_idx] = (key[:, start:end, :], value[:, start:end, :])

    return result


def get_kv_cache_size(
    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, float]:
    """
    计算KV Cache的大小。

    Args:
        kv_cache: KV Cache

    Returns:
        大小信息字典
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
    将KV Cache移动到指定设备。

    Args:
        kv_cache: KV Cache
        device: 目标设备

    Returns:
        移动后的KV Cache
    """
    result = {}

    for layer_idx, (key, value) in kv_cache.items():
        result[layer_idx] = (key.to(device), value.to(device))

    return result
