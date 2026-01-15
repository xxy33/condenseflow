"""Benchmarks module for evaluation"""

from .aime import load_aime
from .gpqa import load_gpqa
from .mbpp import load_mbpp
from .medqa import load_medqa


def load_benchmark(name: str):
    """
    加载指定的benchmark数据。

    Args:
        name: benchmark名称

    Returns:
        (questions, references) 元组
    """
    name_lower = name.lower()

    if "aime2024" in name_lower:
        return load_aime(year=2024)
    elif "aime2025" in name_lower:
        return load_aime(year=2025)
    elif "aime" in name_lower:
        return load_aime()
    elif "gpqa" in name_lower:
        return load_gpqa()
    elif "mbpp" in name_lower:
        return load_mbpp()
    elif "medqa" in name_lower:
        return load_medqa()
    else:
        raise ValueError(f"Unknown benchmark: {name}")


__all__ = [
    "load_benchmark",
    "load_aime",
    "load_gpqa",
    "load_mbpp",
    "load_medqa",
]
