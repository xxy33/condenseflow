"""
评估指标

包含准确率计算和效率指标计算。
"""

import re
import time
from typing import Any, Dict, List, Optional

import torch


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    task_type: str = "math"
) -> Dict[str, float]:
    """
    计算准确率。

    Args:
        predictions: 预测答案列表
        references: 参考答案列表
        task_type: 任务类型 ("math", "code", "mcq")

    Returns:
        准确率指标字典
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        if task_type == "math":
            if _compare_math_answers(pred, ref):
                correct += 1
        elif task_type == "mcq":
            if _compare_mcq_answers(pred, ref):
                correct += 1
        else:
            if _compare_text_answers(pred, ref):
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def _compare_math_answers(pred: str, ref: str) -> bool:
    """比较数学答案"""
    pred_num = _extract_number(pred)
    ref_num = _extract_number(ref)

    if pred_num is None or ref_num is None:
        return pred.strip().lower() == ref.strip().lower()

    return abs(pred_num - ref_num) < 1e-6


def _extract_number(text: str) -> Optional[float]:
    """从文本中提取数字"""
    # 移除逗号
    text = text.replace(",", "")

    # 尝试匹配数字
    patterns = [
        r"[-+]?\d*\.?\d+",
        r"\\boxed\{([-+]?\d*\.?\d+)\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    return None


def _compare_mcq_answers(pred: str, ref: str) -> bool:
    """比较选择题答案"""
    pred_choice = _extract_choice(pred)
    ref_choice = _extract_choice(ref)
    return pred_choice == ref_choice


def _extract_choice(text: str) -> str:
    """提取选择题选项"""
    text = text.strip().upper()
    for char in text:
        if char in "ABCDE":
            return char
    return text[0] if text else ""


def _compare_text_answers(pred: str, ref: str) -> bool:
    """比较文本答案"""
    return pred.strip().lower() == ref.strip().lower()


def compute_efficiency_metrics(
    memory_stats: Dict[str, Dict],
    timing_stats: Dict[str, float]
) -> Dict[str, float]:
    """
    计算效率指标。

    Args:
        memory_stats: 内存统计
        timing_stats: 时间统计

    Returns:
        效率指标字典
    """
    metrics = {}

    # 时间指标
    if "total" in timing_stats:
        metrics["total_time_seconds"] = timing_stats["total"]

    # 内存指标
    if memory_stats:
        max_memory = 0
        for stage, stats in memory_stats.items():
            if "max_allocated_mb" in stats:
                max_memory = max(max_memory, stats["max_allocated_mb"])
        metrics["peak_memory_mb"] = max_memory

    return metrics
