"""
Pipeline基类

定义协作流程的基本接口和通用功能。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from ..models.ltc_wrapper import LTCWrapper
from ..agents.base_agent import CommunicationMode


class BasePipeline(ABC):
    """
    协作Pipeline基类。

    定义了多Agent协作流程的基本接口。
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow"
    ):
        """
        初始化Pipeline。

        Args:
            model_wrapper: LTC封装的模型
            communication_mode: 通信模式 ("text", "dense", "condenseflow")
        """
        self.model_wrapper = model_wrapper
        self.communication_mode = communication_mode

        # 统计信息
        self._timing_stats = {}
        self._memory_stats = {}

    @abstractmethod
    def run(
        self,
        question: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        执行协作流程。

        Args:
            question: 输入问题
            verbose: 是否输出详细信息

        Returns:
            包含答案和统计信息的字典
        """
        pass

    def _start_timer(self, name: str):
        """开始计时"""
        self._timing_stats[f"{name}_start"] = time.time()

    def _end_timer(self, name: str):
        """结束计时"""
        start_time = self._timing_stats.get(f"{name}_start", time.time())
        self._timing_stats[name] = time.time() - start_time

    def _record_memory(self, name: str):
        """记录当前GPU内存使用"""
        if torch.cuda.is_available():
            self._memory_stats[name] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

    def _clear_memory_stats(self):
        """清除内存统计"""
        self._memory_stats = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_timing_stats(self) -> Dict[str, float]:
        """获取时间统计"""
        return {k: v for k, v in self._timing_stats.items() if not k.endswith("_start")}

    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """获取内存统计"""
        return self._memory_stats

    def reset_stats(self):
        """重置所有统计信息"""
        self._timing_stats = {}
        self._clear_memory_stats()
