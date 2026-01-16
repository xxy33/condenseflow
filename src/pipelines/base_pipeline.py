"""
Pipeline Base Class

Defines basic interfaces and common functionality for collaboration workflows.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from ..models.ltc_wrapper import LTCWrapper
from ..agents.base_agent import CommunicationMode


class BasePipeline(ABC):
    """
    Collaboration Pipeline base class.

    Defines basic interfaces for multi-Agent collaboration workflows.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow"
    ):
        """
        Initialize Pipeline.

        Args:
            model_wrapper: LTC wrapped model
            communication_mode: Communication mode ("text", "dense", "condenseflow")
        """
        self.model_wrapper = model_wrapper
        self.communication_mode = communication_mode

        # Statistics
        self._timing_stats = {}
        self._memory_stats = {}

    @abstractmethod
    def run(
        self,
        question: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute collaboration workflow.

        Args:
            question: Input question
            verbose: Whether to output detailed information

        Returns:
            Dictionary containing answer and statistics
        """
        pass

    def _start_timer(self, name: str):
        """Start timer"""
        self._timing_stats[f"{name}_start"] = time.time()

    def _end_timer(self, name: str):
        """End timer"""
        start_time = self._timing_stats.get(f"{name}_start", time.time())
        self._timing_stats[name] = time.time() - start_time

    def _record_memory(self, name: str):
        """Record current GPU memory usage"""
        if torch.cuda.is_available():
            self._memory_stats[name] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }

    def _clear_memory_stats(self):
        """Clear memory statistics"""
        self._memory_stats = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics"""
        return {k: v for k, v in self._timing_stats.items() if not k.endswith("_start")}

    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """Get memory statistics"""
        return self._memory_stats

    def reset_stats(self):
        """Reset all statistics"""
        self._timing_stats = {}
        self._clear_memory_stats()
