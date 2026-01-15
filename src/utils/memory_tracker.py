"""
内存监控工具

提供GPU和CPU内存使用监控功能。
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    name: str
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0
    cpu_used_mb: float = 0.0


class MemoryTracker:
    """
    内存监控器。

    用于跟踪训练和推理过程中的内存使用情况。
    """

    def __init__(self, enabled: bool = True):
        """
        初始化内存监控器。

        Args:
            enabled: 是否启用监控
        """
        self.enabled = enabled
        self.snapshots: List[MemorySnapshot] = []
        self._start_time = time.time()

    def snapshot(self, name: str) -> Optional[MemorySnapshot]:
        """
        记录当前内存快照。

        Args:
            name: 快照名称

        Returns:
            内存快照对象
        """
        if not self.enabled:
            return None

        snapshot = MemorySnapshot(
            timestamp=time.time() - self._start_time,
            name=name,
        )

        if torch.cuda.is_available():
            snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            snapshot.gpu_max_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # CPU内存（需要psutil）
        try:
            import psutil
            process = psutil.Process()
            snapshot.cpu_used_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        self.snapshots.append(snapshot)
        return snapshot

    def reset(self):
        """重置监控器"""
        self.snapshots = []
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self) -> Dict[str, float]:
        """
        获取峰值内存使用。

        Returns:
            峰值内存信息
        """
        if not self.snapshots:
            return {}

        peak_gpu = max(s.gpu_max_allocated_mb for s in self.snapshots)
        peak_cpu = max(s.cpu_used_mb for s in self.snapshots)

        return {
            "peak_gpu_mb": peak_gpu,
            "peak_cpu_mb": peak_cpu,
        }

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """
        获取内存使用时间线。

        Returns:
            时间线数据列表
        """
        return [
            {
                "timestamp": s.timestamp,
                "name": s.name,
                "gpu_allocated_mb": s.gpu_allocated_mb,
                "gpu_reserved_mb": s.gpu_reserved_mb,
                "cpu_used_mb": s.cpu_used_mb,
            }
            for s in self.snapshots
        ]

    def print_summary(self):
        """打印内存使用摘要"""
        if not self.snapshots:
            print("No memory snapshots recorded.")
            return

        print("\n" + "=" * 60)
        print("Memory Usage Summary")
        print("=" * 60)

        for snapshot in self.snapshots:
            print(f"\n[{snapshot.name}] @ {snapshot.timestamp:.2f}s")
            print(f"  GPU Allocated: {snapshot.gpu_allocated_mb:.2f} MB")
            print(f"  GPU Reserved:  {snapshot.gpu_reserved_mb:.2f} MB")
            print(f"  GPU Peak:      {snapshot.gpu_max_allocated_mb:.2f} MB")
            if snapshot.cpu_used_mb > 0:
                print(f"  CPU Used:      {snapshot.cpu_used_mb:.2f} MB")

        peak = self.get_peak_memory()
        print("\n" + "-" * 60)
        print(f"Peak GPU Memory: {peak.get('peak_gpu_mb', 0):.2f} MB")
        print(f"Peak CPU Memory: {peak.get('peak_cpu_mb', 0):.2f} MB")
        print("=" * 60)

    def compare_modes(
        self,
        mode_snapshots: Dict[str, List[MemorySnapshot]]
    ) -> Dict[str, Any]:
        """
        比较不同模式的内存使用。

        Args:
            mode_snapshots: 模式名称 -> 快照列表

        Returns:
            比较结果
        """
        comparison = {}

        for mode, snapshots in mode_snapshots.items():
            if snapshots:
                peak_gpu = max(s.gpu_max_allocated_mb for s in snapshots)
                avg_gpu = sum(s.gpu_allocated_mb for s in snapshots) / len(snapshots)
                comparison[mode] = {
                    "peak_gpu_mb": peak_gpu,
                    "avg_gpu_mb": avg_gpu,
                    "num_snapshots": len(snapshots),
                }

        return comparison


def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取当前GPU内存信息。

    Returns:
        GPU内存信息字典
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
    }


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
