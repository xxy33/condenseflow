"""
Memory Monitoring Utilities

Provides GPU and CPU memory usage monitoring functionality.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch


@dataclass
class MemorySnapshot:
    """Memory snapshot"""
    timestamp: float
    name: str
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0
    cpu_used_mb: float = 0.0


class MemoryTracker:
    """
    Memory tracker.

    Used to track memory usage during training and inference.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize memory tracker.

        Args:
            enabled: whether to enable tracking
        """
        self.enabled = enabled
        self.snapshots: List[MemorySnapshot] = []
        self._start_time = time.time()

    def snapshot(self, name: str) -> Optional[MemorySnapshot]:
        """
        Record current memory snapshot.

        Args:
            name: snapshot name

        Returns:
            memory snapshot object
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

        # CPU memory (requires psutil)
        try:
            import psutil
            process = psutil.Process()
            snapshot.cpu_used_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        self.snapshots.append(snapshot)
        return snapshot

    def reset(self):
        """Reset tracker"""
        self.snapshots = []
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage.

        Returns:
            peak memory information
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
        Get memory usage timeline.

        Returns:
            timeline data list
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
        """Print memory usage summary"""
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
        Compare memory usage across different modes.

        Args:
            mode_snapshots: mode name -> snapshot list

        Returns:
            comparison results
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
    Get current GPU memory information.

    Returns:
        GPU memory information dictionary
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
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
