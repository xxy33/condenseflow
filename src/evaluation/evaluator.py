"""
CondenseFlow Evaluator

Evaluates the performance of different communication modes on various benchmarks.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from ..models.ltc_wrapper import LTCWrapper
from ..pipelines import StandardPipeline, StressTestPipeline
from .metrics import compute_accuracy, compute_efficiency_metrics


class CondenseFlowEvaluator:
    """
    CondenseFlow evaluator.

    Evaluation dimensions:
    - Accuracy: Task accuracy on various benchmarks
    - Efficiency: Memory usage, inference time
    - Compression quality: Effective rank analysis
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        benchmarks: List[str],
        protocol: str = "standard",
        communication_modes: List[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            model_wrapper: LTC wrapped model
            benchmarks: List of evaluation benchmarks
            protocol: Evaluation protocol ("standard" or "stress_test")
            communication_modes: List of communication modes
        """
        self.model_wrapper = model_wrapper
        self.benchmarks = benchmarks
        self.protocol = protocol
        self.communication_modes = communication_modes or ["text", "dense", "condenseflow"]

    def run_evaluation(
        self,
        output_dir: str,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Execute complete evaluation.

        Args:
            output_dir: Output directory
            num_runs: Number of runs

        Returns:
            Evaluation results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "accuracy": {},
            "efficiency": {},
            "per_sample_results": [],
        }

        for benchmark in self.benchmarks:
            print(f"\nEvaluating on {benchmark}...")
            results["accuracy"][benchmark] = {}

            for mode in self.communication_modes:
                print(f"  Mode: {mode}")
                mode_results = []

                for run_idx in range(num_runs):
                    run_result = self.evaluate_single_benchmark(benchmark, mode)
                    mode_results.append(run_result)

                # Compute mean and standard deviation
                accuracies = [r["accuracy"] for r in mode_results]
                mean_acc = sum(accuracies) / len(accuracies)
                std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5

                results["accuracy"][benchmark][mode] = {
                    "mean": mean_acc,
                    "std": std_acc,
                    "runs": accuracies,
                }

        # Efficiency evaluation
        for mode in self.communication_modes:
            results["efficiency"][mode] = self.compute_efficiency_metrics(mode, num_rounds=5)

        # Save results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def evaluate_single_benchmark(
        self,
        benchmark_name: str,
        communication_mode: str
    ) -> Dict[str, Any]:
        """
        Evaluate single benchmark.

        Args:
            benchmark_name: Benchmark name
            communication_mode: Communication mode

        Returns:
            Evaluation results
        """
        # Load benchmark data
        questions, references = self._load_benchmark(benchmark_name)

        # Create pipeline
        if self.protocol == "standard":
            pipeline = StandardPipeline(self.model_wrapper, communication_mode)
        else:
            pipeline = StressTestPipeline(self.model_wrapper, communication_mode=communication_mode)

        predictions = []
        for question in tqdm(questions, desc=f"{benchmark_name}/{communication_mode}"):
            result = pipeline.run(question, verbose=False)
            predictions.append(result["answer"])

        # Compute accuracy
        task_type = self._get_task_type(benchmark_name)
        accuracy_result = compute_accuracy(predictions, references, task_type)

        return {
            "accuracy": accuracy_result["accuracy"],
            "correct": accuracy_result["correct"],
            "total": accuracy_result["total"],
            "predictions": predictions,
        }

    def _load_benchmark(self, benchmark_name: str):
        """Load benchmark data"""
        from .benchmarks import load_benchmark
        return load_benchmark(benchmark_name)

    def _get_task_type(self, benchmark_name: str) -> str:
        """Get task type"""
        if "aime" in benchmark_name.lower() or "math" in benchmark_name.lower():
            return "math"
        elif "gpqa" in benchmark_name.lower() or "medqa" in benchmark_name.lower():
            return "mcq"
        else:
            return "code"

    def compute_efficiency_metrics(
        self,
        communication_mode: str,
        num_rounds: int
    ) -> Dict[str, float]:
        """Compute efficiency metrics"""
        pipeline = StressTestPipeline(
            self.model_wrapper,
            max_rounds=num_rounds,
            communication_mode=communication_mode
        )

        test_question = "What is 2 + 2?"
        result = pipeline.run(test_question, early_stop=False, verbose=False)

        return {
            "total_time_seconds": result["timing_stats"].get("total", 0),
            "memory_stats": result["memory_stats"],
        }
