#!/usr/bin/env python
"""
Stress Test Execution Script
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.ltc_wrapper import LTCWrapper
from src.pipelines.stress_test_pipeline import StressTestPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run stress test")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ltc_checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="aime2024")
    parser.add_argument("--max_rounds", type=int, default=20)
    parser.add_argument("--communication_mode", type=str, default="condenseflow")
    parser.add_argument("--output_dir", type=str, default="./results/stress_test")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_config = load_config(args.model_config)

    print("Loading model...")
    model_wrapper = LTCWrapper(
        model_name_or_path=model_config["model"]["name_or_path"],
        ltc_checkpoint=args.ltc_checkpoint,
    )

    pipeline = StressTestPipeline(
        model_wrapper=model_wrapper,
        max_rounds=args.max_rounds,
        communication_mode=args.communication_mode,
    )

    # Load test questions
    from src.evaluation.benchmarks import load_benchmark
    questions, _ = load_benchmark(args.benchmark)

    print(f"Running stress test with {args.max_rounds} rounds...")
    for i, question in enumerate(questions[:5]):
        print(f"\nQuestion {i+1}:")
        result = pipeline.run(question, verbose=args.verbose)
        print(f"Rounds: {result['num_rounds']}, Early stopped: {result['early_stopped']}")
        print(f"Answer: {result['answer'][:100]}...")


if __name__ == "__main__":
    main()
