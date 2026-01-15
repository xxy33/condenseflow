#!/usr/bin/env python
"""
Evaluation Entry Script
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.ltc_wrapper import LTCWrapper
from src.evaluation.evaluator import CondenseFlowEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CondenseFlow")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ltc_checkpoint", type=str, required=True)
    parser.add_argument("--eval_config", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["aime2024"])
    parser.add_argument("--communication_modes", type=str, nargs="+",
                        default=["text", "dense", "condenseflow"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    model_config = load_config(args.model_config)

    # Load evaluation configuration
    if args.eval_config:
        eval_config = load_config(args.eval_config)
        benchmarks = eval_config.get("evaluation", {}).get("benchmarks", args.benchmarks)
        modes = eval_config.get("evaluation", {}).get("communication_modes", args.communication_modes)
    else:
        benchmarks = args.benchmarks
        modes = args.communication_modes

    # Create model wrapper
    print("Loading model...")
    model_wrapper = LTCWrapper(
        model_name_or_path=model_config["model"]["name_or_path"],
        ltc_checkpoint=args.ltc_checkpoint,
        compression_dim=model_config.get("ltc", {}).get("compression_dim", 64),
    )

    # Create evaluator
    evaluator = CondenseFlowEvaluator(
        model_wrapper=model_wrapper,
        benchmarks=benchmarks,
        communication_modes=modes,
    )

    # Run evaluation
    print(f"Evaluating on: {benchmarks}")
    results = evaluator.run_evaluation(args.output_dir, num_runs=args.num_runs)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for benchmark, mode_results in results["accuracy"].items():
        print(f"\n{benchmark}:")
        for mode, stats in mode_results.items():
            print(f"  {mode}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
