#!/usr/bin/env python
"""
Standard Protocol Execution Script
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.ltc_wrapper import LTCWrapper
from src.pipelines.standard_pipeline import StandardPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard protocol")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ltc_checkpoint", type=str, required=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--communication_mode", type=str, default="condenseflow")
    parser.add_argument("--output_dir", type=str, default="./results/standard")
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

    pipeline = StandardPipeline(
        model_wrapper=model_wrapper,
        communication_mode=args.communication_mode,
    )

    if args.question:
        questions = [args.question]
    elif args.benchmark:
        from src.evaluation.benchmarks import load_benchmark
        questions, _ = load_benchmark(args.benchmark)
        questions = questions[:5]
    else:
        questions = ["What is 2 + 2?"]

    for i, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Question {i+1}: {question[:100]}...")
        print("="*60)

        result = pipeline.run(question, verbose=args.verbose)

        print(f"\nFinal Answer: {result['answer']}")
        print(f"Total Time: {result['timing_stats'].get('total', 0):.2f}s")


if __name__ == "__main__":
    main()
