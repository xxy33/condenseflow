#!/usr/bin/env python
"""
Compression Analysis Script
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.utils.config import load_config
from src.models.ltc_wrapper import LTCWrapper
from src.models.ltc import LatentThoughtCondenser


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze compression")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ltc_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis/compression")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser.parse_args()


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Compute effective rank."""
    U, S, V = torch.svd(matrix)
    normalized_S = S / S.sum()
    entropy = -torch.sum(normalized_S * torch.log(normalized_S + 1e-10))
    return torch.exp(entropy).item()


def analyze_probe_matrix(ltc: LatentThoughtCondenser) -> dict:
    """Analyze semantic probe matrix."""
    probes = ltc.get_probe_matrix()

    # Normalize
    probes_norm = torch.nn.functional.normalize(probes, dim=-1)

    # Compute probe similarity
    similarity = torch.mm(probes_norm, probes_norm.t())

    # Effective rank
    effective_rank = compute_effective_rank(probes)

    # Orthogonality metric
    identity = torch.eye(probes.shape[0], device=probes.device)
    orthogonality_error = torch.norm(similarity - identity).item()

    return {
        "effective_rank": effective_rank,
        "orthogonality_error": orthogonality_error,
        "mean_similarity": similarity.mean().item(),
        "max_off_diagonal": (similarity - identity).abs().max().item(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_config = load_config(args.model_config)

    print("Loading model...")
    model_wrapper = LTCWrapper(
        model_name_or_path=model_config["model"]["name_or_path"],
        ltc_checkpoint=args.ltc_checkpoint,
    )

    # Analyze probe matrix
    print("\nAnalyzing probe matrix...")
    probe_analysis = analyze_probe_matrix(model_wrapper.ltc)
    print(f"Effective Rank: {probe_analysis['effective_rank']:.2f}")
    print(f"Orthogonality Error: {probe_analysis['orthogonality_error']:.4f}")

    # Save results
    results = {
        "probe_analysis": probe_analysis,
        "compression_dim": model_wrapper.compression_dim,
        "kv_dim": model_wrapper.kv_dim,
    }

    with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
