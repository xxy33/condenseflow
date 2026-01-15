#!/usr/bin/env python
"""
LTC Module Training Entry Script
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, merge_configs
from src.training.trainer import LTCTrainer
from src.training.dataset import LTCTrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train LTC module")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model config file")
    parser.add_argument("--training_config", type=str, required=True,
                        help="Path to training config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="condenseflow",
                        help="Wandb project name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)

    # Merge configurations
    config = merge_configs(model_config, training_config)

    # Override output directory
    output_dir = args.output_dir or config.get("training", {}).get("output_dir", "./outputs")

    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=config)
        except ImportError:
            print("wandb not installed, skipping")

    # Create dataset
    print("Loading training dataset...")
    train_dataset = LTCTrainingDataset(
        data_sources=config.get("training", {}).get("data_sources", ["gsm8k"]),
        model_name_or_path=config["model"]["name_or_path"],
        cache_dir=os.path.join(output_dir, "cache"),
        max_samples_per_source=config.get("training", {}).get("max_samples_per_source", 1000),
        max_seq_length=config.get("training", {}).get("max_seq_length", 2048),
    )
    print(f"Loaded {len(train_dataset)} training samples")

    # Create trainer
    trainer = LTCTrainer(
        model_name_or_path=config["model"]["name_or_path"],
        ltc_config=config.get("ltc", {}),
        training_config=config.get("training", {}),
        output_dir=output_dir,
    )

    # Resume training
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Start training
    print("Starting training...")
    trainer.train(train_dataset)
    print("Training completed!")


if __name__ == "__main__":
    main()
