# CondenseFlow

An efficient latent space collaboration framework for multi-agent LLM systems.

## Overview

CondenseFlow implements a **Latent Thought Condenser (LTC)** module that compresses KV Cache into fixed-size semantic representations, achieving **O(1) communication complexity** in multi-agent collaboration.

## Features

- **LTC Module**: Learnable semantic probes for KV Cache compression
- **O(1) Communication**: Fixed-size latent representations regardless of context length
- **Multi-Agent Pipelines**: Standard 4-agent and stress test pipelines
- **Multiple Benchmarks**: Support for AIME, GPQA, MBPP, MedQA evaluation

## Installation

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.ltc_wrapper import LTCWrapper
from src.pipelines.standard_pipeline import StandardPipeline

# Load model with LTC
model = LTCWrapper(
    model_name_or_path="meta-llama/Llama-3-8B",
    ltc_checkpoint="./checkpoints/ltc.pt"
)

# Run standard pipeline
pipeline = StandardPipeline(model, communication_mode="condenseflow")
result = pipeline.run("What is 2 + 2?")
print(result["answer"])
```

## Scripts Usage

### 1. Train LTC Module

Train the Latent Thought Condenser module:

```bash
python scripts/train_ltc.py \
    --model_config configs/model/llama3_8b.yaml \
    --training_config configs/training/default.yaml \
    --output_dir ./outputs/ltc_training \
    --use_wandb
```

**Parameters:**
- `--model_config`: Path to model configuration file (required)
- `--training_config`: Path to training configuration file (required)
- `--output_dir`: Output directory for checkpoints
- `--resume_from`: Resume training from checkpoint
- `--use_wandb`: Enable wandb logging

### 2. Evaluate Model

Evaluate CondenseFlow on benchmarks:

```bash
python scripts/evaluate.py \
    --model_config configs/model/llama3_8b.yaml \
    --ltc_checkpoint ./outputs/ltc_training/checkpoint.pt \
    --benchmarks aime2024 gpqa mbpp \
    --communication_modes text dense condenseflow \
    --num_runs 5
```

**Parameters:**
- `--model_config`: Path to model configuration file (required)
- `--ltc_checkpoint`: Path to trained LTC checkpoint (required)
- `--benchmarks`: List of benchmarks to evaluate (default: aime2024)
- `--communication_modes`: Communication modes to compare
- `--num_runs`: Number of evaluation runs (default: 5)

### 3. Analyze Compression

Analyze LTC compression quality:

```bash
python scripts/analyze_compression.py \
    --model_config configs/model/llama3_8b.yaml \
    --ltc_checkpoint ./outputs/ltc_training/checkpoint.pt \
    --output_dir ./analysis/compression
```

**Parameters:**
- `--model_config`: Path to model configuration file (required)
- `--ltc_checkpoint`: Path to trained LTC checkpoint (required)
- `--output_dir`: Output directory for analysis results
- `--num_samples`: Number of samples for analysis (default: 10)

### 4. Run Standard Protocol

Run the standard 4-agent collaboration pipeline (Planner → Critic → Refiner → Solver):

```bash
python scripts/run_standard_protocol.py \
    --model_config configs/model/llama3_8b.yaml \
    --ltc_checkpoint ./outputs/ltc_training/checkpoint.pt \
    --question "Solve this math problem..." \
    --communication_mode condenseflow \
    --verbose
```

**Parameters:**
- `--model_config`: Path to model configuration file (required)
- `--ltc_checkpoint`: Path to trained LTC checkpoint (required)
- `--question`: Single question to process
- `--benchmark`: Load questions from benchmark instead
- `--communication_mode`: Communication mode (default: condenseflow)

### 5. Run Stress Test

Test model robustness with iterative Solver-Critic interactions:

```bash
python scripts/run_stress_test.py \
    --model_config configs/model/llama3_8b.yaml \
    --ltc_checkpoint ./outputs/ltc_training/checkpoint.pt \
    --benchmark aime2024 \
    --max_rounds 20 \
    --communication_mode condenseflow
```

**Parameters:**
- `--model_config`: Path to model configuration file (required)
- `--ltc_checkpoint`: Path to trained LTC checkpoint (required)
- `--benchmark`: Benchmark to load questions from (default: aime2024)
- `--max_rounds`: Maximum iteration rounds (default: 20)

## Project Structure

```
condenseflow/
├── src/
│   ├── models/          # LTC module and model wrappers
│   ├── agents/          # Multi-agent implementations
│   ├── pipelines/       # Collaboration pipelines
│   ├── training/        # Training infrastructure
│   ├── evaluation/      # Evaluation and benchmarks
│   └── utils/           # Utility functions
├── scripts/             # Entry point scripts
├── configs/             # Configuration files
└── tests/               # Unit tests
```

## Communication Modes

| Mode | Description |
|------|-------------|
| `text` | Pure text communication between agents |
| `dense` | Full KV Cache transmission |
| `condenseflow` | LTC-compressed latent representations |

## License

MIT License
