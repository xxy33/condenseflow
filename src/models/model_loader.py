"""
Model Loading Utilities

Supports loading various pretrained LLM models.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration file.

    Args:
        config_path: YAML configuration file path

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string to torch.dtype.

    Args:
        dtype_str: Data type string

    Returns:
        torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def get_model_config(model_name_or_path: str) -> Dict[str, Any]:
    """
    Get model architecture configuration.

    Args:
        model_name_or_path: HuggingFace model path

    Returns:
        Dictionary containing model architecture information
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Extract key parameters
    model_config = {
        "num_layers": getattr(config, "num_hidden_layers", 32),
        "num_attention_heads": getattr(config, "num_attention_heads", 32),
        "num_kv_heads": getattr(config, "num_key_value_heads",
                                getattr(config, "num_attention_heads", 32)),
        "hidden_size": getattr(config, "hidden_size", 4096),
        "head_dim": getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads
                           if hasattr(config, "hidden_size") else 128),
    }

    # Calculate KV dimension
    model_config["kv_dim"] = model_config["num_kv_heads"] * model_config["head_dim"]

    return model_config


def load_model(
    model_name_or_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load pretrained model and tokenizer.

    Args:
        model_name_or_path: HuggingFace model path
        torch_dtype: Data type
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
        **kwargs: Other parameters

    Returns:
        (model, tokenizer) tuple
    """
    dtype = get_torch_dtype(torch_dtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    # Set to evaluation mode
    model.eval()

    return model, tokenizer


def load_model_from_config(config_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict]:
    """
    Load model from configuration file.

    Args:
        config_path: Configuration file path

    Returns:
        (model, tokenizer, config) tuple
    """
    config = load_model_config(config_path)
    model_config = config.get("model", {})

    model, tokenizer = load_model(
        model_name_or_path=model_config.get("name_or_path"),
        torch_dtype=model_config.get("torch_dtype", "bfloat16"),
        device_map=model_config.get("device_map", "auto"),
    )

    return model, tokenizer, config


def get_ltc_config_from_model(
    model_name_or_path: str,
    compression_dim: int = 64,
    init_std: float = 0.02
) -> Dict[str, Any]:
    """
    Automatically generate LTC configuration based on model.

    Args:
        model_name_or_path: HuggingFace model path
        compression_dim: Compression dimension
        init_std: Initialization standard deviation

    Returns:
        LTC configuration dictionary
    """
    model_config = get_model_config(model_name_or_path)

    return {
        "num_layers": model_config["num_layers"],
        "kv_dim": model_config["kv_dim"],
        "compression_dim": compression_dim,
        "init_std": init_std,
    }
