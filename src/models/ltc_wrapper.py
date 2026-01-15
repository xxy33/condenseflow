"""
LTC Wrapper - Wrapper class for integrating LTC module with pretrained LLMs

Provides generation interface with compression, manages KV Cache injection and extraction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ltc import LatentThoughtCondenser
from .model_loader import load_model, get_model_config


class LTCWrapper:
    """
    Wrapper class for integrating LTC module with pretrained LLMs.

    Features:
    - Load pretrained LLM and trained LTC module
    - Provide generation interface with compression
    - Manage KV Cache injection and extraction
    """

    def __init__(
        self,
        model_name_or_path: str,
        ltc_checkpoint: Optional[str] = None,
        compression_dim: int = 64,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize LTC Wrapper.

        Args:
            model_name_or_path: HuggingFace model path
            ltc_checkpoint: LTC module checkpoint path, None to create new LTC
            compression_dim: Compression dimension
            device: Device
            torch_dtype: Data type
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.compression_dim = compression_dim

        # Load model and tokenizer
        self.model, self.tokenizer = load_model(
            model_name_or_path,
            torch_dtype=str(torch_dtype).split('.')[-1],
            device_map="auto"
        )

        # Get model configuration
        self.model_config = get_model_config(model_name_or_path)
        self.num_layers = self.model_config["num_layers"]
        self.num_kv_heads = self.model_config["num_kv_heads"]
        self.head_dim = self.model_config["head_dim"]
        self.kv_dim = self.model_config["kv_dim"]

        # Initialize LTC module
        self.ltc = LatentThoughtCondenser(
            num_layers=self.num_layers,
            kv_dim=self.kv_dim,
            compression_dim=compression_dim
        )

        # Load LTC checkpoint
        if ltc_checkpoint:
            self.load_ltc_checkpoint(ltc_checkpoint)

        # Move LTC to device
        self.ltc = self.ltc.to(device).to(torch_dtype)
        self.ltc.eval()

    def load_ltc_checkpoint(self, checkpoint_path: str):
        """Load LTC module checkpoint"""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.ltc.load_state_dict(state_dict)

    def save_ltc_checkpoint(self, checkpoint_path: str):
        """Save LTC module checkpoint"""
        torch.save(self.ltc.state_dict(), checkpoint_path)

    @torch.no_grad()
    def generate_with_latent_input(
        self,
        input_ids: torch.Tensor,
        compressed_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_new_tokens: int = 2048,
        return_kv_cache: bool = True,
        **generation_kwargs
    ) -> Tuple[str, Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Perform generation with latent input.

        Args:
            input_ids: Input token ids [batch, seq_len]
            compressed_cache: Compressed cache from upstream Agent
            max_new_tokens: Maximum number of tokens to generate
            return_kv_cache: Whether to return compressed KV Cache
            **generation_kwargs: Other generation parameters

        Returns:
            response_text: Generated text
            compressed_output_cache: Compressed output KV Cache (if return_kv_cache=True)
        """
        input_ids = input_ids.to(self.model.device)

        # Prepare past_key_values
        past_key_values = None
        if compressed_cache is not None:
            past_key_values = self._convert_to_hf_format(compressed_cache)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=False,
            use_cache=True,
            **generation_kwargs
        )

        # Decode generated text
        generated_ids = outputs.sequences[0]
        response_text = self.tokenizer.decode(
            generated_ids[input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract and compress KV Cache
        compressed_output_cache = None
        if return_kv_cache:
            compressed_output_cache = self.extract_and_compress_cache()

        return response_text, compressed_output_cache

    def extract_and_compress_cache(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract current KV Cache and compress using LTC.

        Returns:
            Compressed KV Cache
        """
        # Extract KV Cache from model
        # Note: This needs to be called immediately after generate
        past_key_values = getattr(self.model, '_last_past_key_values', None)

        if past_key_values is None:
            raise RuntimeError("No KV cache available. Make sure to call this after generation.")

        # Convert to LTC format
        kv_cache = self._convert_from_hf_format(past_key_values)

        # Compress
        compressed_cache, _ = self.ltc(kv_cache)

        return compressed_cache

    def compress_kv_cache(
        self,
        past_key_values: Tuple
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress given KV Cache.

        Args:
            past_key_values: HuggingFace format past_key_values

        Returns:
            Compressed KV Cache
        """
        kv_cache = self._convert_from_hf_format(past_key_values)
        compressed_cache, _ = self.ltc(kv_cache)
        return compressed_cache

    def _convert_from_hf_format(
        self,
        past_key_values: Tuple
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert HuggingFace format KV Cache to LTC input format.

        HuggingFace format: tuple of (key, value) per layer
        key/value shape: [batch, num_kv_heads, seq_len, head_dim]

        LTC format: Dict[layer_idx, (key, value)]
        key/value shape: [batch, seq_len, kv_dim]
        """
        kv_cache = {}

        for layer_idx, (key, value) in enumerate(past_key_values):
            batch_size, num_heads, seq_len, head_dim = key.shape

            # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads * head_dim]
            key_reshaped = key.transpose(1, 2).reshape(batch_size, seq_len, -1)
            value_reshaped = value.transpose(1, 2).reshape(batch_size, seq_len, -1)

            kv_cache[layer_idx] = (key_reshaped, value_reshaped)

        return kv_cache

    def _convert_to_hf_format(
        self,
        compressed_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple:
        """
        Convert LTC format compressed Cache back to HuggingFace format.

        LTC format: Dict[layer_idx, (key, value)]
        key/value shape: [batch, compression_dim, kv_dim]

        HuggingFace format: tuple of (key, value) per layer
        key/value shape: [batch, num_kv_heads, compression_dim, head_dim]
        """
        past_key_values = []

        for layer_idx in range(self.num_layers):
            key, value = compressed_cache[layer_idx]
            batch_size, comp_dim, kv_dim = key.shape

            # Reshape: [batch, comp_dim, kv_dim] -> [batch, num_kv_heads, comp_dim, head_dim]
            key_reshaped = key.view(batch_size, comp_dim, self.num_kv_heads, self.head_dim)
            key_reshaped = key_reshaped.transpose(1, 2)

            value_reshaped = value.view(batch_size, comp_dim, self.num_kv_heads, self.head_dim)
            value_reshaped = value_reshaped.transpose(1, 2)

            past_key_values.append((key_reshaped, value_reshaped))

        return tuple(past_key_values)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token ids"""
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        compressed_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Perform forward pass and return KV Cache.

        Args:
            input_ids: Input token ids
            compressed_cache: Compressed KV Cache prefix
            attention_mask: Attention mask

        Returns:
            logits: Output logits
            past_key_values: Complete KV Cache
        """
        input_ids = input_ids.to(self.model.device)

        past_key_values = None
        if compressed_cache is not None:
            past_key_values = self._convert_to_hf_format(compressed_cache)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )

        return outputs.logits, outputs.past_key_values
