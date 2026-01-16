"""
LTC Training Dataset

Supports loading training data from GSM8K, MBPP, and other data sources.
"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LTCTrainingDataset(Dataset):
    """
    LTC training dataset.

    Data sources (non-overlapping with evaluation benchmarks):
    - GSM8K training set: Math reasoning
    - MBPP training set: Code generation
    """

    def __init__(
        self,
        data_sources: List[str],
        model_name_or_path: str,
        cache_dir: str,
        max_samples_per_source: int = 1000,
        max_seq_length: int = 2048,
        precompute_cache: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_sources: List of data sources ["gsm8k", "mbpp"]
            model_name_or_path: Model path (for tokenization)
            cache_dir: Cache directory
            max_samples_per_source: Maximum samples per data source
            max_seq_length: Maximum sequence length
            precompute_cache: Whether to precompute KV Cache
        """
        self.data_sources = data_sources
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.max_samples_per_source = max_samples_per_source
        self.max_seq_length = max_seq_length

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.samples = []
        self._load_data()

        # KV Cache cache
        self.kv_cache_dir = os.path.join(cache_dir, "kv_cache")
        os.makedirs(self.kv_cache_dir, exist_ok=True)

    def _load_data(self):
        """Load all data sources"""
        for source in self.data_sources:
            if source.lower() == "gsm8k":
                self._load_gsm8k()
            elif source.lower() == "mbpp":
                self._load_mbpp()
            else:
                print(f"Warning: Unknown data source {source}")

    def _load_gsm8k(self):
        """Load GSM8K dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="train")

            for i, item in enumerate(dataset):
                if i >= self.max_samples_per_source:
                    break

                question = item["question"]
                answer = item["answer"]

                # Build training sample
                prompt = f"Question: {question}\n\nLet me solve this step by step.\n"

                self.samples.append({
                    "source": "gsm8k",
                    "question": question,
                    "answer": answer,
                    "prompt": prompt,
                })
        except Exception as e:
            print(f"Failed to load GSM8K: {e}")

    def _load_mbpp(self):
        """Load MBPP dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("mbpp", split="train")

            for i, item in enumerate(dataset):
                if i >= self.max_samples_per_source:
                    break

                prompt_text = item["text"]
                code = item["code"]

                # Build training sample
                prompt = f"Task: {prompt_text}\n\nLet me write the code.\n"

                self.samples.append({
                    "source": "mbpp",
                    "question": prompt_text,
                    "answer": code,
                    "prompt": prompt,
                })
        except Exception as e:
            print(f"Failed to load MBPP: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single sample.

        Returns:
            - input_ids: Input token ids
            - attention_mask: Attention mask
            - prompt: Original prompt
        """
        sample = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            sample["prompt"],
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": sample["prompt"],
            "source": sample["source"],
            "idx": idx,
        }

    def get_kv_cache_path(self, idx: int) -> str:
        """Get KV Cache cache path"""
        return os.path.join(self.kv_cache_dir, f"kv_cache_{idx}.pt")

    def has_cached_kv(self, idx: int) -> bool:
        """Check if cached KV Cache exists"""
        return os.path.exists(self.get_kv_cache_path(idx))

    def save_kv_cache(self, idx: int, kv_cache: Dict):
        """Save KV Cache to cache"""
        torch.save(kv_cache, self.get_kv_cache_path(idx))

    def load_kv_cache(self, idx: int) -> Optional[Dict]:
        """Load KV Cache from cache"""
        path = self.get_kv_cache_path(idx)
        if os.path.exists(path):
            return torch.load(path)
        return None


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Dataset collate function"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [item["prompt"] for item in batch],
        "sources": [item["source"] for item in batch],
        "indices": [item["idx"] for item in batch],
    }
