"""
LTC训练数据集

支持从GSM8K和MBPP等数据源加载训练数据。
"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LTCTrainingDataset(Dataset):
    """
    LTC训练数据集。

    数据来源(与评估benchmark不重叠):
    - GSM8K训练集: 数学推理
    - MBPP训练集: 代码生成
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
        初始化数据集。

        Args:
            data_sources: 数据源列表 ["gsm8k", "mbpp"]
            model_name_or_path: 模型路径（用于分词）
            cache_dir: 缓存目录
            max_samples_per_source: 每个数据源的最大样本数
            max_seq_length: 最大序列长度
            precompute_cache: 是否预计算KV Cache
        """
        self.data_sources = data_sources
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.max_samples_per_source = max_samples_per_source
        self.max_seq_length = max_seq_length

        # 加载分词器
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载数据
        self.samples = []
        self._load_data()

        # KV Cache缓存
        self.kv_cache_dir = os.path.join(cache_dir, "kv_cache")
        os.makedirs(self.kv_cache_dir, exist_ok=True)

    def _load_data(self):
        """加载所有数据源"""
        for source in self.data_sources:
            if source.lower() == "gsm8k":
                self._load_gsm8k()
            elif source.lower() == "mbpp":
                self._load_mbpp()
            else:
                print(f"Warning: Unknown data source {source}")

    def _load_gsm8k(self):
        """加载GSM8K数据集"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="train")

            for i, item in enumerate(dataset):
                if i >= self.max_samples_per_source:
                    break

                question = item["question"]
                answer = item["answer"]

                # 构建训练样本
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
        """加载MBPP数据集"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("mbpp", split="train")

            for i, item in enumerate(dataset):
                if i >= self.max_samples_per_source:
                    break

                prompt_text = item["text"]
                code = item["code"]

                # 构建训练样本
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
        获取单个样本。

        Returns:
            - input_ids: 输入token ids
            - attention_mask: 注意力掩码
            - prompt: 原始提示词
        """
        sample = self.samples[idx]

        # 分词
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
        """获取KV Cache缓存路径"""
        return os.path.join(self.kv_cache_dir, f"kv_cache_{idx}.pt")

    def has_cached_kv(self, idx: int) -> bool:
        """检查是否有缓存的KV Cache"""
        return os.path.exists(self.get_kv_cache_path(idx))

    def save_kv_cache(self, idx: int, kv_cache: Dict):
        """保存KV Cache到缓存"""
        torch.save(kv_cache, self.get_kv_cache_path(idx))

    def load_kv_cache(self, idx: int) -> Optional[Dict]:
        """从缓存加载KV Cache"""
        path = self.get_kv_cache_path(idx)
        if os.path.exists(path):
            return torch.load(path)
        return None


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """数据集的collate函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": [item["prompt"] for item in batch],
        "sources": [item["source"] for item in batch],
        "indices": [item["idx"] for item in batch],
    }
