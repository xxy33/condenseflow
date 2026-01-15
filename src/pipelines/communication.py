"""
通信协议实现

定义不同通信模式的协议和工具函数。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch


class CommunicationMode(Enum):
    """通信模式"""
    TEXT = "text"
    DENSE_LATENT = "dense"
    CONDENSEFLOW = "condenseflow"


@dataclass
class Message:
    """Agent间传递的消息"""
    text: Optional[str] = None
    latent: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def has_text(self) -> bool:
        return self.text is not None

    @property
    def has_latent(self) -> bool:
        return self.latent is not None


class CommunicationProtocol:
    """
    通信协议管理器。

    负责管理不同通信模式下的消息传递。
    """

    def __init__(self, mode: str = "condenseflow"):
        """
        初始化通信协议。

        Args:
            mode: 通信模式
        """
        self.mode = CommunicationMode(mode)

    def create_message(
        self,
        text: Optional[str] = None,
        latent: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        创建消息。

        Args:
            text: 文本内容
            latent: 潜在表示
            metadata: 元数据

        Returns:
            Message对象
        """
        if self.mode == CommunicationMode.TEXT:
            return Message(text=text, latent=None, metadata=metadata)
        elif self.mode == CommunicationMode.DENSE_LATENT:
            return Message(text=text, latent=latent, metadata=metadata)
        else:  # CONDENSEFLOW
            return Message(text=text, latent=latent, metadata=metadata)

    def extract_for_downstream(self, message: Message) -> Tuple[Optional[str], Optional[Dict]]:
        """
        从消息中提取下游Agent需要的内容。

        Args:
            message: 消息对象

        Returns:
            (text, latent) 元组
        """
        if self.mode == CommunicationMode.TEXT:
            return message.text, None
        else:
            return message.text, message.latent

    @staticmethod
    def estimate_message_size(message: Message) -> Dict[str, float]:
        """
        估算消息大小。

        Args:
            message: 消息对象

        Returns:
            包含各部分大小的字典（单位：MB）
        """
        sizes = {"text_mb": 0, "latent_mb": 0, "total_mb": 0}

        if message.text:
            # 假设UTF-8编码，每个字符平均2字节
            sizes["text_mb"] = len(message.text) * 2 / 1024 / 1024

        if message.latent:
            latent_bytes = 0
            for layer_idx, (k, v) in message.latent.items():
                latent_bytes += k.numel() * k.element_size()
                latent_bytes += v.numel() * v.element_size()
            sizes["latent_mb"] = latent_bytes / 1024 / 1024

        sizes["total_mb"] = sizes["text_mb"] + sizes["latent_mb"]
        return sizes

    @staticmethod
    def compare_communication_costs(
        text_only: Message,
        dense_latent: Message,
        compressed_latent: Message
    ) -> Dict[str, Any]:
        """
        比较不同通信模式的成本。

        Args:
            text_only: 纯文本消息
            dense_latent: 完整KV Cache消息
            compressed_latent: 压缩后的消息

        Returns:
            比较结果
        """
        text_size = CommunicationProtocol.estimate_message_size(text_only)
        dense_size = CommunicationProtocol.estimate_message_size(dense_latent)
        compressed_size = CommunicationProtocol.estimate_message_size(compressed_latent)

        return {
            "text": text_size,
            "dense": dense_size,
            "compressed": compressed_size,
            "compression_ratio": dense_size["latent_mb"] / compressed_size["latent_mb"]
            if compressed_size["latent_mb"] > 0 else float('inf'),
            "savings_vs_dense_mb": dense_size["total_mb"] - compressed_size["total_mb"],
        }


def serialize_latent(
    latent: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> bytes:
    """
    序列化潜在表示（用于分布式场景）。

    Args:
        latent: 潜在表示字典

    Returns:
        序列化后的字节
    """
    import io
    buffer = io.BytesIO()
    torch.save(latent, buffer)
    return buffer.getvalue()


def deserialize_latent(
    data: bytes
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    反序列化潜在表示。

    Args:
        data: 序列化的字节

    Returns:
        潜在表示字典
    """
    import io
    buffer = io.BytesIO(data)
    return torch.load(buffer)
