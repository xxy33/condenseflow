"""
Communication Protocol Implementation

Defines protocols and utility functions for different communication modes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch


class CommunicationMode(Enum):
    """Communication mode"""
    TEXT = "text"
    DENSE_LATENT = "dense"
    CONDENSEFLOW = "condenseflow"


@dataclass
class Message:
    """Message passed between Agents"""
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
    Communication protocol manager.

    Manages message passing under different communication modes.
    """

    def __init__(self, mode: str = "condenseflow"):
        """
        Initialize communication protocol.

        Args:
            mode: Communication mode
        """
        self.mode = CommunicationMode(mode)

    def create_message(
        self,
        text: Optional[str] = None,
        latent: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> Message:
        """
        Create message.

        Args:
            text: Text content
            latent: Latent representation
            metadata: Metadata

        Returns:
            Message object
        """
        if self.mode == CommunicationMode.TEXT:
            return Message(text=text, latent=None, metadata=metadata)
        elif self.mode == CommunicationMode.DENSE_LATENT:
            return Message(text=text, latent=latent, metadata=metadata)
        else:  # CONDENSEFLOW
            return Message(text=text, latent=latent, metadata=metadata)

    def extract_for_downstream(self, message: Message) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Extract content needed by downstream Agent from message.

        Args:
            message: Message object

        Returns:
            (text, latent) tuple
        """
        if self.mode == CommunicationMode.TEXT:
            return message.text, None
        else:
            return message.text, message.latent

    @staticmethod
    def estimate_message_size(message: Message) -> Dict[str, float]:
        """
        Estimate message size.

        Args:
            message: Message object

        Returns:
            Dictionary containing size of each part (unit: MB)
        """
        sizes = {"text_mb": 0, "latent_mb": 0, "total_mb": 0}

        if message.text:
            # Assume UTF-8 encoding, average 2 bytes per character
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
        Compare costs of different communication modes.

        Args:
            text_only: Text-only message
            dense_latent: Full KV Cache message
            compressed_latent: Compressed message

        Returns:
            Comparison results
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
    Serialize latent representation (for distributed scenarios).

    Args:
        latent: Latent representation dictionary

    Returns:
        Serialized bytes
    """
    import io
    buffer = io.BytesIO()
    torch.save(latent, buffer)
    return buffer.getvalue()


def deserialize_latent(
    data: bytes
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Deserialize latent representation.

    Args:
        data: Serialized bytes

    Returns:
        Latent representation dictionary
    """
    import io
    buffer = io.BytesIO(data)
    return torch.load(buffer)
