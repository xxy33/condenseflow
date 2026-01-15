"""
Agent Base Class

Base class for agents supporting latent space communication, defines communication modes and basic interfaces.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch

from ..models.ltc_wrapper import LTCWrapper


class CommunicationMode(Enum):
    """Communication mode enumeration"""
    TEXT = "text"              # Pure text communication
    DENSE_LATENT = "dense"     # Full KV Cache transmission
    CONDENSEFLOW = "condenseflow"  # LTC compressed transmission


class BaseAgent(ABC):
    """
    Base class for agents supporting latent space communication.

    Communication modes:
    - TEXT: Pure text communication
    - DENSE_LATENT: Full KV Cache transmission
    - CONDENSEFLOW: LTC compressed transmission
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        role: str,
        system_prompt: str,
        communication_mode: str = "condenseflow"
    ):
        """
        Initialize Agent.

        Args:
            model_wrapper: LTC wrapped model
            role: Agent role name
            system_prompt: System prompt
            communication_mode: Communication mode ("text", "dense", "condenseflow")
        """
        self.model_wrapper = model_wrapper
        self.role = role
        self.system_prompt = system_prompt
        self.communication_mode = CommunicationMode(communication_mode)

        # Internal state
        self._last_kv_cache = None
        self._last_response = None

    def process(
        self,
        question: str,
        incoming_latent: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        incoming_text: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Process input and generate output.

        Args:
            question: Original question
            incoming_latent: Latent representation from upstream (for DENSE_LATENT and CONDENSEFLOW modes)
            incoming_text: Text from upstream (for TEXT mode)

        Returns:
            response_text: Generated text response
            outgoing_latent: Latent representation to pass downstream (if communication_mode != TEXT)
        """
        # Build prompt
        prompt = self.build_prompt(question, incoming_text)

        # Encode input
        input_ids = self.model_wrapper.encode_text(prompt)

        # Process based on communication mode
        if self.communication_mode == CommunicationMode.TEXT:
            response_text, _ = self.model_wrapper.generate_with_latent_input(
                input_ids=input_ids,
                compressed_cache=None,
                return_kv_cache=False,
                **self.get_generation_kwargs()
            )
            outgoing_latent = None

        elif self.communication_mode == CommunicationMode.DENSE_LATENT:
            # Full KV Cache transmission - no compression
            response_text, past_kv = self._generate_with_full_cache(
                input_ids=input_ids,
                incoming_cache=incoming_latent
            )
            outgoing_latent = past_kv

        else:  # CONDENSEFLOW
            response_text, outgoing_latent = self.model_wrapper.generate_with_latent_input(
                input_ids=input_ids,
                compressed_cache=incoming_latent,
                return_kv_cache=True,
                **self.get_generation_kwargs()
            )

        # Save state
        self._last_response = response_text
        self._last_kv_cache = outgoing_latent

        return response_text, outgoing_latent

    def _generate_with_full_cache(
        self,
        input_ids: torch.Tensor,
        incoming_cache: Optional[Tuple] = None
    ) -> Tuple[str, Tuple]:
        """
        Generate with full KV Cache (no compression).

        Args:
            input_ids: Input token ids
            incoming_cache: Full KV Cache from upstream

        Returns:
            response_text: Generated text
            past_key_values: Full KV Cache
        """
        input_ids = input_ids.to(self.model_wrapper.model.device)

        outputs = self.model_wrapper.model.generate(
            input_ids=input_ids,
            past_key_values=incoming_cache,
            max_new_tokens=self.get_generation_kwargs().get("max_new_tokens", 2048),
            return_dict_in_generate=True,
            use_cache=True,
            **{k: v for k, v in self.get_generation_kwargs().items() if k != "max_new_tokens"}
        )

        generated_ids = outputs.sequences[0]
        response_text = self.model_wrapper.tokenizer.decode(
            generated_ids[input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response_text, outputs.past_key_values

    def build_prompt(
        self,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build complete input prompt.

        Args:
            question: Original question
            context: Context information (text output from upstream Agent)

        Returns:
            Complete prompt
        """
        parts = [f"System: {self.system_prompt}\n"]

        if context:
            parts.append(f"Context from previous agent:\n{context}\n")

        parts.append(f"Question: {question}\n")
        parts.append(f"{self.role}: ")

        return "\n".join(parts)

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """
        Get generation parameters.

        Subclasses can override this method to customize generation parameters.
        """
        return {
            "max_new_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }

    @property
    def last_response(self) -> Optional[str]:
        """Get last response"""
        return self._last_response

    @property
    def last_kv_cache(self) -> Optional[Dict]:
        """Get last KV Cache"""
        return self._last_kv_cache

    def reset(self):
        """Reset Agent state"""
        self._last_kv_cache = None
        self._last_response = None
