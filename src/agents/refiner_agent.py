"""
Refiner Agent

Responsible for optimizing and improving solutions based on feedback.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ..models.ltc_wrapper import LTCWrapper


REFINER_SYSTEM_PROMPT = """You are a Refiner Agent. Your role is to improve solutions based on feedback.

Your responsibilities:
1. Address all issues identified by the Critic
2. Strengthen weak points in the reasoning
3. Add missing details or steps
4. Ensure the solution is complete and correct

Take the feedback seriously and make meaningful improvements while preserving what was already correct."""


class RefinerAgent(BaseAgent):
    """
    Refiner Agent

    Responsible for optimizing and improving solutions based on feedback.
    Acts as the third Agent in the standard 4-Agent workflow.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        Initialize Refiner Agent.

        Args:
            model_wrapper: LTC wrapped model
            communication_mode: Communication mode
            system_prompt: Custom system prompt, uses default if None
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Refiner",
            system_prompt=system_prompt or REFINER_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        Build Refiner-specific prompt.

        Args:
            question: Original question
            context: Feedback from Critic

        Returns:
            Complete prompt
        """
        prompt = f"""System: {self.system_prompt}

Original Problem:
{question}

"""
        if context:
            prompt += f"""Critic's Feedback:
{context}

Based on the feedback above, please provide an improved solution. Make sure to:
1. Address each issue mentioned by the Critic
2. Keep the parts that were correct
3. Provide clear reasoning for your improvements

"""
        else:
            prompt += """Please provide a refined approach to solve this problem.

"""

        prompt += "Refiner: "
        return prompt

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Refiner's generation parameters"""
        return {
            "max_new_tokens": 1536,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }
