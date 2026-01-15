"""
Planner Agent

Responsible for analyzing problems and creating initial solution plans.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ..models.ltc_wrapper import LTCWrapper


PLANNER_SYSTEM_PROMPT = """You are a Planning Agent. Your role is to analyze the given problem and create a structured plan for solving it.

Your responsibilities:
1. Break down the problem into smaller, manageable steps
2. Identify key concepts and requirements
3. Outline a clear approach to solve the problem
4. Consider potential challenges and edge cases

Provide a clear, step-by-step plan that other agents can follow to solve the problem."""


class PlannerAgent(BaseAgent):
    """
    Planner Agent

    Responsible for analyzing problems and creating initial solution plans.
    Acts as the first agent in the standard 4-Agent pipeline.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        Initialize Planner Agent.

        Args:
            model_wrapper: LTC wrapped model
            communication_mode: Communication mode
            system_prompt: Custom system prompt, None uses default
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Planner",
            system_prompt=system_prompt or PLANNER_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        Build Planner-specific prompt.

        Args:
            question: Original question
            context: Context (Planner is usually first, so context is None)

        Returns:
            Complete prompt
        """
        prompt = f"""System: {self.system_prompt}

Problem to analyze:
{question}

Please provide a detailed plan to solve this problem. Structure your response as:
1. Problem Analysis: [Brief analysis of what the problem is asking]
2. Key Concepts: [Important concepts or knowledge needed]
3. Solution Steps: [Step-by-step approach]
4. Potential Challenges: [Things to watch out for]

Planner: """

        return prompt

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Generation parameters for Planner"""
        return {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }
