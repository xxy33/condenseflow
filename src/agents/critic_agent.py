"""
Critic Agent

Responsible for evaluating and criticizing other Agents' outputs, providing improvement suggestions.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ..models.ltc_wrapper import LTCWrapper


CRITIC_SYSTEM_PROMPT = """You are a Critic Agent. Your role is to carefully evaluate solutions and provide constructive feedback.

Your responsibilities:
1. Identify errors, inconsistencies, or gaps in the reasoning
2. Check for logical correctness and completeness
3. Verify that all requirements are addressed
4. Suggest specific improvements

Be thorough but constructive. Point out issues clearly and explain why they are problems."""


class CriticAgent(BaseAgent):
    """
    Critic Agent

    Responsible for evaluating and criticizing other Agents' outputs, providing improvement suggestions.
    Acts as the second Agent in the standard 4-Agent workflow.
    Works alternately with Solver in the stress test workflow.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        Initialize Critic Agent.

        Args:
            model_wrapper: LTC wrapped model
            communication_mode: Communication mode
            system_prompt: Custom system prompt, uses default if None
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Critic",
            system_prompt=system_prompt or CRITIC_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        Build Critic-specific prompt.

        Args:
            question: Original question
            context: Output from upstream Agent

        Returns:
            Complete prompt
        """
        prompt = f"""System: {self.system_prompt}

Original Problem:
{question}

"""
        if context:
            prompt += f"""Previous Agent's Response:
{context}

Please evaluate the above response and provide feedback. Structure your response as:
1. Correctness: [Is the reasoning/solution correct?]
2. Completeness: [Are all aspects addressed?]
3. Issues Found: [List any problems or errors]
4. Suggestions: [Specific improvements needed]
5. Verdict: [CORRECT/NEEDS_IMPROVEMENT/INCORRECT]

"""
        else:
            prompt += """No previous response to evaluate. Please analyze the problem and identify potential pitfalls.

"""

        prompt += "Critic: "
        return prompt

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Critic's generation parameters"""
        return {
            "max_new_tokens": 1024,
            "temperature": 0.5,  # Lower temperature for more consistent evaluation
            "top_p": 0.95,
            "do_sample": True,
        }

    def is_correct(self, response: str) -> bool:
        """
        Determine if Critic considers the answer correct.

        Args:
            response: Critic's response text

        Returns:
            True if Critic considers the answer correct
        """
        response_lower = response.lower()
        # Check verdict
        if "verdict:" in response_lower:
            verdict_part = response_lower.split("verdict:")[-1].strip()
            return "correct" in verdict_part and "incorrect" not in verdict_part
        # Fallback check
        return "correct" in response_lower and "incorrect" not in response_lower
