"""
Solver Agent

Responsible for generating the final answer.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ..models.ltc_wrapper import LTCWrapper


SOLVER_SYSTEM_PROMPT = """You are a Solver Agent. Your role is to provide the final, definitive answer to the problem.

Your responsibilities:
1. Synthesize all previous analysis and refinements
2. Execute the solution step by step
3. Provide a clear, correct final answer
4. Format the answer appropriately for the problem type

Be precise and thorough. Your answer should be the final, complete solution."""


class SolverAgent(BaseAgent):
    """
    Solver Agent

    Responsible for generating the final answer.
    Acts as the last Agent in the standard 4-Agent workflow.
    Works alternately with Critic in the stress test workflow.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        Initialize Solver Agent.

        Args:
            model_wrapper: LTC wrapped model
            communication_mode: Communication mode
            system_prompt: Custom system prompt, uses default if None
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Solver",
            system_prompt=system_prompt or SOLVER_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        Build Solver-specific prompt.

        Args:
            question: Original question
            context: Output from upstream Agent

        Returns:
            Complete prompt
        """
        prompt = f"""System: {self.system_prompt}

Problem:
{question}

"""
        if context:
            prompt += f"""Previous Analysis/Refinement:
{context}

Based on the above analysis, please provide the final solution. Structure your response as:
1. Solution Process: [Show your work step by step]
2. Final Answer: [Clear, formatted answer]

"""
        else:
            prompt += """Please solve this problem step by step and provide the final answer.

"""

        prompt += "Solver: "
        return prompt

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Solver's generation parameters"""
        return {
            "max_new_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }

    def extract_answer(self, response: str) -> str:
        """
        Extract final answer from response.

        Args:
            response: Solver's complete response

        Returns:
            Extracted answer
        """
        response_lower = response.lower()

        # Try to find "Final Answer:" section
        if "final answer:" in response_lower:
            idx = response_lower.find("final answer:")
            answer_part = response[idx + len("final answer:"):].strip()
            # Take until next double newline or end
            if "\n\n" in answer_part:
                answer_part = answer_part.split("\n\n")[0]
            return answer_part.strip()

        # Try to find "Answer:" section
        if "answer:" in response_lower:
            idx = response_lower.find("answer:")
            answer_part = response[idx + len("answer:"):].strip()
            if "\n\n" in answer_part:
                answer_part = answer_part.split("\n\n")[0]
            return answer_part.strip()

        # If no explicit marker, return last paragraph
        paragraphs = response.strip().split("\n\n")
        return paragraphs[-1].strip() if paragraphs else response.strip()
