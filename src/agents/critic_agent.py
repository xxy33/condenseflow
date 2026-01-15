"""
Critic Agent - 批评Agent

负责评估和批评其他Agent的输出，提供改进建议。
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
    批评Agent

    负责评估和批评其他Agent的输出，提供改进建议。
    在标准4-Agent流程中作为第二个Agent。
    在压力测试流程中与Solver交替工作。
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        初始化Critic Agent。

        Args:
            model_wrapper: LTC封装的模型
            communication_mode: 通信模式
            system_prompt: 自定义系统提示词，None则使用默认
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Critic",
            system_prompt=system_prompt or CRITIC_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        构建Critic特定的提示词。

        Args:
            question: 原始问题
            context: 来自上游Agent的输出

        Returns:
            完整的提示词
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
        """Critic的生成参数"""
        return {
            "max_new_tokens": 1024,
            "temperature": 0.5,  # 较低温度以获得更一致的评估
            "top_p": 0.95,
            "do_sample": True,
        }

    def is_correct(self, response: str) -> bool:
        """
        判断Critic是否认为答案正确。

        Args:
            response: Critic的响应文本

        Returns:
            True如果Critic认为答案正确
        """
        response_lower = response.lower()
        # 检查verdict
        if "verdict:" in response_lower:
            verdict_part = response_lower.split("verdict:")[-1].strip()
            return "correct" in verdict_part and "incorrect" not in verdict_part
        # 备用检查
        return "correct" in response_lower and "incorrect" not in response_lower
