"""
Refiner Agent - 优化Agent

负责根据批评意见优化和改进解决方案。
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
    优化Agent

    负责根据批评意见优化和改进解决方案。
    在标准4-Agent流程中作为第三个Agent。
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        初始化Refiner Agent。

        Args:
            model_wrapper: LTC封装的模型
            communication_mode: 通信模式
            system_prompt: 自定义系统提示词，None则使用默认
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Refiner",
            system_prompt=system_prompt or REFINER_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        构建Refiner特定的提示词。

        Args:
            question: 原始问题
            context: 来自Critic的反馈

        Returns:
            完整的提示词
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
        """Refiner的生成参数"""
        return {
            "max_new_tokens": 1536,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }
