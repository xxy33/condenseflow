"""
Solver Agent - 求解Agent

负责生成最终答案。
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
    求解Agent

    负责生成最终答案。
    在标准4-Agent流程中作为最后一个Agent。
    在压力测试流程中与Critic交替工作。
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow",
        system_prompt: str = None
    ):
        """
        初始化Solver Agent。

        Args:
            model_wrapper: LTC封装的模型
            communication_mode: 通信模式
            system_prompt: 自定义系统提示词，None则使用默认
        """
        super().__init__(
            model_wrapper=model_wrapper,
            role="Solver",
            system_prompt=system_prompt or SOLVER_SYSTEM_PROMPT,
            communication_mode=communication_mode
        )

    def build_prompt(self, question: str, context: str = None) -> str:
        """
        构建Solver特定的提示词。

        Args:
            question: 原始问题
            context: 来自上游Agent的输出

        Returns:
            完整的提示词
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
        """Solver的生成参数"""
        return {
            "max_new_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
        }

    def extract_answer(self, response: str) -> str:
        """
        从响应中提取最终答案。

        Args:
            response: Solver的完整响应

        Returns:
            提取的答案
        """
        response_lower = response.lower()

        # 尝试找到 "Final Answer:" 部分
        if "final answer:" in response_lower:
            idx = response_lower.find("final answer:")
            answer_part = response[idx + len("final answer:"):].strip()
            # 取到下一个换行或结束
            if "\n\n" in answer_part:
                answer_part = answer_part.split("\n\n")[0]
            return answer_part.strip()

        # 尝试找到 "Answer:" 部分
        if "answer:" in response_lower:
            idx = response_lower.find("answer:")
            answer_part = response[idx + len("answer:"):].strip()
            if "\n\n" in answer_part:
                answer_part = answer_part.split("\n\n")[0]
            return answer_part.strip()

        # 如果没有明确标记，返回最后一段
        paragraphs = response.strip().split("\n\n")
        return paragraphs[-1].strip() if paragraphs else response.strip()
