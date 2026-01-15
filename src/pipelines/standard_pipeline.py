"""
标准4-Agent协作Pipeline

Planner -> Critic -> Refiner -> Solver
"""

import time
from typing import Any, Dict, List, Optional

import torch

from .base_pipeline import BasePipeline
from ..models.ltc_wrapper import LTCWrapper
from ..agents import PlannerAgent, CriticAgent, RefinerAgent, SolverAgent


class StandardPipeline(BasePipeline):
    """
    标准4-Agent协作流程: Planner -> Critic -> Refiner -> Solver

    通信协议:
    - 每个Agent接收上游的压缩KV Cache
    - 处理后将自己的KV Cache压缩后传给下游
    - 只有最终Solver输出文本答案
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        communication_mode: str = "condenseflow"
    ):
        """
        初始化标准Pipeline。

        Args:
            model_wrapper: LTC封装的模型
            communication_mode: 通信模式
        """
        super().__init__(model_wrapper, communication_mode)

        # 初始化4个Agent
        self.planner = PlannerAgent(model_wrapper, communication_mode)
        self.critic = CriticAgent(model_wrapper, communication_mode)
        self.refiner = RefinerAgent(model_wrapper, communication_mode)
        self.solver = SolverAgent(model_wrapper, communication_mode)

        self.agents = [self.planner, self.critic, self.refiner, self.solver]

    def run(
        self,
        question: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        执行完整协作流程。

        Args:
            question: 输入问题
            verbose: 是否输出详细信息

        Returns:
            - answer: 最终答案
            - intermediate_outputs: 各Agent的中间输出
            - memory_stats: 内存使用统计
            - timing_stats: 时间统计
        """
        self.reset_stats()
        self._clear_memory_stats()

        intermediate_outputs = {}
        current_latent = None
        current_text = None

        # 记录初始内存
        self._record_memory("initial")

        total_start = time.time()

        # 1. Planner
        if verbose:
            print("=" * 50)
            print("Stage 1: Planner")
            print("=" * 50)

        self._start_timer("planner")
        planner_response, planner_latent = self.planner.process(
            question=question,
            incoming_latent=None,
            incoming_text=None
        )
        self._end_timer("planner")
        self._record_memory("after_planner")

        intermediate_outputs["planner"] = planner_response
        current_latent = planner_latent
        current_text = planner_response

        if verbose:
            print(f"Planner output:\n{planner_response[:500]}...")
            print()

        # 2. Critic
        if verbose:
            print("=" * 50)
            print("Stage 2: Critic")
            print("=" * 50)

        self._start_timer("critic")
        critic_response, critic_latent = self.critic.process(
            question=question,
            incoming_latent=current_latent,
            incoming_text=current_text
        )
        self._end_timer("critic")
        self._record_memory("after_critic")

        intermediate_outputs["critic"] = critic_response
        current_latent = critic_latent
        current_text = critic_response

        if verbose:
            print(f"Critic output:\n{critic_response[:500]}...")
            print()

        # 3. Refiner
        if verbose:
            print("=" * 50)
            print("Stage 3: Refiner")
            print("=" * 50)

        self._start_timer("refiner")
        refiner_response, refiner_latent = self.refiner.process(
            question=question,
            incoming_latent=current_latent,
            incoming_text=current_text
        )
        self._end_timer("refiner")
        self._record_memory("after_refiner")

        intermediate_outputs["refiner"] = refiner_response
        current_latent = refiner_latent
        current_text = refiner_response

        if verbose:
            print(f"Refiner output:\n{refiner_response[:500]}...")
            print()

        # 4. Solver
        if verbose:
            print("=" * 50)
            print("Stage 4: Solver")
            print("=" * 50)

        self._start_timer("solver")
        solver_response, _ = self.solver.process(
            question=question,
            incoming_latent=current_latent,
            incoming_text=current_text
        )
        self._end_timer("solver")
        self._record_memory("after_solver")

        intermediate_outputs["solver"] = solver_response

        if verbose:
            print(f"Solver output:\n{solver_response}")
            print()

        total_time = time.time() - total_start
        self._timing_stats["total"] = total_time

        # 提取最终答案
        final_answer = self.solver.extract_answer(solver_response)

        return {
            "answer": final_answer,
            "full_response": solver_response,
            "intermediate_outputs": intermediate_outputs,
            "memory_stats": self.get_memory_stats(),
            "timing_stats": self.get_timing_stats(),
        }

    def run_with_custom_agents(
        self,
        question: str,
        agent_sequence: List[str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        使用自定义Agent序列运行。

        Args:
            question: 输入问题
            agent_sequence: Agent名称列表，如 ["planner", "critic", "solver"]
            verbose: 是否输出详细信息

        Returns:
            结果字典
        """
        agent_map = {
            "planner": self.planner,
            "critic": self.critic,
            "refiner": self.refiner,
            "solver": self.solver,
        }

        self.reset_stats()
        intermediate_outputs = {}
        current_latent = None
        current_text = None

        total_start = time.time()

        for i, agent_name in enumerate(agent_sequence):
            agent = agent_map.get(agent_name.lower())
            if agent is None:
                raise ValueError(f"Unknown agent: {agent_name}")

            if verbose:
                print(f"Stage {i+1}: {agent_name.capitalize()}")

            self._start_timer(agent_name)
            response, latent = agent.process(
                question=question,
                incoming_latent=current_latent,
                incoming_text=current_text
            )
            self._end_timer(agent_name)
            self._record_memory(f"after_{agent_name}")

            intermediate_outputs[agent_name] = response
            current_latent = latent
            current_text = response

            if verbose:
                print(f"Output:\n{response[:300]}...\n")

        self._timing_stats["total"] = time.time() - total_start

        # 最后一个Agent的输出作为答案
        last_agent = agent_sequence[-1].lower()
        final_response = intermediate_outputs[last_agent]

        # 如果最后是solver，提取答案
        if last_agent == "solver":
            final_answer = self.solver.extract_answer(final_response)
        else:
            final_answer = final_response

        return {
            "answer": final_answer,
            "full_response": final_response,
            "intermediate_outputs": intermediate_outputs,
            "memory_stats": self.get_memory_stats(),
            "timing_stats": self.get_timing_stats(),
        }
