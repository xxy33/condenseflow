"""
压力测试Pipeline

Solver <-> Critic 迭代交互，用于测试方法在长轮次协作下的鲁棒性。
"""

import time
from typing import Any, Dict, List, Optional

import torch

from .base_pipeline import BasePipeline
from ..models.ltc_wrapper import LTCWrapper
from ..agents import CriticAgent, SolverAgent


class StressTestPipeline(BasePipeline):
    """
    压力测试流程: Solver <-> Critic 迭代交互

    用于测试方法在长轮次协作下的鲁棒性。
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        max_rounds: int = 20,
        communication_mode: str = "condenseflow"
    ):
        """
        初始化压力测试Pipeline。

        Args:
            model_wrapper: LTC封装的模型
            max_rounds: 最大迭代轮数
            communication_mode: 通信模式
        """
        super().__init__(model_wrapper, communication_mode)

        self.max_rounds = max_rounds

        # 初始化Agent
        self.solver = SolverAgent(model_wrapper, communication_mode)
        self.critic = CriticAgent(model_wrapper, communication_mode)

    def run(
        self,
        question: str,
        early_stop: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        执行迭代协作流程。

        每轮:
        1. Solver生成/修正答案
        2. Critic评估答案
        3. 如果正确或达到max_rounds则停止

        Args:
            question: 输入问题
            early_stop: 如果Critic认为答案正确则提前停止
            verbose: 是否输出详细信息

        Returns:
            - answer: 最终答案
            - num_rounds: 实际迭代轮数
            - round_outputs: 每轮的输出
            - memory_stats: 内存使用统计
            - timing_stats: 时间统计
            - early_stopped: 是否提前停止
        """
        self.reset_stats()
        self._clear_memory_stats()

        round_outputs = []
        current_latent = None
        current_text = None
        early_stopped = False

        # 记录初始内存
        self._record_memory("initial")

        total_start = time.time()

        for round_idx in range(self.max_rounds):
            round_start = time.time()

            if verbose:
                print("=" * 50)
                print(f"Round {round_idx + 1}/{self.max_rounds}")
                print("=" * 50)

            round_data = {"round": round_idx + 1}

            # Solver生成/修正答案
            self._start_timer(f"solver_round_{round_idx}")
            solver_response, solver_latent = self.solver.process(
                question=question,
                incoming_latent=current_latent,
                incoming_text=current_text
            )
            self._end_timer(f"solver_round_{round_idx}")

            round_data["solver_response"] = solver_response
            current_latent = solver_latent
            current_text = solver_response

            if verbose:
                print(f"Solver:\n{solver_response[:300]}...\n")

            # Critic评估答案
            self._start_timer(f"critic_round_{round_idx}")
            critic_response, critic_latent = self.critic.process(
                question=question,
                incoming_latent=current_latent,
                incoming_text=current_text
            )
            self._end_timer(f"critic_round_{round_idx}")

            round_data["critic_response"] = critic_response
            current_latent = critic_latent
            current_text = critic_response

            if verbose:
                print(f"Critic:\n{critic_response[:300]}...\n")

            # 记录本轮内存
            self._record_memory(f"round_{round_idx}")

            round_data["time"] = time.time() - round_start
            round_outputs.append(round_data)

            # 检查是否应该提前停止
            if early_stop and self.critic.is_correct(critic_response):
                if verbose:
                    print(f"Early stopping at round {round_idx + 1}: Critic approved the answer")
                early_stopped = True
                break

        total_time = time.time() - total_start
        self._timing_stats["total"] = total_time

        # 提取最终答案（从最后一轮的Solver响应）
        final_solver_response = round_outputs[-1]["solver_response"]
        final_answer = self.solver.extract_answer(final_solver_response)

        return {
            "answer": final_answer,
            "full_response": final_solver_response,
            "num_rounds": len(round_outputs),
            "max_rounds": self.max_rounds,
            "early_stopped": early_stopped,
            "round_outputs": round_outputs,
            "memory_stats": self.get_memory_stats(),
            "timing_stats": self.get_timing_stats(),
        }

    def run_fixed_rounds(
        self,
        question: str,
        num_rounds: int,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        运行固定轮数的迭代。

        Args:
            question: 输入问题
            num_rounds: 固定轮数
            verbose: 是否输出详细信息

        Returns:
            结果字典
        """
        original_max = self.max_rounds
        self.max_rounds = num_rounds

        result = self.run(question, early_stop=False, verbose=verbose)

        self.max_rounds = original_max
        return result

    def analyze_memory_growth(
        self,
        question: str,
        num_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        分析内存随轮数增长的情况。

        Args:
            question: 输入问题
            num_rounds: 测试轮数

        Returns:
            内存增长分析结果
        """
        result = self.run_fixed_rounds(question, num_rounds, verbose=False)

        memory_stats = result["memory_stats"]
        rounds = sorted([k for k in memory_stats.keys() if k.startswith("round_")])

        memory_growth = []
        for round_key in rounds:
            stats = memory_stats[round_key]
            memory_growth.append({
                "round": int(round_key.split("_")[1]) + 1,
                "allocated_mb": stats["allocated_mb"],
                "max_allocated_mb": stats["max_allocated_mb"],
            })

        # 计算增长率
        if len(memory_growth) >= 2:
            first_mem = memory_growth[0]["allocated_mb"]
            last_mem = memory_growth[-1]["allocated_mb"]
            growth_rate = (last_mem - first_mem) / first_mem if first_mem > 0 else 0
        else:
            growth_rate = 0

        return {
            "memory_growth": memory_growth,
            "growth_rate": growth_rate,
            "initial_memory_mb": memory_growth[0]["allocated_mb"] if memory_growth else 0,
            "final_memory_mb": memory_growth[-1]["allocated_mb"] if memory_growth else 0,
            "communication_mode": self.communication_mode,
        }
