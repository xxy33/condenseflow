"""
Stress Test Pipeline

Solver <-> Critic iterative interaction for testing method robustness under long-round collaboration.
"""

import time
from typing import Any, Dict, List, Optional

import torch

from .base_pipeline import BasePipeline
from ..models.ltc_wrapper import LTCWrapper
from ..agents import CriticAgent, SolverAgent


class StressTestPipeline(BasePipeline):
    """
    Stress test workflow: Solver <-> Critic iterative interaction

    Used for testing method robustness under long-round collaboration.
    """

    def __init__(
        self,
        model_wrapper: LTCWrapper,
        max_rounds: int = 20,
        communication_mode: str = "condenseflow"
    ):
        """
        Initialize stress test Pipeline.

        Args:
            model_wrapper: LTC wrapped model
            max_rounds: Maximum iteration rounds
            communication_mode: Communication mode
        """
        super().__init__(model_wrapper, communication_mode)

        self.max_rounds = max_rounds

        # Initialize Agents
        self.solver = SolverAgent(model_wrapper, communication_mode)
        self.critic = CriticAgent(model_wrapper, communication_mode)

    def run(
        self,
        question: str,
        early_stop: bool = True,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute iterative collaboration workflow.

        Each round:
        1. Solver generates/corrects answer
        2. Critic evaluates answer
        3. Stop if correct or max_rounds reached

        Args:
            question: Input question
            early_stop: Stop early if Critic approves the answer
            verbose: Whether to output detailed information

        Returns:
            - answer: Final answer
            - num_rounds: Actual iteration rounds
            - round_outputs: Outputs for each round
            - memory_stats: Memory usage statistics
            - timing_stats: Timing statistics
            - early_stopped: Whether stopped early
        """
        self.reset_stats()
        self._clear_memory_stats()

        round_outputs = []
        current_latent = None
        current_text = None
        early_stopped = False

        # Record initial memory
        self._record_memory("initial")

        total_start = time.time()

        for round_idx in range(self.max_rounds):
            round_start = time.time()

            if verbose:
                print("=" * 50)
                print(f"Round {round_idx + 1}/{self.max_rounds}")
                print("=" * 50)

            round_data = {"round": round_idx + 1}

            # Solver generates/corrects answer
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

            # Critic evaluates answer
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

            # Record this round's memory
            self._record_memory(f"round_{round_idx}")

            round_data["time"] = time.time() - round_start
            round_outputs.append(round_data)

            # Check if should stop early
            if early_stop and self.critic.is_correct(critic_response):
                if verbose:
                    print(f"Early stopping at round {round_idx + 1}: Critic approved the answer")
                early_stopped = True
                break

        total_time = time.time() - total_start
        self._timing_stats["total"] = total_time

        # Extract final answer (from last round's Solver response)
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
        Run fixed number of iteration rounds.

        Args:
            question: Input question
            num_rounds: Fixed number of rounds
            verbose: Whether to output detailed information

        Returns:
            Result dictionary
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
        Analyze memory growth over rounds.

        Args:
            question: Input question
            num_rounds: Number of test rounds

        Returns:
            Memory growth analysis results
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

        # Calculate growth rate
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
