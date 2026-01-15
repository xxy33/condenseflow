"""
Agent模块测试
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.agents.base_agent import BaseAgent, CommunicationMode


class TestCommunicationMode:
    """通信模式测试"""

    def test_mode_values(self):
        """测试通信模式枚举值"""
        assert CommunicationMode.TEXT.value == "text"
        assert CommunicationMode.DENSE_LATENT.value == "dense"
        assert CommunicationMode.CONDENSEFLOW.value == "condenseflow"

    def test_mode_from_string(self):
        """测试从字符串创建模式"""
        mode = CommunicationMode("condenseflow")
        assert mode == CommunicationMode.CONDENSEFLOW


class TestBaseAgent:
    """Agent基类测试"""

    def test_agent_init(self):
        """测试Agent初始化"""
        mock_wrapper = Mock()

        # 创建一个具体的Agent子类用于测试
        class TestAgent(BaseAgent):
            pass

        agent = TestAgent(
            model_wrapper=mock_wrapper,
            role="TestRole",
            system_prompt="Test prompt",
            communication_mode="text"
        )

        assert agent.role == "TestRole"
        assert agent.system_prompt == "Test prompt"
        assert agent.communication_mode == CommunicationMode.TEXT

    def test_build_prompt(self):
        """测试提示词构建"""
        mock_wrapper = Mock()

        class TestAgent(BaseAgent):
            pass

        agent = TestAgent(
            model_wrapper=mock_wrapper,
            role="Solver",
            system_prompt="You are a solver.",
            communication_mode="text"
        )

        prompt = agent.build_prompt("What is 2+2?", context="Previous analysis")

        assert "You are a solver." in prompt
        assert "What is 2+2?" in prompt
        assert "Previous analysis" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
