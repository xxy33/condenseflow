"""
Agent Module Tests
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.agents.base_agent import BaseAgent, CommunicationMode


class TestCommunicationMode:
    """Communication mode tests"""

    def test_mode_values(self):
        """Test communication mode enum values"""
        assert CommunicationMode.TEXT.value == "text"
        assert CommunicationMode.DENSE_LATENT.value == "dense"
        assert CommunicationMode.CONDENSEFLOW.value == "condenseflow"

    def test_mode_from_string(self):
        """Test creating mode from string"""
        mode = CommunicationMode("condenseflow")
        assert mode == CommunicationMode.CONDENSEFLOW


class TestBaseAgent:
    """Base Agent tests"""

    def test_agent_init(self):
        """Test Agent initialization"""
        mock_wrapper = Mock()

        # Create a concrete Agent subclass for testing
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
        """Test prompt building"""
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
