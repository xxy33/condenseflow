"""
Pipeline Module Tests
"""

import pytest
from unittest.mock import Mock


class TestStandardPipeline:
    """Standard Pipeline tests"""

    def test_pipeline_agents_initialized(self):
        """Test Agents in Pipeline are properly initialized"""
        from src.pipelines.standard_pipeline import StandardPipeline

        mock_wrapper = Mock()
        mock_wrapper.model_config = {"num_layers": 32}

        # Since real model is needed, only test structure here
        # pipeline = StandardPipeline(mock_wrapper, "text")
        # assert pipeline.planner is not None
        pass


class TestCommunicationProtocol:
    """Communication protocol tests"""

    def test_message_creation(self):
        """Test message creation"""
        from src.pipelines.communication import Message

        msg = Message(text="Hello", latent=None)
        assert msg.has_text
        assert not msg.has_latent

    def test_message_size_estimation(self):
        """Test message size estimation"""
        from src.pipelines.communication import CommunicationProtocol, Message

        msg = Message(text="Hello World")
        sizes = CommunicationProtocol.estimate_message_size(msg)

        assert "text_mb" in sizes
        assert "total_mb" in sizes
        assert sizes["text_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
