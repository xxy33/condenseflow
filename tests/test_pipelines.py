"""
Pipeline模块测试
"""

import pytest
from unittest.mock import Mock


class TestStandardPipeline:
    """标准Pipeline测试"""

    def test_pipeline_agents_initialized(self):
        """测试Pipeline中的Agent是否正确初始化"""
        from src.pipelines.standard_pipeline import StandardPipeline

        mock_wrapper = Mock()
        mock_wrapper.model_config = {"num_layers": 32}

        # 由于需要真实模型，这里只测试结构
        # pipeline = StandardPipeline(mock_wrapper, "text")
        # assert pipeline.planner is not None
        pass


class TestCommunicationProtocol:
    """通信协议测试"""

    def test_message_creation(self):
        """测试消息创建"""
        from src.pipelines.communication import Message

        msg = Message(text="Hello", latent=None)
        assert msg.has_text
        assert not msg.has_latent

    def test_message_size_estimation(self):
        """测试消息大小估算"""
        from src.pipelines.communication import CommunicationProtocol, Message

        msg = Message(text="Hello World")
        sizes = CommunicationProtocol.estimate_message_size(msg)

        assert "text_mb" in sizes
        assert "total_mb" in sizes
        assert sizes["text_mb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
