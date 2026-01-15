"""
压缩效果测试
"""

import pytest
import torch

from src.models.ltc import LatentThoughtCondenser
from src.training.losses import LTCLoss


class TestCompression:
    """压缩效果测试"""

    def test_compression_preserves_information(self):
        """测试压缩是否保留关键信息"""
        ltc = LatentThoughtCondenser(
            num_layers=2,
            kv_dim=128,
            compression_dim=32
        )

        batch_size, seq_len = 1, 64
        kv_cache = {
            l: (torch.randn(batch_size, seq_len, 128),
                torch.randn(batch_size, seq_len, 128))
            for l in range(2)
        }

        compressed, _ = ltc(kv_cache)

        # 压缩后的cache应该有合理的值范围
        for l in range(2):
            k, v = compressed[l]
            assert not torch.isnan(k).any()
            assert not torch.isnan(v).any()
            assert k.abs().mean() > 0  # 不应该全为0

    def test_different_sequence_lengths(self):
        """测试不同序列长度的压缩"""
        ltc = LatentThoughtCondenser(
            num_layers=2,
            kv_dim=64,
            compression_dim=16
        )

        for seq_len in [32, 64, 128, 256]:
            kv_cache = {
                l: (torch.randn(1, seq_len, 64),
                    torch.randn(1, seq_len, 64))
                for l in range(2)
            }

            compressed, _ = ltc(kv_cache)

            # 压缩后维度应该固定
            for l in range(2):
                assert compressed[l][0].shape == (1, 16, 64)


class TestLTCLoss:
    """LTC损失函数测试"""

    def test_loss_computation(self):
        """测试损失计算"""
        loss_fn = LTCLoss(
            lambda_coverage=0.1,
            lambda_orthogonality=0.01,
            num_sampled_queries=32
        )

        # 创建模拟数据
        original_kv = {
            0: (torch.randn(1, 64, 128), torch.randn(1, 64, 128))
        }
        compressed_kv = {
            0: (torch.randn(1, 16, 128), torch.randn(1, 16, 128))
        }
        attention_weights = {
            0: torch.softmax(torch.randn(1, 16, 64), dim=-1)
        }
        probe_matrix = torch.randn(16, 128)

        losses = loss_fn(original_kv, compressed_kv, attention_weights, probe_matrix)

        assert "total" in losses
        assert "recon" in losses
        assert "coverage" in losses
        assert "orthogonality" in losses
        assert losses["total"].item() > 0

    def test_orthogonality_loss(self):
        """测试正交损失"""
        loss_fn = LTCLoss()

        # 正交矩阵应该有较低的正交损失
        orthogonal_probes = torch.eye(16, 128)
        ortho_loss = loss_fn.orthogonality_loss(orthogonal_probes)

        # 随机矩阵应该有较高的正交损失
        random_probes = torch.randn(16, 128)
        random_loss = loss_fn.orthogonality_loss(random_probes)

        # 正交矩阵的损失应该更低
        assert ortho_loss < random_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
