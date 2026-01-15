"""
LTC模块测试
"""

import pytest
import torch

from src.models.ltc import LatentThoughtCondenser


class TestLTC:
    """LTC模块测试类"""

    def test_ltc_init(self):
        """测试LTC初始化"""
        ltc = LatentThoughtCondenser(
            num_layers=32,
            kv_dim=1024,
            compression_dim=64
        )
        assert ltc.num_layers == 32
        assert ltc.kv_dim == 1024
        assert ltc.compression_dim == 64

    def test_ltc_compression_shape(self):
        """测试压缩后的shape是否正确"""
        ltc = LatentThoughtCondenser(
            num_layers=4,
            kv_dim=256,
            compression_dim=16
        )

        batch_size, seq_len = 2, 128
        kv_cache = {
            l: (torch.randn(batch_size, seq_len, 256),
                torch.randn(batch_size, seq_len, 256))
            for l in range(4)
        }

        compressed, attn_weights = ltc(kv_cache)

        for l in range(4):
            assert compressed[l][0].shape == (batch_size, 16, 256)
            assert compressed[l][1].shape == (batch_size, 16, 256)
            assert attn_weights[l].shape == (batch_size, 16, seq_len)

    def test_compression_ratio(self):
        """测试压缩比例计算"""
        ltc = LatentThoughtCondenser(
            num_layers=4,
            kv_dim=256,
            compression_dim=64
        )

        ratio = ltc.get_compression_ratio(original_seq_len=512)
        assert abs(ratio - 64/512) < 1e-6

    def test_attention_weights_sum_to_one(self):
        """测试注意力权重和为1"""
        ltc = LatentThoughtCondenser(
            num_layers=2,
            kv_dim=128,
            compression_dim=8
        )

        keys = torch.randn(1, 32, 128)
        weights = ltc.compute_attention_weights(keys)

        # 检查每行和为1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
