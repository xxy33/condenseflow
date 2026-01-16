"""
Compression Effect Tests
"""

import pytest
import torch

from src.models.ltc import LatentThoughtCondenser
from src.training.losses import LTCLoss


class TestCompression:
    """Compression effect tests"""

    def test_compression_preserves_information(self):
        """Test compression preserves key information"""
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

        # Compressed cache should have reasonable value range
        for l in range(2):
            k, v = compressed[l]
            assert not torch.isnan(k).any()
            assert not torch.isnan(v).any()
            assert k.abs().mean() > 0  # Should not be all zeros

    def test_different_sequence_lengths(self):
        """Test compression with different sequence lengths"""
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

            # Compressed dimension should be fixed
            for l in range(2):
                assert compressed[l][0].shape == (1, 16, 64)


class TestLTCLoss:
    """LTC loss function tests"""

    def test_loss_computation(self):
        """Test loss computation"""
        loss_fn = LTCLoss(
            lambda_coverage=0.1,
            lambda_orthogonality=0.01,
            num_sampled_queries=32
        )

        # Create mock data
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
        """Test orthogonality loss"""
        loss_fn = LTCLoss()

        # Orthogonal matrix should have lower orthogonality loss
        orthogonal_probes = torch.eye(16, 128)
        ortho_loss = loss_fn.orthogonality_loss(orthogonal_probes)

        # Random matrix should have higher orthogonality loss
        random_probes = torch.randn(16, 128)
        random_loss = loss_fn.orthogonality_loss(random_probes)

        # Orthogonal matrix should have lower loss
        assert ortho_loss < random_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
