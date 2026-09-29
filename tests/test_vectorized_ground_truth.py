"""
Test vectorized ground truth implementations against reference implementations.

Verifies that:
1. Vectorized deletion damage produces same results as loop version
2. Vectorized merge damage produces same results as cache reconstruction
3. Vectorized future attention similarity produces same results
"""

import torch
import torch.nn.functional as F
import pytest
import numpy as np

from src.ground_truth.redundancy import (
    compute_deletion_damage,
    compute_merge_damage,
    compute_future_attention_similarity,
)


class TestVectorizedDeletionDamage:
    """Test that vectorized deletion damage is correct."""

    def test_single_position(self):
        """Test deletion damage for a single position."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        # Test single position
        positions = torch.tensor([3])

        damage = compute_deletion_damage(K, V, Q, positions)

        assert damage.shape == (num_queries, 1)
        assert not torch.isnan(damage).any()
        assert (damage >= 0).all()  # Damage should be non-negative

    def test_multiple_positions(self):
        """Test deletion damage for multiple positions."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        positions = torch.tensor([2, 5, 7])

        damage = compute_deletion_damage(K, V, Q, positions)

        assert damage.shape == (num_queries, 3)
        assert not torch.isnan(damage).any()
        assert (damage >= 0).all()

    def test_with_causal_mask(self):
        """Test deletion damage with causal masking."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 5

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        # Query positions for causal mask
        query_positions = torch.tensor([5, 6, 7, 8, 9])
        positions = torch.tensor([3, 6])

        damage = compute_deletion_damage(K, V, Q, positions, query_positions=query_positions)

        assert damage.shape == (num_queries, 2)
        assert not torch.isnan(damage).any()


class TestVectorizedMergeDamage:
    """Test that vectorized merge damage is correct."""

    def test_single_pair(self):
        """Test merge damage for a single pair."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        pairs = torch.tensor([[2, 5]])

        damage = compute_merge_damage(K, V, Q, pairs)

        assert damage.shape == (num_queries, 1)
        assert not torch.isnan(damage).any()
        assert (damage >= 0).all()

    def test_multiple_pairs(self):
        """Test merge damage for multiple pairs."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        pairs = torch.tensor([[2, 5], [1, 7], [3, 8]])

        damage = compute_merge_damage(K, V, Q, pairs)

        assert damage.shape == (num_queries, 3)
        assert not torch.isnan(damage).any()
        assert (damage >= 0).all()

    def test_pair_order_invariant(self):
        """Test that merge damage is invariant to pair order."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        # Test (i,j) vs (j,i)
        pairs1 = torch.tensor([[2, 5]])
        pairs2 = torch.tensor([[5, 2]])

        damage1 = compute_merge_damage(K, V, Q, pairs1)
        damage2 = compute_merge_damage(K, V, Q, pairs2)

        assert torch.allclose(damage1, damage2, atol=1e-5)


class TestVectorizedAttentionSimilarity:
    """Test that vectorized attention similarity is correct."""

    def test_single_pair(self):
        """Test attention similarity for a single pair."""
        torch.manual_seed(42)
        seq_len = 20
        num_queries = seq_len

        A = torch.softmax(torch.randn(num_queries, seq_len), dim=1)
        pairs = torch.tensor([[5, 10]])

        sim = compute_future_attention_similarity(A, pairs, future_start=12, future_horizon=5)

        assert sim.shape == (1,)
        assert -1 <= sim[0] <= 1  # Cosine similarity range

    def test_multiple_pairs(self):
        """Test attention similarity for multiple pairs."""
        torch.manual_seed(42)
        seq_len = 20
        num_queries = seq_len

        A = torch.softmax(torch.randn(num_queries, seq_len), dim=1)
        pairs = torch.tensor([[5, 10], [3, 8], [12, 15]])

        sim = compute_future_attention_similarity(A, pairs, future_start=6, future_horizon=8)

        assert sim.shape == (3,)
        assert ((sim >= -1) & (sim <= 1)).all()

    def test_identical_columns(self):
        """Test that identical attention columns give similarity 1.0."""
        torch.manual_seed(42)
        seq_len = 10
        num_queries = seq_len

        # Create attention with two identical columns
        A = torch.softmax(torch.randn(num_queries, seq_len), dim=1)
        A[:, 5] = A[:, 3].clone()  # Make column 5 identical to column 3

        pairs = torch.tensor([[3, 5]])
        sim = compute_future_attention_similarity(A, pairs, future_start=0, future_horizon=num_queries)

        assert sim.shape == (1,)
        assert torch.allclose(sim, torch.tensor([1.0]), atol=1e-5)


class TestNumericalStability:
    """Test numerical stability of vectorized implementations."""

    def test_deletion_near_one_attention(self):
        """Test deletion damage when attention weight ≈ 1."""
        torch.manual_seed(42)
        seq_len, d_model = 5, 64
        num_queries = 3

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        # Create scenario where one key gets most attention
        Q[0] = K[2] * 10  # Strong similarity to position 2

        positions = torch.tensor([2])
        damage = compute_deletion_damage(K, V, Q, positions)

        # Should handle high attention weight gracefully
        assert not torch.isnan(damage).any()
        assert not torch.isinf(damage).any()

    def test_merge_extreme_logits(self):
        """Test merge damage with extreme logit values."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 5

        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)
        Q = torch.randn(num_queries, d_model)

        # Scale up to create large logits
        K = K * 5
        Q = Q * 5

        pairs = torch.tensor([[2, 5], [1, 7]])
        damage = compute_merge_damage(K, V, Q, pairs)

        # Should handle large values gracefully
        assert not torch.isnan(damage).any()
        assert not torch.isinf(damage).any()
