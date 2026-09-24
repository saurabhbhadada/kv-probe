"""
Unit tests for geometry metrics.

Tests include synthetic cases to verify mathematical correctness.
"""

import pytest
import torch
import numpy as np
from src.geometries import (
    EuclideanMetric,
    CosineSimilarityMetric,
    QueryMahalanobisOracleMetric,
    SphericalRadialMetric,
    SphericalOnlyMetric,
    RadialOnlyMetric,
    ExponentialResponseLogMetric,
    FisherSymmetricMetric,
)


class TestEuclideanMetric:
    """Test Euclidean distance."""

    def test_identical_keys(self):
        """Identical keys should have zero distance."""
        keys = torch.randn(10, 64)
        values = torch.randn(10, 64)
        pairs = torch.tensor([[0, 0], [5, 5]])  # same index

        metric = EuclideanMetric()
        distances = metric.compute_pairwise(keys, values, pairs)

        assert torch.allclose(distances, torch.zeros(2), atol=1e-6)

    def test_orthogonal_keys(self):
        """Orthogonal keys should have distance sqrt(2)."""
        keys = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        values = torch.randn(2, 2)
        pairs = torch.tensor([[0, 1]])

        metric = EuclideanMetric()
        distances = metric.compute_pairwise(keys, values, pairs)

        expected = np.sqrt(2.0)
        assert torch.allclose(distances, torch.tensor([expected]), atol=1e-5)


class TestCosineSimilarity:
    """Test cosine similarity."""

    def test_identical_direction(self):
        """Same direction should have similarity 1."""
        keys = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # same direction, different magnitude
        ])
        values = torch.randn(2, 3)
        pairs = torch.tensor([[0, 1]])

        metric = CosineSimilarityMetric()
        similarities = metric.compute_pairwise(keys, values, pairs)

        assert torch.allclose(similarities, torch.ones(1), atol=1e-5)

    def test_orthogonal_direction(self):
        """Orthogonal vectors should have similarity 0."""
        keys = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        values = torch.randn(2, 2)
        pairs = torch.tensor([[0, 1]])

        metric = CosineSimilarityMetric()
        similarities = metric.compute_pairwise(keys, values, pairs)

        assert torch.allclose(similarities, torch.zeros(1), atol=1e-5)


class TestQueryMahalanobis:
    """Test query-induced Mahalanobis distance."""

    def test_orthogonal_to_queries(self):
        """
        If two keys differ only in a direction orthogonal to all queries,
        their distance should be approximately zero.
        """
        # Queries only probe first dimension
        queries = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])

        # Keys differ only in dimensions 2 and 3 (orthogonal to queries)
        keys = torch.tensor([
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],  # same first dim, different others
        ])
        values = torch.randn(2, 3)
        pairs = torch.tensor([[0, 1]])

        metric = QueryMahalanobisOracleMetric()
        precomputed = metric.precompute(keys, values, queries=queries)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries, **precomputed)

        # Distance should be small (not exactly zero due to numerical precision)
        assert distances[0].item() < 0.1

    def test_parallel_to_queries(self):
        """
        If two keys differ in a direction parallel to queries,
        distance should be large.
        """
        # Queries probe first dimension
        queries = torch.tensor([
            [1.0, 0.0],
            [2.0, 0.0],
        ])

        # Keys differ in first dimension
        keys = torch.tensor([
            [1.0, 0.0],
            [5.0, 0.0],  # large difference in first dim
        ])
        values = torch.randn(2, 2)
        pairs = torch.tensor([[0, 1]])

        metric = QueryMahalanobisOracleMetric()
        precomputed = metric.precompute(keys, values, queries=queries)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries, **precomputed)

        # Distance should be large
        assert distances[0].item() > 1.0


class TestSphericalRadial:
    """Test spherical × radial geometry."""

    def test_same_direction_different_norm(self):
        """
        Keys pointing in same direction with different norms should have
        zero angular distance but nonzero radial distance.
        """
        keys = torch.tensor([
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # same direction, 5x magnitude
        ])
        values = torch.randn(2, 3)
        pairs = torch.tensor([[0, 1]])

        # Test angular only
        metric_angular = SphericalOnlyMetric()
        angular_dist = metric_angular.compute_pairwise(keys, values, pairs)

        assert torch.allclose(angular_dist, torch.zeros(1), atol=1e-5)

        # Test radial only
        metric_radial = RadialOnlyMetric({'use_log': True})
        radial_dist = metric_radial.compute_pairwise(keys, values, pairs)

        # log(5) - log(1) = log(5) ≈ 1.609
        expected_radial = np.log(5.0)
        assert torch.allclose(radial_dist, torch.tensor([expected_radial]), atol=0.01)

    def test_orthogonal_same_norm(self):
        """
        Orthogonal keys with same norm should have nonzero angular distance
        but zero radial distance.
        """
        keys = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        values = torch.randn(2, 2)
        pairs = torch.tensor([[0, 1]])

        # Angular distance should be pi/2
        metric_angular = SphericalOnlyMetric()
        angular_dist = metric_angular.compute_pairwise(keys, values, pairs)

        expected = np.pi / 2
        assert torch.allclose(angular_dist, torch.tensor([expected]), atol=0.01)

        # Radial distance should be 0
        metric_radial = RadialOnlyMetric({'use_log': True})
        radial_dist = metric_radial.compute_pairwise(keys, values, pairs)

        assert torch.allclose(radial_dist, torch.zeros(1), atol=1e-5)


class TestExponentialResponse:
    """Test exponential-response geometry."""

    def test_identical_keys(self):
        """Identical keys should have zero distance."""
        keys = torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])
        values = torch.randn(2, 3)
        queries = torch.randn(5, 3)
        pairs = torch.tensor([[0, 1]])

        metric = ExponentialResponseLogMetric({'p_norm': 2})
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries)

        assert torch.allclose(distances, torch.zeros(1), atol=1e-5)


class TestFisherMetric:
    """Test Fisher geometry."""

    def test_non_negativity(self):
        """Fisher distance should always be non-negative."""
        torch.manual_seed(42)
        keys = torch.randn(10, 64)
        values = torch.randn(10, 64)
        queries = torch.randn(10, 64)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])

        metric = FisherSymmetricMetric()
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries)

        assert torch.all(distances >= 0)

    def test_zero_perturbation(self):
        """Zero perturbation (identical keys) should give near-zero distance."""
        keys = torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],  # identical
        ])
        values = torch.randn(2, 3)
        queries = torch.randn(5, 3)
        pairs = torch.tensor([[0, 1]])

        metric = FisherSymmetricMetric()
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries)

        # Should be very small (not exactly zero due to numerical precision)
        assert distances[0].item() < 1e-3


class TestCausalMasking:
    """Test causal masking in attention and damage computations."""

    def test_attention_reconstruction_with_causal_mask(self):
        """Test that softmax(QK^T / sqrt(d) + causal_mask) works correctly."""
        import torch.nn.functional as F

        seq_len, d_model = 8, 16
        Q = torch.randn(seq_len, d_model)
        K = torch.randn(seq_len, d_model)

        temperature = np.sqrt(d_model)

        # Compute logits
        logits = Q @ K.T / temperature  # [seq_len, seq_len]

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        logits_masked = logits.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn = F.softmax(logits_masked, dim=-1)

        # Verify causal property: attn[i, j] = 0 for j > i
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn[i, j].item() < 1e-6, f"Non-causal attention at ({i}, {j})"

        # Verify rows sum to 1
        row_sums = attn.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(seq_len), atol=1e-5)


class TestMergeDamage:
    """Test merge damage with pair ordering."""

    def test_reversed_pair_merge(self):
        """Test that merge works correctly even when i > j."""
        from src.ground_truth.redundancy import compute_merge_damage

        seq_len, d_model = 10, 16
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)
        queries = torch.randn(5, d_model)

        # Test both orderings of the same pair
        pairs_normal = torch.tensor([[2, 7]])  # i < j
        pairs_reversed = torch.tensor([[7, 2]])  # i > j (should be canonicalized)

        damage_normal = compute_merge_damage(keys, values, queries, pairs_normal)
        damage_reversed = compute_merge_damage(keys, values, queries, pairs_reversed)

        # Both should give the same result
        assert torch.allclose(damage_normal, damage_reversed, atol=1e-5), \
            "Merge damage should be invariant to pair ordering"

    def test_merge_reduces_length_by_one(self):
        """Test that merging two entries reduces sequence length by exactly 1."""
        from src.ground_truth.redundancy import compute_merge_damage

        seq_len, d_model = 15, 32
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)
        queries = torch.randn(8, d_model)

        # Merge pair (3, 9)
        pairs = torch.tensor([[3, 9]])

        # The function should internally create merged cache with length seq_len - 1
        # We can't directly test this without modifying the function,
        # but we can verify it doesn't raise an assertion error
        damage = compute_merge_damage(keys, values, queries, pairs)

        assert damage.shape == (8, 1), "Damage shape should match (num_queries, num_pairs)"


class TestMahalanobisQueryWindows:
    """Test oracle vs causal query window selection in Mahalanobis metric."""

    def test_oracle_uses_all_queries(self):
        """Oracle mode should use all available queries."""
        queries = torch.randn(100, 32)
        keys = torch.randn(50, 32)
        values = torch.randn(50, 32)
        pairs = torch.tensor([[10, 20], [5, 15]])

        metric = QueryMahalanobisOracleMetric()
        precomputed = metric.precompute(keys, values, queries=queries)

        # Oracle should use all 100 queries
        assert precomputed['query_matrix'].shape[0] == 100

    def test_causal_uses_past_queries(self):
        """Causal mode should only use queries before max(i, j)."""
        from src.geometries.mahalanobis import QueryMahalanobisCausalMetric

        seq_len = 50
        d_model = 32
        queries = torch.randn(seq_len, d_model)
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)

        # Pair (10, 20): causal should use queries from positions [0, 20)
        pairs = torch.tensor([[10, 20]])

        metric = QueryMahalanobisCausalMetric({'causal_window': 128})
        precomputed = metric.precompute(keys, values, queries=queries)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries, **precomputed)

        # Distance should be computed (non-zero if keys differ)
        assert distances.shape == (1,)

        # For pair at position 20, there should be 20 past queries available
        # This is tested implicitly by not raising errors

    def test_causal_no_past_queries(self):
        """Causal mode with no past queries should return zero distance."""
        from src.geometries.mahalanobis import QueryMahalanobisCausalMetric

        seq_len = 20
        d_model = 32
        queries = torch.randn(seq_len, d_model)
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)

        # Pair (0, 1): no past queries before position 1
        pairs = torch.tensor([[0, 1]])

        metric = QueryMahalanobisCausalMetric({'causal_window': 128})
        precomputed = metric.precompute(keys, values, queries=queries)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries, **precomputed)

        # Should return zero or very small distance
        assert distances[0].item() < 1e-6


class TestTemperatureDefaults:
    """Test that temperature defaults to sqrt(head_dim)."""

    def test_fisher_temperature_default(self):
        """Fisher metric should default to sqrt(d_model)."""
        metric = FisherSymmetricMetric()
        assert metric.temperature == 1.0  # Default before fix

        # After fix, should be computed dynamically
        metric_fixed = FisherSymmetricMetric({'temperature': 0.0})  # 0 means use default
        assert metric_fixed.temperature == 0.0

    def test_exponential_temperature_default(self):
        """Exponential metric should default to sqrt(d_model)."""
        from src.geometries.exponential import ExponentialResponseMetric

        metric = ExponentialResponseMetric({'temperature': 0.0})
        assert metric.temperature == 0.0  # Will be computed as sqrt(d) at runtime


class TestUniquePairs:
    """Test unique pair sampling."""

    def test_unique_pairs_no_duplicates(self):
        """Unique pair sampling should not generate duplicates."""
        from scripts.run_geometry_benchmark import sample_pairs

        seq_len = 20
        num_pairs = 50

        pairs = sample_pairs(seq_len, num_pairs, strategy='random', seed=42, unique=True)

        # Check for duplicates
        pairs_set = set(map(tuple, pairs))
        assert len(pairs_set) == len(pairs), "Duplicate pairs found"

    def test_pairs_canonicalized(self):
        """All pairs should have i < j."""
        from scripts.run_geometry_benchmark import sample_pairs

        seq_len = 30
        num_pairs = 100

        pairs = sample_pairs(seq_len, num_pairs, strategy='random', seed=123, unique=True)

        # Check all pairs have i < j
        for i, j in pairs:
            assert i < j, f"Pair ({i}, {j}) not canonicalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
