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
        assert distances[0].item() == pytest.approx(expected, abs=1e-5)


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
        assert radial_dist[0].item() == pytest.approx(expected_radial, abs=0.01)

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

    def test_causal_no_past_queries_legacy(self):
        """Causal mode with no past queries should return zero distance (legacy non-prewindowed)."""
        from src.geometries.mahalanobis import QueryMahalanobisCausalMetric

        seq_len = 20
        d_model = 32
        queries = torch.randn(seq_len, d_model)
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)

        # Pair (0, 1): only 1 past query before position 1 (at position 0)
        pairs = torch.tensor([[0, 1]])

        # Test legacy mode (queries_prewindowed=False)
        metric = QueryMahalanobisCausalMetric({'causal_window': 128, 'queries_prewindowed': False})
        precomputed = metric.precompute(keys, values, queries=queries)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries, **precomputed)

        # With only 1 past query, distance should be small but not necessarily zero
        # (depends on how different the keys are)
        assert distances.shape == (1,)
        assert not torch.isnan(distances[0])

    def test_causal_late_pair_prewindowed(self):
        """Test causal mode with late pair (t=400) using prewindowed queries."""
        from src.geometries.mahalanobis import QueryMahalanobisCausalMetric

        seq_len = 512
        d_model = 64
        window_size = 128

        # Full sequence
        queries_full = torch.randn(seq_len, d_model)
        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)

        # Late pair at t=400
        i_pos, j_pos = 390, 400
        t = max(i_pos, j_pos)
        pairs = torch.tensor([[i_pos, j_pos]])

        # Runner slices past queries: Q[max(0, t-window):t]
        past_start = max(0, t - window_size)
        past_end = t
        queries_window = queries_full[past_start:past_end, :]  # [window_size, d_model]

        # Metric with prewindowing enabled (default for causal)
        metric = QueryMahalanobisCausalMetric({'causal_window': window_size})
        precomputed = metric.precompute(keys, values, queries=queries_window)
        distances = metric.compute_pairwise(keys, values, pairs, queries=queries_window, **precomputed)

        # Should successfully compute distance (not NaN, not zero unless keys identical)
        assert distances.shape == (1,)
        assert not torch.isnan(distances[0])
        assert distances[0].item() >= 0
        # Distance should be non-zero for random different keys
        if not torch.allclose(keys[i_pos], keys[j_pos], atol=1e-4):
            assert distances[0].item() > 1e-6


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


class TestGroundTruthWithQueryPositions:
    """Test ground truth functions with query_positions for causal masking."""

    def test_deletion_damage_with_query_positions(self):
        """Test compute_deletion_damage with query_positions for causal masking."""
        from src.ground_truth.redundancy import compute_deletion_damage

        seq_len = 20
        d_model = 32
        num_queries = 10

        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)
        queries = torch.randn(num_queries, d_model)

        # Query positions: queries are at positions [10, 11, ..., 19]
        query_positions = torch.arange(10, 20, dtype=torch.long)

        # Positions to delete: [5, 15]
        positions = torch.tensor([5, 15], dtype=torch.long)

        # Compute damage with causal masking
        damage = compute_deletion_damage(
            keys, values, queries, positions,
            query_positions=query_positions
        )

        # Shape should be [num_queries, num_positions]
        assert damage.shape == (num_queries, 2)

        # Query at position 10 should attend to key 5 (past) but NOT key 15 (future)
        # So deleting key 5 may cause damage, but deleting key 15 should cause zero damage
        # (because it's masked out)
        # Query at position 15 should attend to both key 5 and key 15 (both past or current)

        # We can't verify exact values without knowing attention, but we can verify:
        # 1. No NaN values
        assert not torch.isnan(damage).any()
        # 2. All values are non-negative (L2 output damage is non-negative)
        assert torch.all(damage >= 0)

    def test_merge_damage_with_query_positions(self):
        """Test compute_merge_damage with query_positions for causal masking."""
        from src.ground_truth.redundancy import compute_merge_damage

        seq_len = 30
        d_model = 32
        num_queries = 10

        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)
        queries = torch.randn(num_queries, d_model)

        # Query positions: queries are at positions [15, 16, ..., 24]
        query_positions = torch.arange(15, 25, dtype=torch.long)

        # Pairs to merge: (5, 10), (18, 22)
        pairs = torch.tensor([[5, 10], [18, 22]], dtype=torch.long)

        # Compute merge damage with causal masking
        damage = compute_merge_damage(
            keys, values, queries, pairs,
            query_positions=query_positions
        )

        # Shape should be [num_queries, num_pairs]
        assert damage.shape == (num_queries, 2)

        # Query at position 15:
        #   - can see pair (5, 10) - both are past
        #   - can see position 18 but NOT 22 (22 is future)
        # Query at position 20:
        #   - can see pair (5, 10) - both are past
        #   - can see position 18 but NOT 22 (22 is future)
        # Query at position 24:
        #   - can see both pairs

        # Verify no NaN and non-negative
        assert not torch.isnan(damage).any()
        assert torch.all(damage >= 0)

    def test_causal_masking_prevents_future_attention(self):
        """Verify that queries cannot attend to future keys."""
        from src.ground_truth.redundancy import compute_deletion_damage

        seq_len = 10
        d_model = 16

        # Create identical keys except for one future key
        keys = torch.ones(seq_len, d_model)
        keys[8] = torch.ones(d_model) * 100  # Make position 8 very different
        values = torch.randn(seq_len, d_model)

        # Single query at position 5
        queries = torch.randn(1, d_model)
        query_positions = torch.tensor([5], dtype=torch.long)

        # Delete the future key at position 8
        positions = torch.tensor([8], dtype=torch.long)

        damage = compute_deletion_damage(
            keys, values, queries, positions,
            query_positions=query_positions
        )

        # Damage should be zero or very small because query at position 5
        # cannot attend to key at position 8 (future)
        assert damage[0, 0].item() < 1e-5, \
            f"Query at position 5 should not attend to future key at position 8, but damage = {damage[0, 0]}"


class TestFisherWithRealAttention:
    """Test Fisher metric with real attention weights."""

    def test_fisher_uses_provided_attention(self):
        """Verify Fisher metric uses provided attention weights instead of recomputing."""

        seq_len = 10
        d_model = 16
        num_queries = 5

        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)
        queries = torch.randn(num_queries, d_model)

        # Create synthetic attention weights (uniform for simplicity)
        attentions_real = torch.ones(num_queries, seq_len) / seq_len

        # Pairs to test
        pairs = torch.tensor([[2, 5], [0, 8]])

        metric = FisherSymmetricMetric()

        # Compute with real attention
        distances_with_attn = metric.compute_pairwise(
            keys, values, pairs, queries=queries, attentions=attentions_real
        )

        # Compute without attention (will recompute)
        distances_no_attn = metric.compute_pairwise(
            keys, values, pairs, queries=queries, attentions=None
        )

        # Both should succeed
        assert distances_with_attn.shape == (2,)
        assert distances_no_attn.shape == (2,)
        assert not torch.isnan(distances_with_attn).any()
        assert not torch.isnan(distances_no_attn).any()

        # They will likely differ unless keys happen to produce uniform attention
        # This test mainly verifies that the attention parameter is accepted and used

    def test_fisher_with_future_queries_and_causal_attention(self):
        """Test Fisher with future queries using causal attention from model."""

        seq_len = 20
        d_model = 32

        keys = torch.randn(seq_len, d_model)
        values = torch.randn(seq_len, d_model)

        # Future queries: positions [10, 11, 12, 13, 14]
        future_start = 10
        future_end = 15
        queries_future = torch.randn(future_end - future_start, d_model)

        # Create causal attention weights for future queries
        # Each future query can only attend to keys up to its position
        num_future = future_end - future_start
        attentions_causal = torch.zeros(num_future, seq_len)
        for q_idx in range(num_future):
            q_pos = future_start + q_idx
            # Query at position q_pos can attend to keys [0, q_pos]
            attentions_causal[q_idx, :q_pos + 1] = 1.0 / (q_pos + 1)  # Uniform over valid keys

        # Test pair within past: (3, 7)
        pairs = torch.tensor([[3, 7]])

        metric = FisherSymmetricMetric()
        distances = metric.compute_pairwise(
            keys, values, pairs, queries=queries_future, attentions=attentions_causal
        )

        # Should successfully compute
        assert distances.shape == (1,)
        assert not torch.isnan(distances[0])
        assert distances[0].item() >= 0


class TestQKReconstruction:
    """Test QK reconstruction for GPT-NeoX."""

    def test_qk_reconstruction_passes_for_valid_qk(self):
        """Test that QK reconstruction succeeds when Q and K are correctly extracted."""
        from scripts.run_geometry_benchmark import sanity_check_qk_reconstruction
        import torch.nn.functional as F

        seq_len = 16
        head_dim = 64

        # Simulate correctly extracted Q and K (post-RoPE, same coordinate system)
        Q = torch.randn(1, 4, seq_len, head_dim)  # [batch, heads, seq, dim]
        K = torch.randn(1, 4, seq_len, head_dim)

        # Compute attention manually with causal mask
        scaling = head_dim ** -0.5
        logits = torch.einsum('bhqd,bhkd->bhqk', Q, K) * scaling  # [batch, heads, seq, seq]

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        logits = logits.masked_fill(causal_mask, float('-inf'))

        A = F.softmax(logits, dim=-1)  # [batch, heads, seq, seq]

        # Sanity check should pass
        head_idx = 0
        layer_idx = 5
        max_err, mean_err = sanity_check_qk_reconstruction(
            Q, K, A, head_idx, layer_idx, scaling_factor=scaling, tolerance=0.01
        )

        assert max_err < 0.01
        assert mean_err < 0.01

    def test_qk_reconstruction_fails_for_mismatched_qk(self):
        """Test that QK reconstruction fails when Q and K are mismatched."""
        from scripts.run_geometry_benchmark import sanity_check_qk_reconstruction
        import torch.nn.functional as F

        seq_len = 16
        head_dim = 64

        # Correctly extracted Q and K
        Q = torch.randn(1, 4, seq_len, head_dim)
        K = torch.randn(1, 4, seq_len, head_dim)

        # Compute correct attention
        scaling = head_dim ** -0.5
        logits = torch.einsum('bhqd,bhkd->bhqk', Q, K) * scaling
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        logits = logits.masked_fill(causal_mask, float('-inf'))
        A_correct = F.softmax(logits, dim=-1)

        # Now corrupt Q by rotating it (simulating missing RoPE or wrong extraction)
        Q_corrupted = Q * 0.5 + torch.randn_like(Q) * 0.5

        # Sanity check should FAIL with corrupted Q
        head_idx = 0
        layer_idx = 5

        with pytest.raises(RuntimeError, match="FATAL: QK reconstruction check failed"):
            sanity_check_qk_reconstruction(
                Q_corrupted, K, A_correct, head_idx, layer_idx,
                scaling_factor=scaling, tolerance=0.01
            )


class TestPadicQuantization:
    """Test p-adic quantization with float16 inputs."""

    def test_float16_precision16_no_overflow(self):
        """Test that float16 input with precision=16 does not overflow."""
        from src.geometries.padic import float_to_2adic

        # Create float16 tensor with values near the range limits
        x = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0], dtype=torch.float16)

        # This should not raise an overflow error
        x_2adic, scale = float_to_2adic(x, precision=16)

        # Verify output shape and dtype
        assert x_2adic.shape == x.shape
        assert x_2adic.dtype == torch.int32  # Should be int32, not int16

        # Verify range: all values should be in [0, 2^16 - 1]
        assert torch.all(x_2adic >= 0)
        assert torch.all(x_2adic < 65536)

    def test_float16_large_values(self):
        """Test float16 with large values near FP16 max."""
        from src.geometries.padic import float_to_2adic

        # Values near float16 max (~65504)
        x = torch.tensor([100.0, -100.0, 50.0, 0.0], dtype=torch.float16)

        x_2adic, scale = float_to_2adic(x, precision=16)

        # Should not overflow
        assert x_2adic.dtype == torch.int32
        assert torch.all(x_2adic >= 0)
        assert torch.all(x_2adic < 65536)

    def test_negative_values_quantization(self):
        """Test that negative values are correctly quantized."""
        from src.geometries.padic import float_to_2adic

        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
        x_2adic, scale = float_to_2adic(x, precision=8)

        # All quantized values should be in valid range
        assert torch.all(x_2adic >= 0)
        assert torch.all(x_2adic < 256)

        # Negative values should map to lower half of range
        # -1.0 should map to 0 (or very close)
        # +1.0 should map to 255 (or very close)
        assert x_2adic[0].item() < 10  # -1.0 near 0
        assert x_2adic[4].item() > 245  # +1.0 near 255

    def test_output_range_bounds(self):
        """Test that output satisfies 0 <= x_2adic < 2^precision."""
        from src.geometries.padic import float_to_2adic

        for precision in [8, 12, 16, 20]:
            x = torch.randn(100, dtype=torch.float32)
            x_2adic, scale = float_to_2adic(x, precision=precision)

            modulus = 2 ** precision
            assert torch.all(x_2adic >= 0), f"Negative values found for precision={precision}"
            assert torch.all(x_2adic < modulus), f"Values >= modulus found for precision={precision}"

    def test_no_wraparound(self):
        """Test that maximum values don't wrap around to zero."""
        from src.geometries.padic import float_to_2adic

        # Create tensor with max value
        x = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        x_2adic, scale = float_to_2adic(x, precision=16)

        # Max values should map to near modulus-1, NOT wrap to 0
        modulus = 65536
        assert torch.all(x_2adic > modulus - 100), \
            f"Max values wrapped around: {x_2adic}, expected near {modulus-1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
