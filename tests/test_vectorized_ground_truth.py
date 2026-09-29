"""
Test vectorized ground truth implementations against reference implementations.

Verifies that:
1. Vectorized deletion damage produces same results as loop version
2. Vectorized merge damage produces same results as cache reconstruction
3. Vectorized future attention similarity produces same results
4. Grouped query-aware execution preserves correct semantics
"""

import torch
import torch.nn.functional as F
import pytest
import numpy as np
import math

from src.ground_truth.redundancy import (
    compute_deletion_damage,
    compute_merge_damage,
    compute_future_attention_similarity,
)

from src.geometries.mahalanobis import (
    QueryMahalanobisOracleMetric,
    QueryMahalanobisCausalMetric,
)
from src.geometries.exponential import ExponentialResponseMetric
from src.geometries.fisher import FisherSymmetricMetric


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


class TestMergeDamageEquivalence:
    """Test that vectorized merge damage matches explicit cache reconstruction."""

    def reference_merge_damage_single_pair(self, K, V, Q, pair, temperature, query_positions=None):
        """
        Reference implementation: explicitly reconstruct merged K/V cache.

        This is the slow but obviously correct implementation.
        """
        import math

        seq_len, d_model = K.shape
        device = K.device

        if temperature is None:
            temperature = math.sqrt(d_model)

        # Canonicalize pair
        i_pos = min(pair[0].item(), pair[1].item())
        j_pos = max(pair[0].item(), pair[1].item())

        # Compute original attention and output
        logits = Q @ K.T / temperature
        if query_positions is not None:
            query_pos_expanded = query_positions.unsqueeze(1)
            key_positions = torch.arange(seq_len, device=device).unsqueeze(0)
            causal_mask = key_positions > query_pos_expanded
            logits = logits.masked_fill(causal_mask, float('-inf'))

        attentions = F.softmax(logits, dim=1)
        output_original = attentions @ V

        # Create merged entries
        k_merged = (K[i_pos] + K[j_pos]) / 2
        v_merged = (V[i_pos] + V[j_pos]) / 2

        # Reconstruct K/V cache with merge
        # Place merged entry at position i, remove position j
        K_merged = torch.cat([
            K[:i_pos],
            k_merged.unsqueeze(0),
            K[i_pos+1:j_pos],
            K[j_pos+1:],
        ], dim=0)

        V_merged = torch.cat([
            V[:i_pos],
            v_merged.unsqueeze(0),
            V[i_pos+1:j_pos],
            V[j_pos+1:],
        ], dim=0)

        # Recompute attention with merged cache
        logits_merged = Q @ K_merged.T / temperature

        if query_positions is not None:
            # Map merged positions to original positions for causal masking
            merged_key_positions = torch.cat([
                torch.arange(i_pos, device=device),
                torch.tensor([i_pos], device=device),
                torch.arange(i_pos + 1, j_pos, device=device),
                torch.arange(j_pos + 1, seq_len, device=device),
            ])
            query_pos_expanded = query_positions.unsqueeze(1)
            merged_key_pos_expanded = merged_key_positions.unsqueeze(0)
            causal_mask_merged = merged_key_pos_expanded > query_pos_expanded
            logits_merged = logits_merged.masked_fill(causal_mask_merged, float('-inf'))

        attentions_merged = F.softmax(logits_merged, dim=1)
        output_merged = attentions_merged @ V_merged

        # Compute damage
        diff = output_original - output_merged
        damage = torch.norm(diff, p=2, dim=1)

        return damage

    def test_single_pair_no_causal(self):
        """Test merge damage vs reference for single pair without causal masking."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        pairs = torch.tensor([[2, 5]])

        # Optimized implementation
        damage_opt = compute_merge_damage(K, V, Q, pairs)

        # Reference implementation
        damage_ref = self.reference_merge_damage_single_pair(K, V, Q, pairs[0], temperature=None)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)

    def test_multiple_pairs_no_causal(self):
        """Test merge damage vs reference for multiple pairs."""
        torch.manual_seed(42)
        seq_len, d_model = 15, 64
        num_queries = 10

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        pairs = torch.tensor([[2, 5], [1, 7], [3, 12]])

        # Optimized implementation
        damage_opt = compute_merge_damage(K, V, Q, pairs)

        # Reference implementation (one pair at a time)
        for idx in range(pairs.shape[0]):
            damage_ref = self.reference_merge_damage_single_pair(K, V, Q, pairs[idx], temperature=None)
            assert torch.allclose(damage_opt[:, idx], damage_ref, atol=1e-5, rtol=1e-4)

    def test_reversed_pair_order(self):
        """Test that (i,j) and (j,i) give same result."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        pairs_fwd = torch.tensor([[3, 7]])
        pairs_rev = torch.tensor([[7, 3]])

        damage_fwd = compute_merge_damage(K, V, Q, pairs_fwd)
        damage_rev = compute_merge_damage(K, V, Q, pairs_rev)

        assert torch.allclose(damage_fwd, damage_rev, atol=1e-6)

    def test_with_causal_mask(self):
        """Test merge damage with causal masking."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 5

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        query_positions = torch.tensor([5, 6, 7, 8, 9])
        pairs = torch.tensor([[2, 4]])

        # Optimized implementation
        damage_opt = compute_merge_damage(K, V, Q, pairs, query_positions=query_positions)

        # Reference implementation
        damage_ref = self.reference_merge_damage_single_pair(K, V, Q, pairs[0],
                                                             temperature=None,
                                                             query_positions=query_positions)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)

    def test_future_query_window(self):
        """Test merge damage with future query window (simulates benchmark usage)."""
        torch.manual_seed(42)
        seq_len, d_model = 20, 64

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)

        # Pair at positions (5, 8), use future queries starting at t+1=9
        pairs = torch.tensor([[5, 8]])
        future_start = 9
        future_horizon = 6
        future_end = min(future_start + future_horizon, seq_len)

        Q_future = torch.randn(future_end - future_start, d_model, dtype=torch.float32)
        query_positions = torch.arange(future_start, future_end)

        # Optimized implementation
        damage_opt = compute_merge_damage(K, V, Q_future, pairs, query_positions=query_positions)

        # Reference implementation
        damage_ref = self.reference_merge_damage_single_pair(K, V, Q_future, pairs[0],
                                                             temperature=None,
                                                             query_positions=query_positions)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)


class TestDeletionDamageEquivalence:
    """Test that vectorized deletion damage matches explicit removal."""

    def reference_deletion_damage_single_position(self, K, V, Q, position, temperature, query_positions=None):
        """
        Reference implementation: explicitly remove position and renormalize.
        """
        import math

        seq_len, d_model = K.shape
        device = K.device

        if temperature is None:
            temperature = math.sqrt(d_model)

        pos = position.item()

        # Compute original attention and output
        logits = Q @ K.T / temperature
        if query_positions is not None:
            query_pos_expanded = query_positions.unsqueeze(1)
            key_positions = torch.arange(seq_len, device=device).unsqueeze(0)
            causal_mask = key_positions > query_pos_expanded
            logits = logits.masked_fill(causal_mask, float('-inf'))

        attentions = F.softmax(logits, dim=1)
        output_original = attentions @ V

        # Remove position pos from K and V
        K_removed = torch.cat([K[:pos], K[pos+1:]], dim=0)
        V_removed = torch.cat([V[:pos], V[pos+1:]], dim=0)

        # Recompute attention
        logits_removed = Q @ K_removed.T / temperature

        if query_positions is not None:
            # Map removed positions
            removed_key_positions = torch.cat([
                torch.arange(pos, device=device),
                torch.arange(pos + 1, seq_len, device=device)
            ])
            query_pos_expanded = query_positions.unsqueeze(1)
            removed_key_pos_expanded = removed_key_positions.unsqueeze(0)
            causal_mask_removed = removed_key_pos_expanded > query_pos_expanded
            logits_removed = logits_removed.masked_fill(causal_mask_removed, float('-inf'))

        attentions_removed = F.softmax(logits_removed, dim=1)
        output_removed = attentions_removed @ V_removed

        # Compute damage
        diff = output_original - output_removed
        damage = torch.norm(diff, p=2, dim=1)

        return damage

    def test_single_position_no_causal(self):
        """Test deletion damage vs reference for single position."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 8

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        positions = torch.tensor([3])

        # Optimized implementation
        damage_opt = compute_deletion_damage(K, V, Q, positions)

        # Reference implementation
        damage_ref = self.reference_deletion_damage_single_position(K, V, Q, positions[0], temperature=None)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)

    def test_multiple_positions_no_causal(self):
        """Test deletion damage vs reference for multiple positions."""
        torch.manual_seed(42)
        seq_len, d_model = 15, 64
        num_queries = 10

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        positions = torch.tensor([2, 5, 9])

        # Optimized implementation
        damage_opt = compute_deletion_damage(K, V, Q, positions)

        # Reference implementation (one position at a time)
        for idx in range(positions.shape[0]):
            damage_ref = self.reference_deletion_damage_single_position(K, V, Q, positions[idx], temperature=None)
            assert torch.allclose(damage_opt[:, idx], damage_ref, atol=1e-5, rtol=1e-4)

    def test_with_causal_mask(self):
        """Test deletion damage with causal masking."""
        torch.manual_seed(42)
        seq_len, d_model = 10, 64
        num_queries = 5

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(num_queries, d_model, dtype=torch.float32)

        query_positions = torch.tensor([5, 6, 7, 8, 9])
        positions = torch.tensor([3])

        # Optimized implementation
        damage_opt = compute_deletion_damage(K, V, Q, positions, query_positions=query_positions)

        # Reference implementation
        damage_ref = self.reference_deletion_damage_single_position(K, V, Q, positions[0],
                                                                    temperature=None,
                                                                    query_positions=query_positions)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)

    def test_future_query_window(self):
        """Test deletion damage with future query window."""
        torch.manual_seed(42)
        seq_len, d_model = 20, 64

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)

        # Test position 7, use future queries starting at 8
        positions = torch.tensor([7])
        future_start = 8
        future_horizon = 6
        future_end = min(future_start + future_horizon, seq_len)

        Q_future = torch.randn(future_end - future_start, d_model, dtype=torch.float32)
        query_positions = torch.arange(future_start, future_end)

        # Optimized implementation
        damage_opt = compute_deletion_damage(K, V, Q_future, positions, query_positions=query_positions)

        # Reference implementation
        damage_ref = self.reference_deletion_damage_single_position(K, V, Q_future, positions[0],
                                                                    temperature=None,
                                                                    query_positions=query_positions)

        assert torch.allclose(damage_opt.squeeze(), damage_ref, atol=1e-5, rtol=1e-4)


class TestGroupedQueryAwareExecution:
    """Test that grouping pairs by t=max(i,j) preserves correct query-aware semantics."""

    def test_oracle_mahalanobis_grouped(self):
        """Test oracle Mahalanobis with grouped execution."""
        torch.manual_seed(42)
        seq_len, d_model = 30, 64
        future_horizon = 10

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(seq_len, d_model, dtype=torch.float32)

        # Pairs with different t values
        pairs = torch.tensor([[5, 8], [10, 12], [5, 7], [10, 15]])
        # t values: 8, 12, 7, 15

        geom = QueryMahalanobisOracleMetric()

        # Reference: compute each pair individually with its own query window
        reference_distances = []
        for pair in pairs:
            i_pos, j_pos = pair[0].item(), pair[1].item()
            t = max(i_pos, j_pos)
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]

            precomputed = geom.precompute(K, V, queries=Q_window)
            dist = geom.compute_pairwise(K, V, pair.unsqueeze(0), queries=Q_window, **precomputed)
            reference_distances.append(dist[0].item())

        # Grouped: simulate benchmark grouping by t
        pairs_by_t = {}
        for idx, pair in enumerate(pairs):
            i_pos, j_pos = pair[0].item(), pair[1].item()
            t = max(i_pos, j_pos)
            if t not in pairs_by_t:
                pairs_by_t[t] = {'indices': [], 'pairs': []}
            pairs_by_t[t]['indices'].append(idx)
            pairs_by_t[t]['pairs'].append(pair.tolist())

        grouped_distances = [None] * len(pairs)
        for t, group_data in pairs_by_t.items():
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]

            group_pairs = torch.tensor(group_data['pairs'], dtype=torch.long)
            precomputed = geom.precompute(K, V, queries=Q_window)
            dists = geom.compute_pairwise(K, V, group_pairs, queries=Q_window, **precomputed)

            for local_idx, global_idx in enumerate(group_data['indices']):
                grouped_distances[global_idx] = dists[local_idx].item()

        # Compare
        assert len(reference_distances) == len(grouped_distances)
        for ref, grp in zip(reference_distances, grouped_distances):
            assert abs(ref - grp) < 1e-5, f"Mismatch: {ref} vs {grp}"

    def test_causal_mahalanobis_grouped(self):
        """Test causal Mahalanobis with grouped execution."""
        torch.manual_seed(42)
        seq_len, d_model = 30, 64
        window_size = 128  # Default causal window

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(seq_len, d_model, dtype=torch.float32)

        # Pairs with different t values
        pairs = torch.tensor([[5, 8], [10, 12], [5, 7], [10, 15]])

        geom = QueryMahalanobisCausalMetric()

        # Reference: compute each pair individually with its own past query window
        reference_distances = []
        for pair in pairs:
            i_pos, j_pos = pair[0].item(), pair[1].item()
            t = max(i_pos, j_pos)
            past_start = max(0, t - window_size)
            past_end = t
            Q_window = Q[past_start:past_end, :]

            precomputed = geom.precompute(K, V, queries=Q_window)
            dist = geom.compute_pairwise(K, V, pair.unsqueeze(0), queries=Q_window, **precomputed)
            reference_distances.append(dist[0].item())

        # Grouped
        pairs_by_t = {}
        for idx, pair in enumerate(pairs):
            i_pos, j_pos = pair[0].item(), pair[1].item()
            t = max(i_pos, j_pos)
            if t not in pairs_by_t:
                pairs_by_t[t] = {'indices': [], 'pairs': []}
            pairs_by_t[t]['indices'].append(idx)
            pairs_by_t[t]['pairs'].append(pair.tolist())

        grouped_distances = [None] * len(pairs)
        for t, group_data in pairs_by_t.items():
            past_start = max(0, t - window_size)
            past_end = t
            Q_window = Q[past_start:past_end, :]

            group_pairs = torch.tensor(group_data['pairs'], dtype=torch.long)
            precomputed = geom.precompute(K, V, queries=Q_window)
            dists = geom.compute_pairwise(K, V, group_pairs, queries=Q_window, **precomputed)

            for local_idx, global_idx in enumerate(group_data['indices']):
                grouped_distances[global_idx] = dists[local_idx].item()

        # Compare
        for ref, grp in zip(reference_distances, grouped_distances):
            assert abs(ref - grp) < 1e-5

    def test_exponential_response_grouped(self):
        """Test exponential response metric with grouped execution."""
        torch.manual_seed(42)
        seq_len, d_model = 30, 64
        future_horizon = 10

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(seq_len, d_model, dtype=torch.float32)

        temp = math.sqrt(d_model)
        pairs = torch.tensor([[5, 8], [10, 12]])

        geom = ExponentialResponseMetric({'temperature': temp, 'normalized': True})

        # Reference
        reference_distances = []
        for pair in pairs:
            t = max(pair[0].item(), pair[1].item())
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]

            precomputed = geom.precompute(K, V, queries=Q_window)
            dist = geom.compute_pairwise(K, V, pair.unsqueeze(0), queries=Q_window, **precomputed)
            reference_distances.append(dist[0].item())

        # Grouped
        pairs_by_t = {}
        for idx, pair in enumerate(pairs):
            t = max(pair[0].item(), pair[1].item())
            if t not in pairs_by_t:
                pairs_by_t[t] = {'indices': [], 'pairs': []}
            pairs_by_t[t]['indices'].append(idx)
            pairs_by_t[t]['pairs'].append(pair.tolist())

        grouped_distances = [None] * len(pairs)
        for t, group_data in pairs_by_t.items():
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]

            group_pairs = torch.tensor(group_data['pairs'], dtype=torch.long)
            precomputed = geom.precompute(K, V, queries=Q_window)
            dists = geom.compute_pairwise(K, V, group_pairs, queries=Q_window, **precomputed)

            for local_idx, global_idx in enumerate(group_data['indices']):
                grouped_distances[global_idx] = dists[local_idx].item()

        # Compare
        for ref, grp in zip(reference_distances, grouped_distances):
            assert abs(ref - grp) < 1e-5

    def test_fisher_symmetric_grouped(self):
        """Test Fisher symmetric metric with grouped execution."""
        torch.manual_seed(42)
        seq_len, d_model = 30, 64
        future_horizon = 10

        K = torch.randn(seq_len, d_model, dtype=torch.float32)
        V = torch.randn(seq_len, d_model, dtype=torch.float32)
        Q = torch.randn(seq_len, d_model, dtype=torch.float32)
        A = torch.softmax(torch.randn(seq_len, seq_len, dtype=torch.float32), dim=1)

        temp = math.sqrt(d_model)
        pairs = torch.tensor([[5, 8], [10, 12]])

        geom = FisherSymmetricMetric({'temperature': temp})

        # Reference
        reference_distances = []
        for pair in pairs:
            t = max(pair[0].item(), pair[1].item())
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]
            A_window = A[future_start:future_end, :]

            precomputed = geom.precompute(K, V, queries=Q_window)
            dist = geom.compute_pairwise(K, V, pair.unsqueeze(0), queries=Q_window,
                                        attentions=A_window, **precomputed)
            reference_distances.append(dist[0].item())

        # Grouped
        pairs_by_t = {}
        for idx, pair in enumerate(pairs):
            t = max(pair[0].item(), pair[1].item())
            if t not in pairs_by_t:
                pairs_by_t[t] = {'indices': [], 'pairs': []}
            pairs_by_t[t]['indices'].append(idx)
            pairs_by_t[t]['pairs'].append(pair.tolist())

        grouped_distances = [None] * len(pairs)
        for t, group_data in pairs_by_t.items():
            future_start = t + 1
            future_end = min(future_start + future_horizon, seq_len)
            Q_window = Q[future_start:future_end, :]
            A_window = A[future_start:future_end, :]

            group_pairs = torch.tensor(group_data['pairs'], dtype=torch.long)
            precomputed = geom.precompute(K, V, queries=Q_window)
            dists = geom.compute_pairwise(K, V, group_pairs, queries=Q_window,
                                         attentions=A_window, **precomputed)

            for local_idx, global_idx in enumerate(group_data['indices']):
                grouped_distances[global_idx] = dists[local_idx].item()

        # Compare
        for ref, grp in zip(reference_distances, grouped_distances):
            assert abs(ref - grp) < 1e-5
