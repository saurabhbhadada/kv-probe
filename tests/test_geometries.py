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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
