"""
Euclidean (L2) distance metric.

Standard L2 distance: d(i,j) = ||k_i - k_j||_2

This is the simplest baseline geometry.
"""

import torch
from typing import Dict, Any, Optional
from .base import GeometryMetric


class EuclideanMetric(GeometryMetric):
    """
    Euclidean (L2) distance between keys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "euclidean"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Euclidean distance for pairs.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs]
        """
        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute L2 distance
        distances = torch.norm(ki - kj, p=2, dim=1)  # [num_pairs]

        return distances


class CosineDistanceMetric(GeometryMetric):
    """
    Cosine distance (1 - cosine similarity) between keys.

    Distance is in [0, 2], where:
    - 0 = identical direction
    - 1 = orthogonal
    - 2 = opposite direction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "cosine_distance"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute cosine distance for pairs.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs]
        """
        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Normalize
        ki_norm = torch.nn.functional.normalize(ki, p=2, dim=1)
        kj_norm = torch.nn.functional.normalize(kj, p=2, dim=1)

        # Cosine similarity
        cos_sim = torch.sum(ki_norm * kj_norm, dim=1)  # [num_pairs]

        # Convert to distance: 1 - similarity
        distances = 1.0 - cos_sim

        return distances


class CosineSimilarityMetric(GeometryMetric):
    """
    Cosine similarity (not distance) between keys.

    Returns similarity in [-1, 1], where:
    - 1 = identical direction
    - 0 = orthogonal
    - -1 = opposite direction

    Note: This returns similarity, not distance. Higher values = more similar.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "cosine_similarity"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute cosine similarity for pairs.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            similarities: [num_pairs] - higher is more similar
        """
        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Normalize
        ki_norm = torch.nn.functional.normalize(ki, p=2, dim=1)
        kj_norm = torch.nn.functional.normalize(kj, p=2, dim=1)

        # Cosine similarity
        similarities = torch.sum(ki_norm * kj_norm, dim=1)  # [num_pairs]

        return similarities


class ManhattanMetric(GeometryMetric):
    """
    Manhattan (L1) distance between keys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "manhattan"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Manhattan distance for pairs.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs]
        """
        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute L1 distance
        distances = torch.norm(ki - kj, p=1, dim=1)  # [num_pairs]

        return distances
