"""
Query-induced Mahalanobis distance.

For keys k_i and k_j:
    delta_k = k_i - k_j
    C_Q = mean(q q^T)  # query covariance
    d_Q^2(i,j) = delta_k^T C_Q delta_k

Equivalent empirical form (more efficient):
    d_Q(i,j) = ||Q delta_k||_2 / sqrt(T)

where Q is [T, d_model] matrix of queries.

Supports two modes:
- oracle: C_Q estimated from future queries (ground truth)
- causal: C_Q estimated only from past/recent queries (realistic)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import GeometryMetric


class QueryMahalanobisMetric(GeometryMetric):
    """
    Query-induced Mahalanobis distance.

    Measures distance in the query-weighted subspace, i.e., only directions
    that queries actually probe matter.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mode = self.config.get('mode', 'oracle')  # 'oracle' or 'causal'
        self.causal_window = self.config.get('causal_window', 128)  # for causal mode
        self.reg_epsilon = self.config.get('reg_epsilon', 1e-6)  # regularization
        self.name = f"mahalanobis_{self.mode}"

    def precompute(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Precompute query matrix for efficient pairwise computation.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            queries: [seq_len, d_model] - queries from the same sequence

        Returns:
            dict with 'query_matrix' for efficient distance computation
        """
        if queries is None:
            raise ValueError("QueryMahalanobisMetric requires queries")

        # queries shape: [seq_len, d_model]
        seq_len, d_model = queries.shape
        device = queries.device

        if self.mode == 'oracle':
            # Use all queries (oracle mode)
            Q = queries  # [seq_len, d_model]
        else:
            # Causal mode: for each position, only use past queries
            # This is handled per-pair in compute_pairwise
            Q = queries

        return {'query_matrix': Q}

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        query_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute query-induced Mahalanobis distance for pairs.

        Uses empirical form: d_Q(i,j) = ||Q delta_k||_2 / sqrt(T)

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]
            queries: [seq_len, d_model]
            query_matrix: Precomputed query matrix from precompute()

        Returns:
            distances: [num_pairs]
        """
        if query_matrix is None:
            raise ValueError("query_matrix must be precomputed")

        num_pairs = pairs.shape[0]
        seq_len, d_model = keys.shape
        device = keys.device

        # Extract indices
        i_indices = pairs[:, 0]  # [num_pairs]
        j_indices = pairs[:, 1]  # [num_pairs]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]
        delta_k = ki - kj  # [num_pairs, d_model]

        distances = torch.zeros(num_pairs, device=device)

        if self.mode == 'oracle':
            # Use all queries
            Q = query_matrix  # [seq_len, d_model]
            T = Q.shape[0]

            # Compute ||Q delta_k||_2 / sqrt(T) for each pair
            # Q delta_k^T: [seq_len, d_model] @ [num_pairs, d_model]^T = [seq_len, num_pairs]
            Q_delta = Q @ delta_k.T  # [seq_len, num_pairs]
            norms = torch.norm(Q_delta, p=2, dim=0)  # [num_pairs]
            distances = norms / (T ** 0.5)

        else:  # causal mode
            # For each pair (i, j), only use queries from BEFORE max(i, j)
            # This represents what we know about the query distribution from past context

            for idx in range(num_pairs):
                i_pos = i_indices[idx].item()
                j_pos = j_indices[idx].item()
                max_pos = max(i_pos, j_pos)

                # Get causal queries: positions BEFORE max_pos (past context)
                # Use a window of recent past queries
                causal_start = max(0, max_pos - self.causal_window)
                causal_end = max_pos  # Exclusive: only queries before this position

                if causal_end <= causal_start:
                    # No past queries available
                    distances[idx] = 0.0
                    continue

                Q_causal = query_matrix[causal_start:causal_end]  # [T_causal, d_model]
                T_causal = Q_causal.shape[0]

                if T_causal == 0:
                    distances[idx] = 0.0
                    continue

                # ||Q delta_k||_2 / sqrt(T)
                delta_k_single = delta_k[idx]  # [d_model]
                Q_delta_single = Q_causal @ delta_k_single  # [T_causal]
                norm = torch.norm(Q_delta_single, p=2)
                distances[idx] = norm / (T_causal ** 0.5)

        return distances


class QueryMahalanobisOracleMetric(QueryMahalanobisMetric):
    """Oracle variant: uses all future queries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['mode'] = 'oracle'
        super().__init__(config)


class QueryMahalanobisCausalMetric(QueryMahalanobisMetric):
    """Causal variant: uses only past queries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['mode'] = 'causal'
        super().__init__(config)
