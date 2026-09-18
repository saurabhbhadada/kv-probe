"""
Fisher / softmax-sensitive geometry.

For a query q:
    z = qK^T / sqrt(d)
    a = softmax(z)
    J = diag(a) - a a^T  (Jacobian of softmax)

The Fisher metric measures how much the attention distribution changes
when perturbing keys.

Perturbation definition:
We consider replacing key k_i with k_j (or vice versa), which changes
the logit z_i to z_j. The first-order change in attention is:
    delta_a ≈ J * e_i * (z_j - z_i)

where e_i is the i-th unit vector.

The Fisher distance is:
    d_F^2(i,j) = mean_q [||J * e_i * (z_j - z_i)||^2]

Mathematical simplification:
    ||J * e_i * (z_j - z_i)||^2
    = (z_j - z_i)^2 * ||J * e_i||^2
    = (z_j - z_i)^2 * a_i * (1 - a_i)

This avoids constructing the full Jacobian matrix.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import GeometryMetric
import math


class FisherMetric(GeometryMetric):
    """
    Fisher / softmax-sensitive distance.

    Measures how much the attention distribution changes when swapping keys.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.temperature = self.config.get('temperature', 1.0)
        self.perturbation = self.config.get('perturbation', 'swap')  # 'swap' or 'merge'
        self.name = f"fisher_{self.perturbation}"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Fisher distance for key pairs.

        Perturbation: swapping k_i with k_j.

        Mathematical form (efficient):
            d_F^2(i,j) = mean_q [(logit_j - logit_i)^2 * a_i * (1 - a_i)]

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]
            queries: [num_queries, d_model]

        Returns:
            distances: [num_pairs]
        """
        if queries is None:
            raise ValueError("FisherMetric requires queries")

        device = keys.device
        seq_len, d_model = keys.shape
        num_queries, _ = queries.shape
        num_pairs = pairs.shape[0]

        # Compute temperature
        temperature = self.temperature if self.temperature > 0 else math.sqrt(d_model)

        # Compute all logits: Q @ K^T / sqrt(d)
        logits_all = queries @ keys.T / temperature  # [num_queries, seq_len]

        # Compute attention weights (softmax)
        attentions = F.softmax(logits_all, dim=1)  # [num_queries, seq_len]

        # Extract indices
        i_indices = pairs[:, 0]  # [num_pairs]
        j_indices = pairs[:, 1]  # [num_pairs]

        # Get logits for pairs
        logits_i = logits_all[:, i_indices]  # [num_queries, num_pairs]
        logits_j = logits_all[:, j_indices]  # [num_queries, num_pairs]

        # Get attention weights
        a_i = attentions[:, i_indices]  # [num_queries, num_pairs]
        a_j = attentions[:, j_indices]  # [num_queries, num_pairs]

        if self.perturbation == 'swap':
            # Swap perturbation: replace k_i with k_j
            # Fisher metric: (logit_j - logit_i)^2 * a_i * (1 - a_i)
            logit_diff = logits_j - logits_i  # [num_queries, num_pairs]
            fisher_local = (logit_diff ** 2) * a_i * (1 - a_i)  # [num_queries, num_pairs]

        elif self.perturbation == 'merge':
            # Merge perturbation: average k_i and k_j
            # This requires computing the merged key and its logit
            # More expensive but potentially more informative

            ki = keys[i_indices]  # [num_pairs, d_model]
            kj = keys[j_indices]  # [num_pairs, d_model]
            k_merged = (ki + kj) / 2  # [num_pairs, d_model]

            # Compute logit for merged key
            logits_merged = queries @ k_merged.T / temperature  # [num_queries, num_pairs]

            # Change in logit when merging: average of two changes
            delta_logit_i = logits_merged - logits_i  # [num_queries, num_pairs]
            delta_logit_j = logits_merged - logits_j  # [num_queries, num_pairs]

            # Fisher for both positions
            fisher_i = (delta_logit_i ** 2) * a_i * (1 - a_i)
            fisher_j = (delta_logit_j ** 2) * a_j * (1 - a_j)

            # Sum the effects
            fisher_local = fisher_i + fisher_j  # [num_queries, num_pairs]

        else:
            raise ValueError(f"Unknown perturbation: {self.perturbation}")

        # Average over queries
        fisher_avg = torch.mean(fisher_local, dim=0)  # [num_pairs]

        # Take square root to get distance metric
        distances = torch.sqrt(fisher_avg + 1e-10)  # add epsilon for stability

        return distances


class FisherSwapMetric(FisherMetric):
    """Fisher metric with swap perturbation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['perturbation'] = 'swap'
        super().__init__(config)


class FisherMergeMetric(FisherMetric):
    """Fisher metric with merge perturbation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['perturbation'] = 'merge'
        super().__init__(config)


class FisherSymmetricMetric(GeometryMetric):
    """
    Symmetric Fisher metric.

    Averages Fisher sensitivity from both positions:
        d_F^2(i,j) = mean_q [(logit_j - logit_i)^2 * (a_i(1-a_i) + a_j(1-a_j)) / 2]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.temperature = self.config.get('temperature', 1.0)
        self.name = "fisher_symmetric"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute symmetric Fisher distance.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]
            queries: [num_queries, d_model]

        Returns:
            distances: [num_pairs]
        """
        if queries is None:
            raise ValueError("FisherSymmetricMetric requires queries")

        device = keys.device
        seq_len, d_model = keys.shape
        num_queries, _ = queries.shape
        num_pairs = pairs.shape[0]

        # Compute temperature
        temperature = self.temperature if self.temperature > 0 else math.sqrt(d_model)

        # Compute all logits
        logits_all = queries @ keys.T / temperature  # [num_queries, seq_len]
        attentions = F.softmax(logits_all, dim=1)  # [num_queries, seq_len]

        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get logits and attentions
        logits_i = logits_all[:, i_indices]  # [num_queries, num_pairs]
        logits_j = logits_all[:, j_indices]  # [num_queries, num_pairs]
        a_i = attentions[:, i_indices]  # [num_queries, num_pairs]
        a_j = attentions[:, j_indices]  # [num_queries, num_pairs]

        # Symmetric Fisher
        logit_diff = logits_j - logits_i  # [num_queries, num_pairs]
        sensitivity_avg = (a_i * (1 - a_i) + a_j * (1 - a_j)) / 2  # [num_queries, num_pairs]

        fisher_local = (logit_diff ** 2) * sensitivity_avg  # [num_queries, num_pairs]

        # Average over queries
        fisher_avg = torch.mean(fisher_local, dim=0)  # [num_pairs]

        # Take square root
        distances = torch.sqrt(fisher_avg + 1e-10)

        return distances
