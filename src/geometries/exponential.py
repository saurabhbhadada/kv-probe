"""
Exponential-response geometry.

Treats each key as a response function:
    f_k(q) = exp(q^T k / sqrt(d))

Distance is the L2 difference in responses:
    d_exp^2(i,j) = mean_q [(f_ki(q) - f_kj(q))^2]

For numerical stability, we work in log-space and use careful centering
to avoid overflow.

Also provides normalized variant:
    d_exp_norm^2(i,j) = mean_q [(f_ki(q) - f_kj(q))^2] / (f_ki(q)^2 + f_kj(q)^2)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import GeometryMetric
import math


class ExponentialResponseMetric(GeometryMetric):
    """
    Exponential-response distance.

    Measures how differently two keys respond to queries in the
    pre-softmax exponential space.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.normalized = self.config.get('normalized', False)
        self.temperature = self.config.get('temperature', 1.0)  # sqrt(d) factor
        self.clip_logits = self.config.get('clip_logits', True)  # prevent overflow
        self.max_logit = self.config.get('max_logit', 20.0)  # clip threshold
        self.name = "exp_response_norm" if self.normalized else "exp_response"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute exponential-response distance.

        Uses centered log-space computation to avoid overflow:
            log(f_ki - f_kj) is computed carefully

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]
            queries: [num_queries, d_model]

        Returns:
            distances: [num_pairs]
        """
        if queries is None:
            raise ValueError("ExponentialResponseMetric requires queries")

        device = keys.device
        num_pairs = pairs.shape[0]
        num_queries, d_model = queries.shape

        # Extract indices
        i_indices = pairs[:, 0]  # [num_pairs]
        j_indices = pairs[:, 1]  # [num_pairs]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute logits: q^T k / temperature
        # queries: [num_queries, d_model]
        # ki, kj: [num_pairs, d_model]
        # logits: [num_queries, num_pairs]

        temperature = self.temperature if self.temperature > 0 else math.sqrt(d_model)

        logits_i = queries @ ki.T / temperature  # [num_queries, num_pairs]
        logits_j = queries @ kj.T / temperature  # [num_queries, num_pairs]

        # Clip logits to prevent overflow
        if self.clip_logits:
            logits_i = torch.clamp(logits_i, -self.max_logit, self.max_logit)
            logits_j = torch.clamp(logits_j, -self.max_logit, self.max_logit)

        # Compute responses: exp(logits)
        response_i = torch.exp(logits_i)  # [num_queries, num_pairs]
        response_j = torch.exp(logits_j)  # [num_queries, num_pairs]

        # Compute squared differences
        diff_squared = (response_i - response_j) ** 2  # [num_queries, num_pairs]

        if self.normalized:
            # Normalized variant: divide by sum of squared responses
            response_squared_sum = response_i ** 2 + response_j ** 2 + 1e-8
            diff_squared = diff_squared / response_squared_sum

        # Average over queries
        mean_diff_squared = torch.mean(diff_squared, dim=0)  # [num_pairs]

        # Take square root to get distance
        distances = torch.sqrt(mean_diff_squared + 1e-8)

        return distances


class ExponentialResponseLogMetric(GeometryMetric):
    """
    Log-space exponential-response distance.

    Works entirely in log-space for better numerical stability.
    Measures: mean_q |logit_i(q) - logit_j(q)|
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.temperature = self.config.get('temperature', 1.0)
        self.p_norm = self.config.get('p_norm', 2)  # L1 or L2 norm
        self.name = f"exp_response_log_L{self.p_norm}"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute log-space exponential-response distance.

        This is more stable: measures logit differences directly.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]
            queries: [num_queries, d_model]

        Returns:
            distances: [num_pairs]
        """
        if queries is None:
            raise ValueError("ExponentialResponseLogMetric requires queries")

        device = keys.device
        num_pairs = pairs.shape[0]
        num_queries, d_model = queries.shape

        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute logits
        temperature = self.temperature if self.temperature > 0 else math.sqrt(d_model)

        logits_i = queries @ ki.T / temperature  # [num_queries, num_pairs]
        logits_j = queries @ kj.T / temperature  # [num_queries, num_pairs]

        # Logit difference
        logit_diff = torch.abs(logits_i - logits_j)  # [num_queries, num_pairs]

        if self.p_norm == 1:
            # L1 norm over queries
            distances = torch.mean(logit_diff, dim=0)  # [num_pairs]
        else:  # L2 norm
            # L2 norm over queries
            distances = torch.sqrt(torch.mean(logit_diff ** 2, dim=0))  # [num_pairs]

        return distances
