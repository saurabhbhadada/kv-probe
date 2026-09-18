"""
Spherical × radial geometry.

Decomposes keys into direction and magnitude:
    r_i = ||k_i||
    u_i = k_i / ||k_i||  (unit direction)

Distance combines angular and radial components:
    d_SR^2 = acos(clamp(u_i^T u_j, -1, 1))^2
             + lambda * (log(r_i + eps) - log(r_j + eps))^2

This is useful for models with RoPE or other positional encodings that
affect magnitude differently than direction.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import GeometryMetric


class SphericalRadialMetric(GeometryMetric):
    """
    Spherical × radial distance metric.

    Separates angular distance (direction) from radial distance (magnitude).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lambda_radial = self.config.get('lambda_radial', 1.0)  # weight for radial term
        self.eps = self.config.get('eps', 1e-8)  # for numerical stability
        self.name = f"spherical_radial_lambda{self.lambda_radial}"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute spherical × radial distance for pairs.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs]
        """
        device = keys.device
        num_pairs = pairs.shape[0]

        # Extract indices
        i_indices = pairs[:, 0]  # [num_pairs]
        j_indices = pairs[:, 1]  # [num_pairs]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute norms (radii)
        ri = torch.norm(ki, p=2, dim=1) + self.eps  # [num_pairs]
        rj = torch.norm(kj, p=2, dim=1) + self.eps  # [num_pairs]

        # Compute unit directions
        ui = ki / ri.unsqueeze(1)  # [num_pairs, d_model]
        uj = kj / rj.unsqueeze(1)  # [num_pairs, d_model]

        # Angular distance: acos(u_i^T u_j)
        # Clamp dot product to [-1, 1] for numerical stability
        cos_sim = torch.sum(ui * uj, dim=1)  # [num_pairs]
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)  # [num_pairs]

        # Radial distance: log(r_i) - log(r_j)
        log_ri = torch.log(ri)  # [num_pairs]
        log_rj = torch.log(rj)  # [num_pairs]
        radial_dist = log_ri - log_rj  # [num_pairs]

        # Combined distance: sqrt(angular^2 + lambda * radial^2)
        dist_squared = angular_dist ** 2 + self.lambda_radial * (radial_dist ** 2)
        distances = torch.sqrt(dist_squared + self.eps)  # add eps for numerical stability

        return distances


class SphericalOnlyMetric(GeometryMetric):
    """
    Spherical (angular) distance only.

    Measures only direction, ignoring magnitude.
    Equivalent to spherical_radial with lambda=0.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.eps = self.config.get('eps', 1e-8)
        self.name = "spherical_only"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute angular distance only.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs] - angular distances in radians
        """
        device = keys.device

        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Normalize
        ki_norm = F.normalize(ki, p=2, dim=1)  # [num_pairs, d_model]
        kj_norm = F.normalize(kj, p=2, dim=1)  # [num_pairs, d_model]

        # Angular distance
        cos_sim = torch.sum(ki_norm * kj_norm, dim=1)  # [num_pairs]
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angular_dist = torch.acos(cos_sim)  # [num_pairs]

        return angular_dist


class RadialOnlyMetric(GeometryMetric):
    """
    Radial (magnitude) distance only.

    Measures only norm difference, ignoring direction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.eps = self.config.get('eps', 1e-8)
        self.use_log = self.config.get('use_log', True)  # log-space or linear
        self.name = "radial_only_log" if self.use_log else "radial_only_linear"

    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute radial distance only.

        Args:
            keys: [seq_len, d_model]
            values: [seq_len, d_model]
            pairs: [num_pairs, 2]

        Returns:
            distances: [num_pairs] - magnitude differences
        """
        # Extract indices
        i_indices = pairs[:, 0]
        j_indices = pairs[:, 1]

        # Get key pairs
        ki = keys[i_indices]  # [num_pairs, d_model]
        kj = keys[j_indices]  # [num_pairs, d_model]

        # Compute norms
        ri = torch.norm(ki, p=2, dim=1) + self.eps  # [num_pairs]
        rj = torch.norm(kj, p=2, dim=1) + self.eps  # [num_pairs]

        if self.use_log:
            # Log-space distance
            distances = torch.abs(torch.log(ri) - torch.log(rj))
        else:
            # Linear distance
            distances = torch.abs(ri - rj)

        return distances
