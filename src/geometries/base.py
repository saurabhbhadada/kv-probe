"""
Base class for geometry metrics.

All geometry implementations should inherit from GeometryMetric.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class GeometryMetric(ABC):
    """
    Abstract base class for geometry metrics that measure distance/similarity
    between KV-cache entries.

    Each metric computes how "different" or "redundant" two keys/values are
    according to some geometric notion.

    Attributes:
        higher_is_more_similar: If True, higher metric values indicate MORE similar keys
                               (e.g., cosine similarity, where 1.0 = identical).
                               If False, higher values indicate LESS similar keys
                               (e.g., Euclidean distance, where 0 = identical).
                               This is critical for correct AUROC/AUPRC computation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize geometry metric.

        Args:
            config: Optional configuration dict with metric-specific parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__.replace('Metric', '').lower()
        # Default: assume metric is a distance (higher = less similar)
        # Subclasses should override this if they are similarity metrics
        self.higher_is_more_similar = False

    @abstractmethod
    def compute_pairwise(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        pairs: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute metric for pairs of KV entries.

        Args:
            keys: Key tensor [seq_len, d_model]
            values: Value tensor [seq_len, d_model]
            pairs: Pairs to evaluate [num_pairs, 2] with indices (i, j)
            queries: Optional queries [num_queries, d_model]
            attentions: Optional attention weights [num_queries, seq_len]
            **kwargs: Additional metric-specific arguments

        Returns:
            distances: Tensor of shape [num_pairs] with metric values
        """
        pass

    def precompute(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Precompute any expensive quantities needed for this metric.

        This is called once per (layer, head) and the results are passed
        to compute_pairwise via **kwargs.

        Args:
            keys: All keys for this head [seq_len, d_model]
            values: All values for this head [seq_len, d_model]
            queries: Optional queries [num_queries, d_model]

        Returns:
            Dictionary of precomputed quantities
        """
        return {}

    def __str__(self):
        return self.name
