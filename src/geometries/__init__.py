"""
Geometry implementations for KV cache probing.

This module contains different geometric distance metrics for analyzing
transformer KV cache structure.
"""

# Legacy p-adic functions (for backward compatibility)
from .padic import (
    PadicConfig,
    float_to_2adic,
    _2adic_to_float,
    float_to_2adic_signed,
    _2adic_to_float_signed,
    _padic_valuation,
    _2adic_valuation,
    ultrametric_distance,
    _2adic_norm,
    visualize_2adic,
    Float2Adic,
    float_to_2adic_differentiable,
    PadicSimilarityMetric,
)

# Backward compatibility alias
UltrametricMetric = PadicSimilarityMetric

# Base class
from .base import GeometryMetric

# Baseline geometries
from .euclidean import (
    EuclideanMetric,
    CosineDistanceMetric,
    CosineSimilarityMetric,
    ManhattanMetric,
)

# Advanced geometries
from .mahalanobis import (
    QueryMahalanobisMetric,
    QueryMahalanobisOracleMetric,
    QueryMahalanobisCausalMetric,
)

from .spherical_radial import (
    SphericalRadialMetric,
    SphericalOnlyMetric,
    RadialOnlyMetric,
)

from .exponential import (
    ExponentialResponseMetric,
    ExponentialResponseLogMetric,
)

from .fisher import (
    FisherMetric,
    FisherSwapMetric,
    FisherMergeMetric,
    FisherSymmetricMetric,
)

__all__ = [
    # Legacy p-adic
    'PadicConfig',
    'float_to_2adic',
    '_2adic_to_float',
    'float_to_2adic_signed',
    '_2adic_to_float_signed',
    '_padic_valuation',
    '_2adic_valuation',
    'ultrametric_distance',
    '_2adic_norm',
    'visualize_2adic',
    'Float2Adic',
    'float_to_2adic_differentiable',
    'PadicSimilarityMetric',
    'UltrametricMetric',  # Backward compatibility alias
    # Base class
    'GeometryMetric',
    # Baseline geometries
    'EuclideanMetric',
    'CosineDistanceMetric',
    'CosineSimilarityMetric',
    'ManhattanMetric',
    # Advanced geometries
    'QueryMahalanobisMetric',
    'QueryMahalanobisOracleMetric',
    'QueryMahalanobisCausalMetric',
    'SphericalRadialMetric',
    'SphericalOnlyMetric',
    'RadialOnlyMetric',
    'ExponentialResponseMetric',
    'ExponentialResponseLogMetric',
    'FisherMetric',
    'FisherSwapMetric',
    'FisherMergeMetric',
    'FisherSymmetricMetric',
]
