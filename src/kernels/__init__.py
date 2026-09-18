"""
Kernel operations for neural networks.

NOTE: This module is deprecated. Use src.geometries instead.
Kept for backward compatibility during migration.
"""

# Import from new location
from ..geometries.padic import (
    float_to_2adic,
    _2adic_to_float,
    float_to_2adic_signed,
    _2adic_to_float_signed,
    ultrametric_distance,
    _2adic_norm,
    _2adic_valuation,
    _padic_valuation,
    float_to_2adic_differentiable,
    visualize_2adic,
    PadicConfig,
)

__all__ = [
    'float_to_2adic',
    '_2adic_to_float',
    'float_to_2adic_signed',
    '_2adic_to_float_signed',
    'ultrametric_distance',
    '_2adic_norm',
    '_2adic_valuation',
    '_padic_valuation',
    'float_to_2adic_differentiable',
    'visualize_2adic',
    'PadicConfig',
]
