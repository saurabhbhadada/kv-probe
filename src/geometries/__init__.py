"""
Geometry implementations for KV cache probing.

This module contains different geometric distance metrics for analyzing
transformer KV cache structure.
"""

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
)

__all__ = [
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
]
