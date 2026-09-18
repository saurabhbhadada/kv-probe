"""
Ground truth metrics for functional redundancy.

These metrics measure whether two KV entries are actually redundant
in terms of their effect on the attention output.
"""

from .redundancy import (
    compute_deletion_damage,
    compute_merge_damage,
    compute_future_attention_similarity,
)

__all__ = [
    'compute_deletion_damage',
    'compute_merge_damage',
    'compute_future_attention_similarity',
]
