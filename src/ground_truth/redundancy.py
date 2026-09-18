"""
Ground truth metrics for functional redundancy.

These measure the actual impact of removing or merging KV entries
on the attention output.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def compute_deletion_damage(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    positions: torch.Tensor,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute damage from deleting individual tokens.

    For position j, compute:
        o = sum_i a_i v_i
        o_minus_j = (o - a_j v_j) / (1 - a_j)
        damage_j = ||o - o_minus_j||

    Closed-form expression (more stable):
        damage_j = a_j / (1 - a_j) * ||v_j - o||

    Args:
        keys: [seq_len, d_model]
        values: [seq_len, d_model]
        queries: [num_queries, d_model]
        positions: [num_positions] - positions to compute damage for
        temperature: Temperature for attention (default: sqrt(d_model))

    Returns:
        damage: [num_queries, num_positions] - deletion damage per query
    """
    seq_len, d_model = keys.shape
    num_queries = queries.shape[0]
    device = keys.device

    if temperature is None:
        temperature = math.sqrt(d_model)

    # Compute attention: [num_queries, seq_len]
    logits = queries @ keys.T / temperature  # [num_queries, seq_len]
    attentions = F.softmax(logits, dim=1)  # [num_queries, seq_len]

    # Compute original output: o = sum_i a_i v_i
    # [num_queries, seq_len] @ [seq_len, d_model] = [num_queries, d_model]
    output_original = attentions @ values  # [num_queries, d_model]

    # Compute damage for specified positions
    num_positions = positions.shape[0]
    damage = torch.zeros(num_queries, num_positions, device=device)

    for idx, pos in enumerate(positions):
        pos_idx = pos.item()

        # Get attention weight and value for this position
        a_j = attentions[:, pos_idx]  # [num_queries]
        v_j = values[pos_idx]  # [d_model]

        # Closed-form damage: a_j / (1 - a_j) * ||v_j - o||
        # Broadcast v_j to match output_original shape
        v_j_expanded = v_j.unsqueeze(0)  # [1, d_model]
        diff = v_j_expanded - output_original  # [num_queries, d_model]
        diff_norm = torch.norm(diff, p=2, dim=1)  # [num_queries]

        # Handle case where a_j ≈ 1 (numerical protection)
        # When a_j → 1, damage → infinity, but we clip it
        a_j_safe = torch.clamp(a_j, 0.0, 0.9999)  # prevent division by zero
        damage_factor = a_j_safe / (1 - a_j_safe)  # [num_queries]

        damage[:, idx] = damage_factor * diff_norm

    return damage


def compute_merge_damage(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    pairs: torch.Tensor,
    temperature: Optional[float] = None,
    merge_strategy: str = 'average',
) -> torch.Tensor:
    """
    Compute damage from merging pairs of tokens.

    Simulates replacing positions i and j with a single merged token.

    Merge strategies:
    - 'average': k_merged = (k_i + k_j)/2, v_merged = (v_i + v_j)/2
    - 'weighted': weight by original attention weights (more complex)

    Args:
        keys: [seq_len, d_model]
        values: [seq_len, d_model]
        queries: [num_queries, d_model]
        pairs: [num_pairs, 2] - pairs to merge
        temperature: Temperature for attention
        merge_strategy: How to merge ('average' or 'weighted')

    Returns:
        damage: [num_queries, num_pairs] - merge damage per query
    """
    seq_len, d_model = keys.shape
    num_queries, _ = queries.shape
    num_pairs = pairs.shape[0]
    device = keys.device

    if temperature is None:
        temperature = math.sqrt(d_model)

    # Compute original attention and output
    logits = queries @ keys.T / temperature  # [num_queries, seq_len]
    attentions = F.softmax(logits, dim=1)  # [num_queries, seq_len]
    output_original = attentions @ values  # [num_queries, d_model]

    damage = torch.zeros(num_queries, num_pairs, device=device)

    for pair_idx in range(num_pairs):
        i_pos = pairs[pair_idx, 0].item()
        j_pos = pairs[pair_idx, 1].item()

        # Get keys and values
        ki = keys[i_pos]  # [d_model]
        kj = keys[j_pos]  # [d_model]
        vi = values[i_pos]  # [d_model]
        vj = values[j_pos]  # [d_model]

        if merge_strategy == 'average':
            # Simple average
            k_merged = (ki + kj) / 2
            v_merged = (vi + vj) / 2

            # Create merged keys/values
            # Replace position i with merged, remove position j
            keys_merged = torch.cat([
                keys[:i_pos],
                k_merged.unsqueeze(0),
                keys[i_pos+1:j_pos],
                keys[j_pos+1:],
            ], dim=0)  # [seq_len - 1, d_model]

            values_merged = torch.cat([
                values[:i_pos],
                v_merged.unsqueeze(0),
                values[i_pos+1:j_pos],
                values[j_pos+1:],
            ], dim=0)  # [seq_len - 1, d_model]

        else:
            raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

        # Recompute attention with merged cache
        logits_merged = queries @ keys_merged.T / temperature  # [num_queries, seq_len-1]
        attentions_merged = F.softmax(logits_merged, dim=1)  # [num_queries, seq_len-1]
        output_merged = attentions_merged @ values_merged  # [num_queries, d_model]

        # Compute damage: ||output_original - output_merged||
        diff = output_original - output_merged  # [num_queries, d_model]
        damage[:, pair_idx] = torch.norm(diff, p=2, dim=1)  # [num_queries]

    return damage


def compute_future_attention_similarity(
    attentions: torch.Tensor,
    pairs: torch.Tensor,
    future_start: int,
    future_horizon: int,
) -> torch.Tensor:
    """
    Compute how similarly future queries attend to two positions.

    For positions i and j, compare attention columns A[:,i] vs A[:,j]
    over future queries.

    Args:
        attentions: [num_queries, seq_len] - full attention matrix
        pairs: [num_pairs, 2] - pairs of positions
        future_start: Start index for future queries
        future_horizon: Number of future queries to use

    Returns:
        similarities: [num_pairs] - cosine similarity of attention columns
    """
    num_queries, seq_len = attentions.shape
    num_pairs = pairs.shape[0]
    device = attentions.device

    # Validate future range
    future_end = future_start + future_horizon
    if future_end > num_queries:
        future_end = num_queries

    if future_start >= num_queries:
        # No future queries available
        return torch.zeros(num_pairs, device=device)

    # Extract future attention window
    attn_future = attentions[future_start:future_end, :]  # [horizon, seq_len]

    similarities = torch.zeros(num_pairs, device=device)

    for pair_idx in range(num_pairs):
        i_pos = pairs[pair_idx, 0].item()
        j_pos = pairs[pair_idx, 1].item()

        # Extract attention columns for positions i and j
        attn_col_i = attn_future[:, i_pos]  # [horizon]
        attn_col_j = attn_future[:, j_pos]  # [horizon]

        # Compute cosine similarity
        if attn_col_i.sum() == 0 or attn_col_j.sum() == 0:
            similarities[pair_idx] = 0.0
        else:
            sim = F.cosine_similarity(
                attn_col_i.unsqueeze(0),
                attn_col_j.unsqueeze(0),
                dim=1
            )
            similarities[pair_idx] = sim.item()

    return similarities


def compute_pairwise_attention_similarity(
    attentions: torch.Tensor,
    pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention-based similarity for pairs.

    Returns both:
    - Column similarity: how similarly do queries attend to i vs j?
    - Row similarity: how similarly do i and j attend to others? (less relevant for keys)

    Args:
        attentions: [num_queries, seq_len]
        pairs: [num_pairs, 2]

    Returns:
        (col_sim, row_sim): Both [num_pairs]
    """
    num_queries, seq_len = attentions.shape
    num_pairs = pairs.shape[0]
    device = attentions.device

    col_sim = torch.zeros(num_pairs, device=device)
    row_sim = torch.zeros(num_pairs, device=device)

    for pair_idx in range(num_pairs):
        i_pos = pairs[pair_idx, 0].item()
        j_pos = pairs[pair_idx, 1].item()

        # Column similarity (more relevant for keys)
        attn_col_i = attentions[:, i_pos]  # [num_queries]
        attn_col_j = attentions[:, j_pos]  # [num_queries]

        if attn_col_i.sum() > 0 and attn_col_j.sum() > 0:
            col_sim[pair_idx] = F.cosine_similarity(
                attn_col_i.unsqueeze(0),
                attn_col_j.unsqueeze(0),
                dim=1
            ).item()

        # Row similarity (how do i and j attend to others?)
        if i_pos < num_queries and j_pos < num_queries:
            attn_row_i = attentions[i_pos, :]  # [seq_len]
            attn_row_j = attentions[j_pos, :]  # [seq_len]

            if attn_row_i.sum() > 0 and attn_row_j.sum() > 0:
                row_sim[pair_idx] = F.cosine_similarity(
                    attn_row_i.unsqueeze(0),
                    attn_row_j.unsqueeze(0),
                    dim=1
                ).item()

    return col_sim, row_sim
