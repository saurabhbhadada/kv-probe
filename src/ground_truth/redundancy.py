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
    query_positions: Optional[torch.Tensor] = None,
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
        query_positions: [num_queries] - absolute position of each query in sequence
                         Used for causal masking: query at position q attends only to keys <= q

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

    # Apply causal mask if query_positions provided
    if query_positions is not None:
        # query_positions: [num_queries], values are absolute positions in sequence
        # Create causal mask: query at position q can only attend to keys at positions <= q
        # mask[q, k] = True if key position k > query position q (should be masked)
        query_pos_expanded = query_positions.unsqueeze(1)  # [num_queries, 1]
        key_positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        causal_mask = key_positions > query_pos_expanded  # [num_queries, seq_len]
        logits = logits.masked_fill(causal_mask, float('-inf'))

    attentions = F.softmax(logits, dim=1)  # [num_queries, seq_len]

    # Compute original output: o = sum_i a_i v_i
    # [num_queries, seq_len] @ [seq_len, d_model] = [num_queries, d_model]
    output_original = attentions @ values  # [num_queries, d_model]

    # Vectorized computation for all positions
    num_positions = positions.shape[0]

    # Get attention weights for all positions: [num_queries, num_positions]
    a_j = attentions[:, positions]  # [num_queries, num_positions]

    # Get values for all positions: [num_positions, d_model]
    v_j = values[positions]  # [num_positions, d_model]

    # Compute ||v_j - o|| for all positions
    # v_j: [num_positions, d_model], output_original: [num_queries, d_model]
    # Need: [num_queries, num_positions, d_model]
    v_j_expanded = v_j.unsqueeze(0)  # [1, num_positions, d_model]
    output_expanded = output_original.unsqueeze(1)  # [num_queries, 1, d_model]
    diff = v_j_expanded - output_expanded  # [num_queries, num_positions, d_model]
    diff_norm = torch.norm(diff, p=2, dim=2)  # [num_queries, num_positions]

    # Damage factor: a_j / (1 - a_j), with numerical protection
    a_j_safe = torch.clamp(a_j, 0.0, 0.9999)  # [num_queries, num_positions]
    damage_factor = a_j_safe / (1 - a_j_safe)  # [num_queries, num_positions]

    # Final damage
    damage = damage_factor * diff_norm  # [num_queries, num_positions]

    return damage


def compute_merge_damage(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    pairs: torch.Tensor,
    temperature: Optional[float] = None,
    merge_strategy: str = 'average',
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute damage from merging pairs of tokens.

    Uses exact closed-form formula without reconstructing KV cache:
        For merging i and j:
        z_m = (z_i + z_j)/2  (merged logit)
        w_m = exp(z_m)       (merged unnormalized weight)
        Z' = Z - w_i - w_j + w_m
        N' = N - w_i*v_i - w_j*v_j + w_m*v_m
        o' = N'/Z'
        damage = ||o - o'||

    Merge strategies:
    - 'average': k_merged = (k_i + k_j)/2, v_merged = (v_i + v_j)/2
    - 'weighted': weight by original attention weights (more complex)

    Args:
        keys: [seq_len, d_model]
        values: [seq_len, d_model]
        queries: [num_queries, d_model]
        pairs: [num_pairs, 2] - pairs to merge (will be canonicalized to i < j)
        temperature: Temperature for attention
        merge_strategy: How to merge ('average' or 'weighted')
        query_positions: [num_queries] - absolute position of each query in sequence
                         Used for causal masking: query at position q attends only to keys <= q

    Returns:
        damage: [num_queries, num_pairs] - merge damage per query
    """
    seq_len, d_model = keys.shape
    num_queries, _ = queries.shape
    num_pairs = pairs.shape[0]
    device = keys.device

    if temperature is None:
        temperature = math.sqrt(d_model)

    if merge_strategy != 'average':
        raise ValueError(f"Only 'average' merge_strategy is supported, got: {merge_strategy}")

    # Compute original logits and attention
    logits = queries @ keys.T / temperature  # [num_queries, seq_len]

    # Apply causal mask if query_positions provided
    if query_positions is not None:
        query_pos_expanded = query_positions.unsqueeze(1)  # [num_queries, 1]
        key_positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        causal_mask = key_positions > query_pos_expanded  # [num_queries, seq_len]
        logits = logits.masked_fill(causal_mask, float('-inf'))

    # For numerical stability, use log-sum-exp trick
    # Store max logit for each query for later use
    max_logits = logits.max(dim=1, keepdim=True).values  # [num_queries, 1]

    # Compute unnormalized weights (shifted by max for stability)
    logits_shifted = logits - max_logits  # [num_queries, seq_len]
    unnorm_weights = torch.exp(logits_shifted)  # [num_queries, seq_len]

    # Compute normalization constant Z
    Z = unnorm_weights.sum(dim=1, keepdim=True)  # [num_queries, 1]

    # Compute original output: N = sum_k w_k v_k
    N = unnorm_weights @ values  # [num_queries, d_model]
    output_original = N / Z  # [num_queries, d_model]

    # Vectorized computation for all pairs
    # Canonicalize pairs: ensure i < j
    i_positions = torch.minimum(pairs[:, 0], pairs[:, 1])  # [num_pairs]
    j_positions = torch.maximum(pairs[:, 0], pairs[:, 1])  # [num_pairs]

    # Get logits for positions i and j: [num_queries, num_pairs]
    z_i = logits_shifted[:, i_positions]  # [num_queries, num_pairs]
    z_j = logits_shifted[:, j_positions]  # [num_queries, num_pairs]

    # Compute merged logit (average in original space, then shift)
    # z_merged_orig = (z_i_orig + z_j_orig) / 2
    # z_i_orig = z_i + max_logits, z_j_orig = z_j + max_logits
    # z_merged_orig = (z_i + z_j) / 2 + max_logits
    # z_merged_shifted = z_merged_orig - max_logits = (z_i + z_j) / 2
    z_m = (z_i + z_j) / 2  # [num_queries, num_pairs]

    # Compute unnormalized weights
    w_i = torch.exp(z_i)  # [num_queries, num_pairs]
    w_j = torch.exp(z_j)  # [num_queries, num_pairs]
    w_m = torch.exp(z_m)  # [num_queries, num_pairs]

    # Get values for positions i and j
    v_i = values[i_positions]  # [num_pairs, d_model]
    v_j = values[j_positions]  # [num_pairs, d_model]
    v_m = (v_i + v_j) / 2  # [num_pairs, d_model]

    # Compute adjusted normalization: Z' = Z - w_i - w_j + w_m
    # Z: [num_queries, 1], w_i/w_j/w_m: [num_queries, num_pairs]
    Z_adjusted = Z - w_i - w_j + w_m  # [num_queries, num_pairs]

    # Compute adjusted numerator: N' = N - w_i*v_i - w_j*v_j + w_m*v_m
    # N: [num_queries, d_model]
    # w_i: [num_queries, num_pairs], v_i: [num_pairs, d_model]
    # w_i * v_i needs broadcasting: [num_queries, num_pairs, 1] * [1, num_pairs, d_model]
    w_i_expanded = w_i.unsqueeze(2)  # [num_queries, num_pairs, 1]
    w_j_expanded = w_j.unsqueeze(2)  # [num_queries, num_pairs, 1]
    w_m_expanded = w_m.unsqueeze(2)  # [num_queries, num_pairs, 1]

    v_i_expanded = v_i.unsqueeze(0)  # [1, num_pairs, d_model]
    v_j_expanded = v_j.unsqueeze(0)  # [1, num_pairs, d_model]
    v_m_expanded = v_m.unsqueeze(0)  # [1, num_pairs, d_model]

    # N: [num_queries, d_model] -> [num_queries, 1, d_model]
    N_expanded = N.unsqueeze(1)  # [num_queries, 1, d_model]

    # Compute adjustment
    N_adjusted = N_expanded - w_i_expanded * v_i_expanded - w_j_expanded * v_j_expanded + w_m_expanded * v_m_expanded
    # [num_queries, num_pairs, d_model]

    # Compute merged output: o' = N' / Z'
    Z_adjusted_expanded = Z_adjusted.unsqueeze(2)  # [num_queries, num_pairs, 1]
    output_merged = N_adjusted / Z_adjusted_expanded  # [num_queries, num_pairs, d_model]

    # Compute damage: ||o - o'||
    output_original_expanded = output_original.unsqueeze(1)  # [num_queries, 1, d_model]
    diff = output_original_expanded - output_merged  # [num_queries, num_pairs, d_model]
    damage = torch.norm(diff, p=2, dim=2)  # [num_queries, num_pairs]

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

    # Extract attention columns for all pairs
    # pairs[:, 0]: [num_pairs], pairs[:, 1]: [num_pairs]
    attn_col_i = attn_future[:, pairs[:, 0]]  # [horizon, num_pairs]
    attn_col_j = attn_future[:, pairs[:, 1]]  # [horizon, num_pairs]

    # Compute cosine similarity for all pairs
    # F.cosine_similarity expects [N, d] shapes, computes similarity along dim
    # Transpose to [num_pairs, horizon]
    attn_col_i_T = attn_col_i.T  # [num_pairs, horizon]
    attn_col_j_T = attn_col_j.T  # [num_pairs, horizon]

    # Compute norms
    norm_i = torch.norm(attn_col_i_T, p=2, dim=1)  # [num_pairs]
    norm_j = torch.norm(attn_col_j_T, p=2, dim=1)  # [num_pairs]

    # Compute dot products
    dot_products = (attn_col_i_T * attn_col_j_T).sum(dim=1)  # [num_pairs]

    # Compute cosine similarity with protection against zero norms
    similarities = torch.zeros(num_pairs, device=device)
    valid_mask = (norm_i > 0) & (norm_j > 0)
    similarities[valid_mask] = dot_products[valid_mask] / (norm_i[valid_mask] * norm_j[valid_mask])

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
