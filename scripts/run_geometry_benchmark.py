#!/usr/bin/env python3
"""
Geometry benchmark for KV-cache functional redundancy.

Compares multiple geometric distance metrics to determine which best predicts
whether two KV-cache entries are functionally redundant.

Usage:
    python scripts/run_geometry_benchmark.py \
        --model pythia-1b \
        --dataset wikitext \
        --num-samples 100 \
        --layer 6 \
        --num-pairs 500 \
        --output results/geometry_benchmark.csv
"""

import argparse
import sys
import os
from pathlib import Path

# Module-level storage for geometry metadata
# Maps geometry name -> higher_is_more_similar (bool)
GEOMETRY_METADATA = {}

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

# Import geometries
from src.geometries import (
    EuclideanMetric,
    CosineDistanceMetric,
    CosineSimilarityMetric,
    SphericalRadialMetric,
    SphericalOnlyMetric,
    RadialOnlyMetric,
    ExponentialResponseMetric,
    FisherSymmetricMetric,
)
from src.geometries.mahalanobis import QueryMahalanobisOracleMetric, QueryMahalanobisCausalMetric

# Import ground truth
from src.ground_truth import (
    compute_deletion_damage,
    compute_merge_damage,
    compute_future_attention_similarity,
)


def extract_single_sample(model, tokenizer, text, max_length=512):
    """
    Extract KV cache states, attention weights, AND queries from a single text sample.

    Uses exact GPT-NeoX/Pythia implementation to reconstruct post-RoPE queries.

    Returns:
        dict with 'keys', 'values', 'queries', 'attention_weights' (all on GPU)
        or None if sample is invalid
    """
    model.eval()
    device = next(model.parameters()).device

    if not text or not text.strip():
        return None

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

        if inputs['input_ids'].shape[1] == 0:
            return None

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass to get KV cache and attention
        outputs = model(**inputs, output_attentions=True, use_cache=True, output_hidden_states=True)

        past_kv = outputs.past_key_values
        if past_kv is None or len(past_kv) == 0:
            return None

        # Extract keys, values from past_key_values (these are post-RoPE)
        layer_keys = [kv[0] for kv in past_kv]
        layer_values = [kv[1] for kv in past_kv]
        layer_attentions = [attn for attn in outputs.attentions]

        # Reconstruct queries from hidden states using correct layer norm + RoPE
        # Following exact GPT-NeoX/Pythia implementation (transformers 5.17.0)
        hidden_states = outputs.hidden_states
        layer_queries = []

        # Determine model architecture
        if hasattr(model, 'gpt_neox'):
            # Pythia / GPTNeoX (transformers 5.17.0+)
            layers = model.gpt_neox.layers
            num_heads = model.config.num_attention_heads

            # Import the actual apply_rotary_pos_emb from transformers
            from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

            for layer_idx, layer in enumerate(layers):
                # CRITICAL: Use layer-normalized hidden state
                # This is what actually enters the attention module
                h_raw = hidden_states[layer_idx]  # [batch, seq, hidden]
                h = layer.input_layernorm(h_raw)  # Layer norm before attention

                attn = layer.attention
                head_size = attn.head_size

                # Project to QKV
                qkv = attn.query_key_value(h)  # [batch, seq, 3 * hidden]
                batch_size, seq_len = h.shape[:2]

                # Reshape: [batch, seq, num_heads, 3 * head_size]
                qkv = qkv.view(batch_size, seq_len, num_heads, 3 * head_size)
                # Transpose: [batch, num_heads, seq, 3 * head_size]
                qkv = qkv.transpose(1, 2)

                # Split into Q, K, V
                q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, num_heads, seq, head_size]

                # Apply RoPE using model-level rotary embeddings
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                cos, sin = model.gpt_neox.rotary_emb(h, position_ids=position_ids)

                # Apply rotary position embeddings (exact HF implementation)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                layer_queries.append(q)
        else:
            # Unsupported architecture - fail immediately
            raise ValueError(
                f"Unsupported model architecture: {model.__class__.__name__}. "
                "Only GPT-NeoX/Pythia models are currently supported. "
                "Add support for this architecture or use a supported model."
            )

        # Keep tensors on GPU - caller will move to CPU for benchmarking
        return {
            'keys': layer_keys,
            'values': layer_values,
            'queries': layer_queries,
            'attention_weights': layer_attentions,
        }


def sanity_check_qk_reconstruction(queries, keys, attention_weights, head_idx, layer_idx,
                                   scaling_factor, tolerance=0.01):
    """
    Sanity check: verify that softmax(Q @ K^T * scaling + causal_mask) ≈ attention_weights.

    For GPT-NeoX: scaling = 1 / sqrt(head_dim)

    This check MUST pass - if reconstruction fails, the Q extraction is incorrect and
    the benchmark results would be invalid. Fails hard on tolerance violation.

    Args:
        queries: [batch, heads, seq, dim]
        keys: [batch, heads, seq, dim]
        attention_weights: [batch, heads, seq, seq]
        head_idx: which head to check
        layer_idx: which layer to check
        scaling_factor: attention scaling (1/sqrt(head_dim) for GPT-NeoX)
        tolerance: maximum allowed error (default: 0.01)

    Returns:
        max_error: maximum absolute difference
        mean_error: mean absolute difference

    Raises:
        RuntimeError: if max_error > tolerance
    """
    Q = queries[0, head_idx, :, :]  # [seq, dim]
    K = keys[0, head_idx, :, :]  # [seq, dim]
    A_true = attention_weights[0, head_idx, :, :]  # [seq, seq]

    seq_len, head_dim = Q.shape
    device = Q.device

    # Compute Q @ K^T * scaling
    logits = (Q @ K.T) * scaling_factor  # [seq, seq]

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    logits = logits.masked_fill(causal_mask, float('-inf'))

    # Apply softmax
    A_recon = torch.nn.functional.softmax(logits, dim=-1)  # [seq, seq]

    # Compute errors
    # Ignore positions with -inf or NaN
    valid_mask = ~torch.isnan(A_true) & ~torch.isinf(A_true)

    errors = torch.abs(A_recon - A_true)
    max_error = errors[valid_mask].max().item() if valid_mask.sum() > 0 else 0.0
    mean_error = errors[valid_mask].mean().item() if valid_mask.sum() > 0 else 0.0

    if max_error > tolerance:
        error_msg = (
            f"FATAL: QK reconstruction check failed for layer={layer_idx}, head={head_idx}\n"
            f"  Max error: {max_error:.6f} (tolerance: {tolerance})\n"
            f"  Mean error: {mean_error:.6f}\n"
            f"  Q shape: {Q.shape}, K shape: {K.shape}\n"
            f"  Scaling factor: {scaling_factor:.6f}\n"
            f"  This indicates Q or K tensors are not correctly extracted.\n"
            f"  Possible causes:\n"
            f"    - Wrong layer normalization (must use layer.input_layernorm)\n"
            f"    - Incorrect RoPE application\n"
            f"    - Mismatched coordinate systems between Q and K\n"
            f"  Benchmark results would be invalid - aborting."
        )
        raise RuntimeError(error_msg)

    print(f"✓ QK reconstruction verified: layer={layer_idx}, head={head_idx}, "
          f"max_err={max_error:.6f}, mean_err={mean_error:.6f}")

    return max_error, mean_error


def sample_pairs(seq_len, num_pairs, strategy='random', seed=42, unique=True):
    """
    Sample pairs of positions to evaluate.

    Strategies:
    - random: uniform random pairs
    - nearby: pairs within small distance
    - mixed: combination

    Args:
        seq_len: sequence length
        num_pairs: number of pairs to sample
        strategy: sampling strategy
        seed: random seed
        unique: if True, ensure all pairs are unique unordered pairs

    Returns:
        pairs: [num_pairs, 2] array of (i, j) pairs where i < j
    """
    np.random.seed(seed)

    if strategy == 'random':
        if unique:
            # Sample unique unordered pairs: ensure i < j and no duplicates
            max_possible_pairs = seq_len * (seq_len - 1) // 2
            num_to_sample = min(num_pairs, max_possible_pairs)

            # Generate all possible pairs (i, j) where i < j
            if num_to_sample > max_possible_pairs * 0.1:
                # If we need many pairs, generate all and sample
                all_pairs = []
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        all_pairs.append((i, j))
                indices = np.random.choice(len(all_pairs), size=num_to_sample, replace=False)
                pairs = [all_pairs[idx] for idx in indices]
            else:
                # If we need few pairs, sample with rejection
                pairs_set = set()
                while len(pairs_set) < num_to_sample:
                    i, j = np.random.choice(seq_len, size=2, replace=False)
                    # Canonicalize: ensure i < j
                    if i > j:
                        i, j = j, i
                    pairs_set.add((i, j))
                pairs = list(pairs_set)
        else:
            # Non-unique pairs (may have duplicates)
            pairs = []
            for _ in range(num_pairs):
                i, j = np.random.choice(seq_len, size=2, replace=False)
                # Canonicalize: ensure i < j
                if i > j:
                    i, j = j, i
                pairs.append((i, j))

        return np.array(pairs)

    elif strategy == 'nearby':
        # Sample pairs within distance 10
        pairs = [] if not unique else set()
        max_dist = min(10, seq_len // 4)

        attempts = 0
        max_attempts = num_pairs * 10  # prevent infinite loop

        while len(pairs) < num_pairs and attempts < max_attempts:
            i = np.random.randint(0, seq_len - 1)
            offset = np.random.randint(1, min(max_dist + 1, seq_len - i))
            j = i + offset

            if j < seq_len:
                # Always canonicalize: i < j (already satisfied by construction)
                if unique:
                    pairs.add((i, j))
                else:
                    pairs.append((i, j))

            attempts += 1

        if unique:
            pairs = list(pairs)

        return np.array(pairs[:num_pairs])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_benchmark_single_head(
    keys, values, queries, attentions,
    head_idx, layer_idx, dataset_name, sample_id,
    num_pairs=500, future_horizon=64, min_future_queries=32,
    pair_strategy='random', seed=42,
    sanity_check=True
):
    """
    Run benchmark for a single (layer, head, sample).

    Processes one sample at a time on CPU to save memory.

    Args:
        min_future_queries: Minimum number of future queries required to evaluate a pair.
                           Pairs near sequence end without sufficient future context are skipped.
                           Default: 32 (ensures stable ground truth estimates)

    Returns:
        DataFrame with one row per pair, columns for all metrics + ground truth
    """
    # keys: [batch=1, heads, seq, dim]
    # Move to CPU for processing (one sample at a time is more memory-efficient)
    keys = keys.cpu()
    values = values.cpu()
    queries = queries.cpu()
    attentions = attentions.cpu()

    # Extract for this head
    K = keys[0, head_idx, :, :]  # [seq, dim]
    V = values[0, head_idx, :, :]  # [seq, dim]
    Q = queries[0, head_idx, :, :]  # [seq, dim]
    A = attentions[0, head_idx, :, :]  # [seq, seq]

    seq_len, d_model = K.shape
    device = K.device
    head_dim = d_model

    # Sanity check: verify Q @ K^T reconstruction
    # CRITICAL: This MUST pass for valid Q extraction - fails hard on error
    if sanity_check and sample_id == 0:  # Only check first sample (one check per head is sufficient)
        # GPT-NeoX uses scaling = 1 / sqrt(head_dim)
        scaling_factor = head_dim ** -0.5
        max_err, mean_err = sanity_check_qk_reconstruction(
            queries, keys, attentions, head_idx, layer_idx,
            scaling_factor=scaling_factor, tolerance=0.01
        )

    # Sample pairs (unique unordered pairs)
    if seq_len < 2:
        return None

    pairs_np = sample_pairs(seq_len, num_pairs=min(num_pairs, seq_len * (seq_len - 1) // 2),
                           strategy=pair_strategy, seed=seed, unique=True)
    pairs = torch.tensor(pairs_np, dtype=torch.long, device=device)  # [num_pairs, 2]

    num_actual_pairs = pairs.shape[0]

    # Initialize result storage
    results = {
        'dataset': [dataset_name] * num_actual_pairs,
        'sample_id': [sample_id] * num_actual_pairs,
        'layer': [layer_idx] * num_actual_pairs,
        'head': [head_idx] * num_actual_pairs,
        'token_i': pairs_np[:, 0].tolist(),
        'token_j': pairs_np[:, 1].tolist(),
    }

    # Define geometries to test (with corrected temperature defaults)
    import math
    temp_sqrt_d = math.sqrt(head_dim)

    # Non-query geometries: compute on all pairs at once
    non_query_geometries = [
        EuclideanMetric(),
        CosineDistanceMetric(),
        CosineSimilarityMetric(),
        SphericalRadialMetric({'lambda_radial': 1.0}),
        SphericalOnlyMetric(),
        RadialOnlyMetric(),
    ]

    # Add p-adic baseline
    try:
        from src.geometries.padic import UltrametricMetric
        non_query_geometries.append(UltrametricMetric({'prime': 2, 'precision': 16}))
    except ImportError:
        print("Warning: Could not import p-adic metric")

    # Compute non-query geometry metrics (don't need queries)
    for geom in non_query_geometries:
        try:
            # Store metadata in global dict
            GEOMETRY_METADATA[geom.name] = geom.higher_is_more_similar

            # Precompute if needed
            precomputed = geom.precompute(K, V, queries=None)

            # Compute pairwise distances
            distances = geom.compute_pairwise(
                K, V, pairs, queries=None, **precomputed
            )

            results[geom.name] = distances.cpu().numpy().tolist()
        except Exception as e:
            import traceback
            print(f"ERROR: {geom.name} failed with exception:")
            print(f"  {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            results[geom.name] = [np.nan] * num_actual_pairs

    # Query-aware geometries: compute per-pair with appropriate query windows
    query_aware_geometries = [
        ('oracle', QueryMahalanobisOracleMetric()),
        ('oracle', ExponentialResponseMetric({'temperature': temp_sqrt_d, 'normalized': True})),
        ('oracle', FisherSymmetricMetric({'temperature': temp_sqrt_d})),
        ('causal', QueryMahalanobisCausalMetric()),
    ]

    for query_mode, geom in query_aware_geometries:
        try:
            # Store metadata in global dict
            GEOMETRY_METADATA[geom.name] = geom.higher_is_more_similar

            distances_list = []

            for pair_idx in range(num_actual_pairs):
                i_pos = pairs_np[pair_idx, 0]
                j_pos = pairs_np[pair_idx, 1]
                t = max(i_pos, j_pos)

                if query_mode == 'oracle':
                    # Oracle: use future queries Q[t+1:t+1+future_horizon]
                    future_start = t + 1
                    future_end = min(future_start + future_horizon, seq_len)
                    if future_start < seq_len:
                        Q_window = Q[future_start:future_end, :]
                        # Extract attention weights for future queries
                        A_window = A[future_start:future_end, :]  # [num_future, seq_len]
                    else:
                        Q_window = torch.empty(0, d_model, device=device)
                        A_window = None
                elif query_mode == 'causal':
                    # Causal: use past queries Q[max(0, t-window):t]
                    window_size = getattr(geom, 'causal_window', 128)
                    past_start = max(0, t - window_size)
                    past_end = t
                    if past_end > past_start:
                        Q_window = Q[past_start:past_end, :]
                        # Extract attention weights for past queries
                        A_window = A[past_start:past_end, :]  # [num_past, seq_len]
                    else:
                        Q_window = torch.empty(0, d_model, device=device)
                        A_window = None
                else:
                    Q_window = Q  # Fallback (shouldn't happen)
                    A_window = A

                # Compute distance for this single pair
                if Q_window.shape[0] > 0:
                    pair_single = torch.tensor([[i_pos, j_pos]], dtype=torch.long, device=device)
                    precomputed = geom.precompute(K, V, queries=Q_window)
                    distance = geom.compute_pairwise(
                        K, V, pair_single, queries=Q_window, attentions=A_window, **precomputed
                    )
                    distances_list.append(distance[0].item())
                else:
                    distances_list.append(np.nan)

            results[geom.name] = distances_list
        except Exception as e:
            import traceback
            print(f"ERROR: {geom.name} ({query_mode}) failed with exception:")
            print(f"  {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            results[geom.name] = [np.nan] * num_actual_pairs

    # Compute ground truth: future attention similarity
    # Use only valid future queries: Q[t+1:t+1+future_horizon] where t = max(i,j)
    # Skip pairs without sufficient future queries
    try:
        future_sims = []
        for pair_idx in range(num_actual_pairs):
            i_pos = pairs_np[pair_idx, 0]
            j_pos = pairs_np[pair_idx, 1]
            t = max(i_pos, j_pos)  # Latest position in the pair
            future_start = t + 1  # First valid future query

            # Check if we have minimum future queries
            future_end = min(future_start + future_horizon, seq_len)
            num_future = future_end - future_start

            if num_future < min_future_queries:
                # Skip: insufficient future context
                future_sims.append(np.nan)
                continue

            # Only use queries that occur after the pair exists in cache
            if future_start < seq_len and future_end > future_start:
                # Extract attention columns for future queries only
                attn_col_i = A[future_start:future_start+future_horizon, i_pos]
                attn_col_j = A[future_start:future_start+future_horizon, j_pos]

                sim = torch.nn.functional.cosine_similarity(
                    attn_col_i.unsqueeze(0),
                    attn_col_j.unsqueeze(0),
                    dim=1
                ).item()
                future_sims.append(sim)
            elif future_start < seq_len:
                # Use whatever future queries we have
                attn_col_i = A[future_start:seq_len, i_pos]
                attn_col_j = A[future_start:seq_len, j_pos]

                if len(attn_col_i) > 0:
                    sim = torch.nn.functional.cosine_similarity(
                        attn_col_i.unsqueeze(0),
                        attn_col_j.unsqueeze(0),
                        dim=1
                    ).item()
                    future_sims.append(sim)
                else:
                    future_sims.append(np.nan)
            else:
                future_sims.append(np.nan)

        results['future_attention_similarity'] = future_sims
    except Exception as e:
        import traceback
        print(f"ERROR: future_attention_similarity failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        results['future_attention_similarity'] = [np.nan] * num_actual_pairs

    # Compute ground truth: deletion damage
    # Use only valid future queries for each position
    try:
        deletion_damage_i = []
        deletion_damage_j = []

        for pair_idx in range(num_actual_pairs):
            i_pos = pairs_np[pair_idx, 0]
            j_pos = pairs_np[pair_idx, 1]
            t = max(i_pos, j_pos)
            future_start = t + 1

            # Check minimum future queries
            future_end = min(future_start + future_horizon, seq_len)
            num_future = future_end - future_start

            if num_future < min_future_queries:
                deletion_damage_i.append(np.nan)
                deletion_damage_j.append(np.nan)
                continue

            # Get valid future queries
            if future_start < seq_len:
                future_end = min(future_start + future_horizon, seq_len)
                Q_future = Q[future_start:future_end, :]
                if Q_future.shape[0] > 0:
                    # Compute absolute query positions for causal masking
                    query_positions = torch.arange(future_start, future_end, dtype=torch.long, device=device)

                    # Compute deletion damage for position i
                    pos_i_tensor = torch.tensor([i_pos], dtype=torch.long, device=device)
                    damage_i = compute_deletion_damage(
                        K, V, Q_future, pos_i_tensor,
                        temperature=temp_sqrt_d,
                        query_positions=query_positions
                    )
                    deletion_damage_i.append(damage_i.mean().item())

                    # Compute deletion damage for position j
                    pos_j_tensor = torch.tensor([j_pos], dtype=torch.long, device=device)
                    damage_j = compute_deletion_damage(
                        K, V, Q_future, pos_j_tensor,
                        temperature=temp_sqrt_d,
                        query_positions=query_positions
                    )
                    deletion_damage_j.append(damage_j.mean().item())
                else:
                    deletion_damage_i.append(np.nan)
                    deletion_damage_j.append(np.nan)
            else:
                deletion_damage_i.append(np.nan)
                deletion_damage_j.append(np.nan)

        results['deletion_damage_i'] = deletion_damage_i
        results['deletion_damage_j'] = deletion_damage_j
    except Exception as e:
        import traceback
        print(f"ERROR: deletion_damage failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        results['deletion_damage_i'] = [np.nan] * num_actual_pairs
        results['deletion_damage_j'] = [np.nan] * num_actual_pairs

    # Compute ground truth: merge damage
    # Use only valid future queries
    try:
        merge_damages = []

        for pair_idx in range(num_actual_pairs):
            i_pos = pairs_np[pair_idx, 0]
            j_pos = pairs_np[pair_idx, 1]
            t = max(i_pos, j_pos)
            future_start = t + 1

            # Check minimum future queries
            future_end = min(future_start + future_horizon, seq_len)
            num_future = future_end - future_start

            if num_future < min_future_queries:
                merge_damages.append(np.nan)
                continue

            # Get valid future queries
            if future_start < seq_len:
                future_end = min(future_start + future_horizon, seq_len)
                Q_future = Q[future_start:future_end, :]
                if Q_future.shape[0] > 0:
                    # Compute absolute query positions for causal masking
                    query_positions = torch.arange(future_start, future_end, dtype=torch.long, device=device)

                    # Compute merge damage for this pair
                    pair_tensor = torch.tensor([[i_pos, j_pos]], dtype=torch.long, device=device)
                    damage = compute_merge_damage(
                        K, V, Q_future, pair_tensor,
                        temperature=temp_sqrt_d,
                        query_positions=query_positions
                    )
                    merge_damages.append(damage.mean().item())
                else:
                    merge_damages.append(np.nan)
            else:
                merge_damages.append(np.nan)

        results['merge_damage'] = merge_damages
    except Exception as e:
        import traceback
        print(f"ERROR: merge_damage failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        results['merge_damage'] = [np.nan] * num_actual_pairs

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Geometry benchmark for KV-cache redundancy")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
    parser.add_argument("--head", type=int, default=None, help="Head to analyze (None = all heads)")
    parser.add_argument("--num-pairs", type=int, default=500, help="Pairs per head")
    parser.add_argument("--future-horizon", type=int, default=64, help="Future queries for attention sim")
    parser.add_argument("--min-future-queries", type=int, default=32,
                        help="Minimum future queries required to evaluate a pair (default: 32)")
    parser.add_argument("--pair-strategy", type=str, default="random", choices=["random", "nearby"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--cache-dir", type=str, default="/workspace/kv-probe/checkpoints/pretrained")

    args = parser.parse_args()

    print("="*80)
    print("GEOMETRY BENCHMARK FOR KV-CACHE REDUNDANCY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Layer: {args.layer}")
    print(f"Head: {args.head if args.head is not None else 'all'}")
    print(f"Pairs per head: {args.num_pairs}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{args.model}",
        cache_dir=args.cache_dir,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{args.model}", cache_dir=args.cache_dir)
    print("✓ Model loaded")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    if args.dataset == 'wikitext':
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
        texts = [ex['text'] for ex in dataset if ex['text'].strip()][:args.num_samples]
    else:
        raise ValueError(f"Dataset {args.dataset} not yet supported")

    print(f"Found {len(texts)} valid text samples")

    # Determine which heads to test (need to extract one sample first to know num_heads)
    print(f"\nExtracting first sample to determine model structure...")
    sample_data_first = None
    for text in texts:
        sample_data_first = extract_single_sample(model, tokenizer, text, max_length=args.max_length)
        if sample_data_first is not None:
            break

    if sample_data_first is None:
        print("ERROR: No valid samples found")
        return

    layer_idx = args.layer
    num_heads = sample_data_first['keys'][layer_idx].shape[1]  # [batch, heads, seq, dim]
    heads_to_test = [args.head] if args.head is not None else list(range(num_heads))

    print(f"Model has {num_heads} heads per layer")
    print(f"Testing layer {layer_idx}, heads: {heads_to_test}")

    # Clean up first sample
    del sample_data_first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Process samples one at a time
    print(f"\nProcessing samples and benchmarking...")
    all_results = []

    for sample_id, text in enumerate(tqdm(texts, desc="Benchmarking samples")):
        # Extract Q/K/V/attention for this sample
        sample_data = extract_single_sample(model, tokenizer, text, max_length=args.max_length)

        if sample_data is None:
            continue

        # Get tensors for requested layer (still on GPU)
        keys_layer = sample_data['keys'][layer_idx]
        values_layer = sample_data['values'][layer_idx]
        queries_layer = sample_data['queries'][layer_idx]
        attns_layer = sample_data['attention_weights'][layer_idx]

        # Benchmark each requested head
        for head_idx in heads_to_test:
            df = run_benchmark_single_head(
                keys_layer, values_layer, queries_layer, attns_layer,
                head_idx, layer_idx, args.dataset, sample_id,
                num_pairs=args.num_pairs,
                future_horizon=args.future_horizon,
                min_future_queries=args.min_future_queries,
                pair_strategy=args.pair_strategy,
                seed=args.seed + sample_id * 100 + head_idx,
                sanity_check=(sample_id == 0),  # Only check first sample
            )

            if df is not None:
                all_results.append(df)

        # Explicitly delete tensors to free memory
        del sample_data, keys_layer, values_layer, queries_layer, attns_layer

        # Clear GPU cache after each sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine all results
    if len(all_results) == 0:
        print("ERROR: No valid results generated")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")

    # Compute aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80)

    # Get geometry columns
    geom_cols = [c for c in results_df.columns if c not in [
        'dataset', 'sample_id', 'layer', 'head', 'token_i', 'token_j',
        'future_attention_similarity', 'deletion_damage_i', 'deletion_damage_j', 'merge_damage'
    ]]

    ground_truths = ['future_attention_similarity', 'merge_damage']

    # Overall aggregate
    for gt in ground_truths:
        if gt not in results_df.columns:
            continue

        print(f"\nGround truth: {gt} (Overall)")
        print("-" * 80)

        for geom_col in geom_cols:
            # Remove NaN values
            mask = ~(results_df[geom_col].isna() | results_df[gt].isna())
            if mask.sum() < 10:
                continue

            x = results_df.loc[mask, geom_col].values
            y = results_df.loc[mask, gt].values

            # Compute correlations
            try:
                pearson_r, _ = pearsonr(x, y)
                spearman_r, _ = spearmanr(x, y)

                # Compute AUROC / AUPRC for classification
                # Threshold ground truth at median to create binary labels
                if gt == 'future_attention_similarity':
                    # High similarity = safe to merge (positive class)
                    y_binary = (y > np.median(y)).astype(int)
                elif 'damage' in gt:
                    # Low damage = safe to merge (positive class)
                    y_binary = (y < np.median(y)).astype(int)
                else:
                    y_binary = None

                # Compute predictor scores from geometry metric
                # Use metadata to determine if we need to negate
                if y_binary is not None:
                    # Retrieve geometry metadata from global dict
                    is_similarity = GEOMETRY_METADATA.get(geom_col, False)

                    if is_similarity:
                        # Similarity metric: higher = more similar = safer to merge
                        # Use directly as "merge safety" score
                        scores = x
                    else:
                        # Distance metric: higher = less similar = less safe to merge
                        # Negate to get "merge safety" score
                        scores = -x

                    try:
                        auroc = roc_auc_score(y_binary, scores)
                        auprc = average_precision_score(y_binary, scores)
                        print(f"  {geom_col:30s} | Pearson: {pearson_r:+.3f} | Spearman: {spearman_r:+.3f} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")
                    except:
                        print(f"  {geom_col:30s} | Pearson: {pearson_r:+.3f} | Spearman: {spearman_r:+.3f}")
                else:
                    print(f"  {geom_col:30s} | Pearson: {pearson_r:+.3f} | Spearman: {spearman_r:+.3f}")
            except:
                pass

    # Per layer × head analysis
    print("\n" + "="*80)
    print("PER-LAYER PER-HEAD RESULTS")
    print("="*80)

    # Group by layer and head
    for (layer_id, head_id), group_df in results_df.groupby(['layer', 'head']):
        if len(group_df) < 10:
            continue

        print(f"\nLayer {layer_id}, Head {head_id}")
        print("-" * 40)

        for gt in ground_truths:
            if gt not in group_df.columns:
                continue

            # Find best geometry for this head
            best_geom = None
            best_abs_spearman = -np.inf
            best_spearman = 0.0

            for geom_col in geom_cols:
                mask = ~(group_df[geom_col].isna() | group_df[gt].isna())
                if mask.sum() < 5:
                    continue

                x = group_df.loc[mask, geom_col].values
                y = group_df.loc[mask, gt].values

                try:
                    spearman_r, _ = spearmanr(x, y)
                    abs_spearman = abs(spearman_r)
                    if abs_spearman > best_abs_spearman:
                        best_abs_spearman = abs_spearman
                        best_spearman = spearman_r
                        best_geom = geom_col
                except:
                    pass

            if best_geom:
                print(f"  Best for {gt}: {best_geom} (ρ={best_spearman:+.3f})")

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
