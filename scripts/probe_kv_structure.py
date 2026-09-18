#!/usr/bin/env python3
"""
Probe for p-adic/ultrametric structure in transformer KEYS (K only).

SCOPE: This script probes KEY structure. Values (V) are extracted but NOT analyzed.
A separate V probe is needed to test if values exhibit p-adic structure.

This is the MAIN RESEARCH EXPERIMENT - determines if p-adic compression is viable.

Research Question:
    Do transformer KEY representations exhibit meaningful ultrametric structure
    where p-adic distance between K[i] and K[j] predicts their behavioral
    similarity better than Euclidean distance?

Behavioral similarity:
    How similarly do future queries attend to K[i] vs K[j]?
    Measured as correlation between attention columns A[:,i] and A[:,j]

Usage:
    python scripts/probe_kv_structure.py \\
        --model pythia-1b \\
        --dataset wikitext \\
        --num-samples 2000 \\
        --layer 6 \\
        --output results/K_probe_layer6.json

TODO: Create probe_V_structure.py to analyze value similarity vs output similarity
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path (works both locally and in Docker)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import spearmanr

from src.geometries.padic import _padic_valuation


def extract_kv_states(model, tokenizer, texts, max_length=512):
    """
    Extract KV cache states from real model inference.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of input texts
        max_length: Maximum sequence length

    Returns:
        dict: {
            'keys': List of key tensors per layer,
            'values': List of value tensors per layer,
            'attention_weights': Attention weights for analysis,
            'tokens': Token IDs
        }
    """
    model.eval()
    device = next(model.parameters()).device

    all_keys = []
    all_values = []
    all_attention = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting KV states"):
            # Skip empty texts
            if not text or not text.strip():
                continue

            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

            # Skip if tokenization failed
            if inputs['input_ids'].shape[1] == 0:
                continue

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with output_attentions=True
            outputs = model(**inputs, output_attentions=True, use_cache=True)

            # Extract KV cache from past_key_values
            past_kv = outputs.past_key_values
            if past_kv is None or len(past_kv) == 0:
                continue

            # Store keys, values, and attentions
            layer_keys = [kv[0].cpu() for kv in past_kv]
            layer_values = [kv[1].cpu() for kv in past_kv]
            layer_attentions = [attn.cpu() for attn in outputs.attentions]

            all_keys.append(layer_keys)
            all_values.append(layer_values)
            all_attention.append(layer_attentions)

    return {
        'keys': all_keys,
        'values': all_values,
        'attention_weights': all_attention,
    }


def compute_euclidean_distance(k1, k2):
    """Compute L2 distance between two tensors."""
    return torch.norm(k1 - k2, p=2).item()


def compute_cosine_similarity(k1, k2):
    """Compute cosine similarity between two tensors."""
    k1_flat = k1.flatten()
    k2_flat = k2.flatten()
    return torch.nn.functional.cosine_similarity(k1_flat, k2_flat, dim=0).item()


def compute_quantization_scale(keys_all, precision=16):
    """
    Compute ONE fixed quantization scale for all keys in a layer/head.

    CRITICAL: The integer representation of a key must NOT depend on which
    other key it is compared against. All keys must use the same scale.

    Args:
        keys_all: All key vectors for this layer/head [num_samples, dim]
        precision: Bit precision

    Returns:
        scale: Fixed quantization scale to use for all keys
    """
    # Find max absolute value across ALL keys
    max_abs = keys_all.abs().max().item()

    if max_abs < 1e-8:
        return 1.0

    # Symmetric quantization: map to [-2^(p-1), 2^(p-1)]
    int_range = 2 ** (precision - 1)
    scale = int_range / max_abs

    return scale


def quantize_keys_fixed_scale(keys, scale, precision=16):
    """
    Quantize keys using a FIXED pre-computed scale.

    Args:
        keys: Key vectors to quantize [N, dim]
        scale: Pre-computed fixed scale
        precision: Bit precision

    Returns:
        keys_int: Quantized keys
    """
    keys_float = keys.float()
    keys_int = torch.round(keys_float * scale).to(torch.int64)
    return keys_int


def compute_padic_statistics_fixed_scale(k1_int, k2_int, precision=16, prime=2):
    """
    Compute p-adic statistics between two PRE-QUANTIZED integer tensors.

    CRITICAL: Uses FIXED scale quantization - keys already quantized with
    shared scale, so integer representation doesn't depend on pair.

    Traditional min(v_p) is degenerate in high dimensions!
    For 256-D vectors, P(min > 0) ≈ (1/2)^256 ≈ 0, making the metric flat.

    We compute robust S_k statistics instead:
    S_k = fraction of dimensions where v_p(K[i,d] - K[j,d]) >= k

    Interpretation:
    - S_1: % of dims sharing ≥1 low-order binary digit
    - S_2: % of dims sharing ≥2 low-order binary digits (most robust)
    - High S_k => many dimensions have shared p-adic structure

    Args:
        k1_int, k2_int: Pre-quantized integer tensors (shape: [dim])
        precision: Bit precision (for capping valuations)
        prime: Prime for p-adic metric (2, 3, 5, etc.)

    Returns:
        dict with:
        - min_valuation: min v_p across dimensions (degenerate but traditional)
        - mean_valuation: mean v_p (robust alternative)
        - S_k: fraction of dims with v_p >= k for k=1,2,3,4
        - valuations: full per-dimension valuations
    """

    # Compute difference
    diff = k1_int - k2_int

    # Compute p-adic valuation per dimension
    diff_flat = diff.flatten()
    valuations = _padic_valuation(diff_flat, prime=prime, precision=precision)

    # Traditional min-based metric (degenerate in high dims)
    min_val = valuations.min().item()

    # Robust statistics
    mean_val = valuations.float().mean().item()

    # S_k: fraction of dimensions with v_p >= k
    dim = len(valuations)
    S1 = (valuations >= 1).float().sum().item() / dim
    S2 = (valuations >= 2).float().sum().item() / dim
    S3 = (valuations >= 3).float().sum().item() / dim
    S4 = (valuations >= 4).float().sum().item() / dim

    return {
        'min_valuation': min_val,
        'mean_valuation': mean_val,
        'S1': S1,
        'S2': S2,
        'S3': S3,
        'S4': S4,
        'valuations': valuations,
    }


def generate_pair_manifest(keys, attentions, num_pairs_per_head, future_horizon=64, seed=42):
    """
    Generate ONE fixed manifest of (seq_id, head, i, j) pairs to analyze.

    CRITICAL: Must use the SAME pairs for all metrics (S_k, cosine, Euclidean,
    different primes, different precisions, permutation control, etc.)

    FIX 2: Generate pairs PER HEAD, not randomly selecting heads.
    Each head gets its own independent set of pairs.

    FIX 4: Safety guard against infinite loops.

    Args:
        keys: List of key tensors per sequence
        attentions: List of attention tensors per sequence
        num_pairs_per_head: Number of pairs to sample for EACH head
        future_horizon: Number of future queries required
        seed: Random seed for reproducibility

    Returns:
        Dict mapping head_idx -> List of (seq_id, i, j) tuples
    """
    np.random.seed(seed)

    num_heads = keys[0][0].shape[0]
    pairs_per_head = {h: [] for h in range(num_heads)}

    # Safety: maximum attempts per head
    max_attempts_per_head = num_pairs_per_head * 100

    for head_idx in range(num_heads):
        attempts = 0

        while len(pairs_per_head[head_idx]) < num_pairs_per_head:
            if attempts >= max_attempts_per_head:
                raise RuntimeError(
                    f"Unable to generate {num_pairs_per_head} pairs for head {head_idx}. "
                    f"Only found {len(pairs_per_head[head_idx])} valid pairs after {attempts} attempts. "
                    f"Try reducing --future-horizon or --num-pairs."
                )

            attempts += 1

            # Sample random sequence
            seq_idx = np.random.randint(len(keys))
            key_tensor = keys[seq_idx][0]  # [heads, seq, dim]

            seq_len = key_tensor.shape[1]

            # Sample two positions
            if seq_len < 2:
                continue

            i, j = np.random.choice(seq_len, size=2, replace=False)

            # Check if we have enough future context
            future_start = max(i, j) + 1
            if future_start + future_horizon > seq_len:
                continue  # Not enough future context

            pairs_per_head[head_idx].append((seq_idx, i, j))

    return pairs_per_head


def generate_random_baseline(keys):
    """
    Generate random tensors with same marginal distribution as real keys.

    This is a critical control: if p-adic structure exists in transformers,
    it should be STRONGER than random tensors with the same statistics.
    """
    random_keys = []
    for key_tensor in keys:
        # key_tensor shape: [batch=1, heads, seq, dim]
        shape = key_tensor.shape

        # Match mean and std of real keys
        mean = key_tensor.mean().item()
        std = key_tensor.std().item()

        # Generate random tensor with same distribution
        random_tensor = torch.randn(shape) * std + mean
        random_keys.append(random_tensor)

    return random_keys


def compute_all_statistics_fixed_pairs(keys, attentions, head_idx, pair_list, prime, precision, future_horizon, quantization_scales, use_permuted=False):
    """
    Compute statistics for all pairs in ONE HEAD using FIXED quantization.

    FIX 2: This function now operates on a single head only.

    Args:
        keys: List of key tensors [seq][batch=1, heads, seq, dim]
        attentions: List of attention tensors [seq][batch=1, heads, seq, seq]
        head_idx: Which attention head to analyze
        pair_list: List of (seq_id, i, j) tuples for this head
        prime: Prime for p-adic
        precision: Bit precision
        future_horizon: Number of future queries to use
        quantization_scales: Dict[(head, precision)] -> scale (FIX 1: shared across sequences)
        use_permuted: If True, permute keys across positions (null control)

    Returns:
        dict with arrays of: euclidean, cosine, S1, S2, S3, S4, attention_sim
    """
    euclidean_dists = []
    cosine_sims = []
    S1_vals = []
    S2_vals = []
    S3_vals = []
    S4_vals = []
    attention_sims = []

    # FIX 1: Get the shared scale for this head
    scale_key = (head_idx, precision)
    if scale_key not in quantization_scales:
        raise ValueError(f"Scale for head {head_idx}, precision {precision} not precomputed")
    scale = quantization_scales[scale_key]

    # Pre-quantize all keys for this head using the SHARED scale
    quantized_keys = {}
    for seq_id in range(len(keys)):
        key_tensor = keys[seq_id][0]  # [heads, seq, dim]
        seq_len = key_tensor.shape[1]

        K_head = key_tensor[head_idx]  # [seq, dim]

        # Optionally permute keys across positions (null control)
        if use_permuted:
            perm = torch.randperm(seq_len)
            K_head = K_head[perm]

        # Quantize with SHARED scale (same scale for all sequences)
        K_int = quantize_keys_fixed_scale(K_head, scale, precision)
        quantized_keys[seq_id] = K_int

    # Process each pair for this head
    for seq_id, i, j in tqdm(pair_list, desc=f"head={head_idx}, p={prime}, prec={precision}, perm={use_permuted}", leave=False):
        K_int = quantized_keys[seq_id]  # [seq, dim]
        key_tensor = keys[seq_id][0]  # [heads, seq, dim]
        attn_tensor = attentions[seq_id][0]  # [heads, seq, seq]

        # Extract key vectors (float for Euclidean/cosine)
        ki_float = key_tensor[head_idx, i, :]
        kj_float = key_tensor[head_idx, j, :]

        # Euclidean distance
        euclidean_dists.append(compute_euclidean_distance(ki_float, kj_float))

        # Cosine similarity
        cosine_sims.append(compute_cosine_similarity(ki_float, kj_float))

        # P-adic statistics (using pre-quantized integers)
        ki_int = K_int[i]
        kj_int = K_int[j]
        padic_stats = compute_padic_statistics_fixed_scale(ki_int, kj_int, precision, prime)

        S1_vals.append(padic_stats['S1'])
        S2_vals.append(padic_stats['S2'])
        S3_vals.append(padic_stats['S3'])
        S4_vals.append(padic_stats['S4'])

        # Attention similarity (COLUMNS, fixed future horizon)
        future_start = max(i, j) + 1
        attn_col_i = attn_tensor[head_idx, future_start:future_start+future_horizon, i]
        attn_col_j = attn_tensor[head_idx, future_start:future_start+future_horizon, j]

        attn_sim = torch.nn.functional.cosine_similarity(
            attn_col_i, attn_col_j, dim=0
        ).item()
        attention_sims.append(attn_sim)

    return {
        'euclidean': np.array(euclidean_dists),
        'cosine': np.array(cosine_sims),
        'S1': np.array(S1_vals),
        'S2': np.array(S2_vals),
        'S3': np.array(S3_vals),
        'S4': np.array(S4_vals),
        'attention': np.array(attention_sims),
    }


def bootstrap_correlation_difference(x1, x2, y, pair_list, n_bootstrap=1000, ci=95):
    """
    Bootstrap the DIFFERENCE between two Spearman correlations.

    FIX 3: Use observed delta as point estimate, bootstrap only for CI.

    Uses sequence-level resampling: sample sequences with replacement,
    include all pairs from sampled sequences.

    Args:
        x1, x2: Two metrics to compare (e.g., S2 and cosine)
        y: Target (attention similarity)
        pair_list: List of (seq_id, i, j) tuples for one head
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level

    Returns:
        (delta_observed, ci_lower, ci_upper)
    """
    # FIX 3: Compute observed delta (point estimate)
    rho1_obs, _ = spearmanr(x1, y)
    rho2_obs, _ = spearmanr(x2, y)
    delta_observed = rho1_obs - rho2_obs

    # Get unique sequence IDs
    seq_ids = list(set([seq_id for seq_id, _, _ in pair_list]))
    n_seqs = len(seq_ids)

    deltas = []

    for _ in range(n_bootstrap):
        # Resample sequences with replacement
        boot_seqs = np.random.choice(seq_ids, size=n_seqs, replace=True)

        # Get indices of pairs from sampled sequences
        boot_indices = []
        for s in boot_seqs:
            indices = [idx for idx, (seq_id, _, _) in enumerate(pair_list) if seq_id == s]
            boot_indices.extend(indices)

        if len(boot_indices) < 10:  # Need enough pairs
            continue

        # Compute correlations on bootstrap sample (use Spearman)
        rho1, _ = spearmanr(x1[boot_indices], y[boot_indices])
        rho2, _ = spearmanr(x2[boot_indices], y[boot_indices])

        delta = rho1 - rho2
        deltas.append(delta)

    # Compute CI
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    ci_lower = np.percentile(deltas, lower_percentile)
    ci_upper = np.percentile(deltas, upper_percentile)

    return delta_observed, ci_lower, ci_upper


def analyze_K_structure(kv_states, layer_idx=6, num_pairs_per_head=500, future_horizon=64, seed=42, n_bootstrap=100):
    """
    Probe for p-adic structure in KEYS with 7 FIXES:

    1. Fixed quantization scale per (layer, head, precision) - shared across ALL sequences
    2. Per-head analysis - compute correlations separately for each head
    3. Use observed delta as point estimate, bootstrap only for CI
    4. Safety guard against infinite loops in pair generation
    5. Preserve all previous fixes (fixed pairs, Spearman, permutation null, etc.)
    6. Output per-head results in structured format
    7. Tested with small smoke test

    Args:
        kv_states: Extracted KV cache states
        layer_idx: Layer to analyze
        num_pairs_per_head: Number of pairs to sample PER HEAD
        future_horizon: Number of future queries for attention similarity
        seed: Random seed for pair manifest
        n_bootstrap: Number of bootstrap iterations for CI

    Returns:
        dict with per-head results
    """
    print(f"\n{'='*80}")
    print(f"Layer {layer_idx}: Per-Head Analysis with Fixed Shared Scales")
    print(f"{'='*80}")
    print(f"Pairs per head: {num_pairs_per_head}, Future horizon: {future_horizon}, Seed: {seed}")

    # Extract keys and attention for this layer
    keys = [sample[layer_idx] for sample in kv_states['keys']]
    attentions = [sample[layer_idx] for sample in kv_states['attention_weights']]

    num_heads = keys[0][0].shape[0]
    print(f"Number of heads: {num_heads}")

    # FIX 1: Compute quantization scales per (head, precision) - SHARED across all sequences
    print("\n[1/6] Computing shared quantization scales per head...")
    quantization_scales = {}

    for precision in [8, 12, 16]:
        for head_idx in range(num_heads):
            # Collect ALL key vectors for this head across ALL sequences
            all_keys_for_head = []
            for seq_id in range(len(keys)):
                key_tensor = keys[seq_id][0]  # [heads, seq, dim]
                K_head = key_tensor[head_idx]  # [seq, dim]
                all_keys_for_head.append(K_head)

            # Concatenate all keys for this head
            all_keys_concat = torch.cat(all_keys_for_head, dim=0)  # [total_tokens, dim]

            # Compute ONE shared scale for this head
            scale = compute_quantization_scale(all_keys_concat, precision)
            quantization_scales[(head_idx, precision)] = scale

    print(f"  Computed {len(quantization_scales)} scales (shared across sequences)")

    # FIX 4: Generate pair manifest per head with safety guard
    print("\n[2/6] Generating fixed pair manifest per head...")
    pair_manifest_per_head = generate_pair_manifest(
        keys, attentions, num_pairs_per_head, future_horizon, seed
    )
    for h in range(num_heads):
        print(f"  Head {h}: {len(pair_manifest_per_head[h])} pairs")

    # FIX 2: Analyze EACH HEAD separately
    results = {'per_head': []}

    for head_idx in range(num_heads):
        print(f"\n{'='*80}")
        print(f"Analyzing Head {head_idx}")
        print(f"{'='*80}")

        pair_list = pair_manifest_per_head[head_idx]

        # Primary analysis: p=2, 16-bit
        print(f"\n[3/6] Computing statistics for head {head_idx} (p=2, 16-bit)...")
        stats_real = compute_all_statistics_fixed_pairs(
            keys, attentions, head_idx, pair_list, prime=2, precision=16,
            future_horizon=future_horizon, quantization_scales=quantization_scales,
            use_permuted=False
        )

        # Permutation null control
        print(f"[4/6] Computing permutation null for head {head_idx}...")
        stats_perm = compute_all_statistics_fixed_pairs(
            keys, attentions, head_idx, pair_list, prime=2, precision=16,
            future_horizon=future_horizon, quantization_scales=quantization_scales,
            use_permuted=True
        )

        # FIX 3: Compute observed correlations and bootstrap CI
        print(f"[5/6] Computing correlations and bootstrap CI for head {head_idx}...")
        att = stats_real['attention']

        rho_S2_real, _ = spearmanr(stats_real['S2'], att)
        rho_cos_real, _ = spearmanr(stats_real['cosine'], att)
        rho_S2_perm, _ = spearmanr(stats_perm['S2'], att)

        # FIX 3: Observed delta with bootstrap CI
        delta_rho, ci_lower, ci_upper = bootstrap_correlation_difference(
            stats_real['S2'], stats_real['cosine'], att, pair_list, n_bootstrap=n_bootstrap
        )

        # Verify: delta_rho should equal rho_S2 - rho_cos
        assert abs(delta_rho - (rho_S2_real - rho_cos_real)) < 1e-6, \
            f"Delta mismatch: {delta_rho} != {rho_S2_real - rho_cos_real}"

        print(f"\n  Results for Head {head_idx}:")
        print(f"    rho_S2:             {rho_S2_real:.4f}")
        print(f"    rho_cosine:         {rho_cos_real:.4f}")
        print(f"    delta_rho:          {delta_rho:.4f}")
        print(f"    95% bootstrap CI:   [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    rho_S2_permuted:    {rho_S2_perm:.4f}")
        print(f"    real - permuted:    {rho_S2_real - rho_S2_perm:.4f}")

        # FIX 6: Per-head output structure
        head_result = {
            'head': head_idx,
            'num_pairs': len(pair_list),
            'rho_S2': float(rho_S2_real),
            'rho_cosine': float(rho_cos_real),
            'delta_rho': float(delta_rho),
            'delta_ci_95': [float(ci_lower), float(ci_upper)],
            'rho_S2_permuted': float(rho_S2_perm),
            'real_minus_permuted': float(rho_S2_real - rho_S2_perm),
        }

        results['per_head'].append(head_result)

        # Optional: Test other primes for this head
        print(f"[6/6] Testing other primes for head {head_idx}...")
        head_result['other_primes'] = {}
        for prime in [3, 5]:
            stats_p = compute_all_statistics_fixed_pairs(
                keys, attentions, head_idx, pair_list, prime=prime, precision=16,
                future_horizon=future_horizon, quantization_scales=quantization_scales,
                use_permuted=False
            )
            rho_S2_p, _ = spearmanr(stats_p['S2'], att)
            head_result['other_primes'][f'p{prime}'] = {'rho_S2': float(rho_S2_p)}
            print(f"    p={prime}: rho_S2={rho_S2_p:.4f}")

        # Optional: Test other precisions
        head_result['other_precisions'] = {}
        for prec in [8, 12]:
            stats_prec = compute_all_statistics_fixed_pairs(
                keys, attentions, head_idx, pair_list, prime=2, precision=prec,
                future_horizon=future_horizon, quantization_scales=quantization_scales,
                use_permuted=False
            )
            rho_S2_prec, _ = spearmanr(stats_prec['S2'], att)
            head_result['other_precisions'][f'{prec}bit'] = {'rho_S2': float(rho_S2_prec)}
            print(f"    {prec}-bit: rho_S2={rho_S2_prec:.4f}")

    # Aggregate summary across heads
    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY ACROSS HEADS")
    print(f"{'='*80}")

    all_rho_S2 = [h['rho_S2'] for h in results['per_head']]
    all_delta = [h['delta_rho'] for h in results['per_head']]
    all_real_minus_perm = [h['real_minus_permuted'] for h in results['per_head']]

    results['aggregate'] = {
        'mean_rho_S2': float(np.mean(all_rho_S2)),
        'std_rho_S2': float(np.std(all_rho_S2)),
        'mean_delta_rho': float(np.mean(all_delta)),
        'std_delta_rho': float(np.std(all_delta)),
        'mean_real_minus_permuted': float(np.mean(all_real_minus_perm)),
    }

    print(f"  Mean rho_S2:             {results['aggregate']['mean_rho_S2']:.4f} ± {results['aggregate']['std_rho_S2']:.4f}")
    print(f"  Mean delta_rho:          {results['aggregate']['mean_delta_rho']:.4f} ± {results['aggregate']['std_delta_rho']:.4f}")
    print(f"  Mean real-permuted:      {results['aggregate']['mean_real_minus_permuted']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Probe for p-adic structure in KV cache")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset shorthand (wikitext/code/math)")
    parser.add_argument("--dataset-path", type=str, help="Full dataset path (optional, overrides --dataset)")
    parser.add_argument("--dataset-split", type=str, default="test", help="Dataset split")
    parser.add_argument("--text-field", type=str, default="text", help="Text field name")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of text samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
    parser.add_argument("--num-pairs", type=int, default=500, help="Number of pairs to sample per head")
    parser.add_argument("--future-horizon", type=int, default=64, help="Number of future queries for attention similarity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair manifest")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="Number of bootstrap iterations for CI")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--cache-dir", type=str, default="/workspace/padic-transformers/checkpoints/pretrained")

    args = parser.parse_args()

    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
        dataset_name = args.dataset_path.split('/')[-1]
    else:
        # Shortcuts for common datasets
        dataset_shortcuts = {
            'wikitext': ('Salesforce/wikitext', 'wikitext-103-raw-v1', 'text'),
            'code': ('bigcode/the-stack-dedup', 'data/python', 'content'),
            'math': ('hendrycks/competition_math', 'train', 'problem'),
        }
        if args.dataset in dataset_shortcuts:
            dataset_path, config, field = dataset_shortcuts[args.dataset]
            args.text_field = field
            dataset_name = args.dataset
        else:
            dataset_path = args.dataset
            dataset_name = args.dataset

    print("="*80)
    print("P-ADIC STRUCTURE PROBE FOR KEYS (K)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Analyzing layer: {args.layer}")
    print(f"Estimated tokens: {args.num_samples * args.max_length:,}")
    print("\nScope:")
    print("  - Probing KEY structure only (not VALUES)")
    print("  - Values extracted but not analyzed")
    print("  - Separate V probe needed for complete picture")
    print("\nFixes applied:")
    print("  ✓ Fixed quantization scale per layer/head")
    print("  ✓ Spearman for all, bootstrap delta directly")
    print("  ✓ Same pair manifest everywhere")
    print("  ✓ Fixed future-attention horizon")
    print("  ✓ Token-permutation null control")
    print("\nControls:")
    print("  ✓ Permutation null (token shuffling)")
    print("  ✓ Multiple primes (p=2, 3, 5)")
    print("  ✓ Multiple precisions (8, 12, 16 bit)")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{args.model}",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",  # Required for output_attentions to work
    )
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{args.model}", cache_dir=args.cache_dir)
    print("✓ Model loaded")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}...")
    try:
        if args.dataset == 'wikitext':
            dataset = load_dataset(dataset_path, 'wikitext-103-raw-v1', split=args.dataset_split)
        elif args.dataset == 'code':
            dataset = load_dataset(dataset_path, data_dir='data/python', split=args.dataset_split, streaming=True)
            # Take first N samples from streaming dataset
            dataset = list(dataset.take(args.num_samples))
        elif args.dataset == 'math':
            dataset = load_dataset(dataset_path, split=args.dataset_split)
        else:
            dataset = load_dataset(dataset_path, split=args.dataset_split)

        texts = [ex[args.text_field] for ex in dataset if ex[args.text_field].strip()][:args.num_samples]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying simple load...")
        dataset = load_dataset(dataset_path, split=args.dataset_split)
        texts = [ex[args.text_field] for ex in dataset if ex[args.text_field].strip()][:args.num_samples]

    # Extract KV states
    print(f"\nExtracting KV states from {len(texts)} samples...")
    kv_states = extract_kv_states(model, tokenizer, texts, max_length=args.max_length)

    # Analyze K structure (values probed separately)
    print(f"\n{'='*80}")
    print("PROBING KEY (K) STRUCTURE ONLY")
    print("NOTE: Values (V) should be probed separately")
    print(f"{'='*80}")

    results = analyze_K_structure(
        kv_states,
        layer_idx=args.layer,
        num_pairs_per_head=args.num_pairs,
        future_horizon=args.future_horizon,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'layer_analyzed': args.layer,
        'correlations': results,
    }

    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
