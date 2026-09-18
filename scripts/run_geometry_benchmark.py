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
    QueryMahalanobisOracleMetric,
    QueryMahalanobisCausalMetric,
    SphericalRadialMetric,
    SphericalOnlyMetric,
    RadialOnlyMetric,
    ExponentialResponseMetric,
    ExponentialResponseLogMetric,
    FisherSwapMetric,
    FisherSymmetricMetric,
)

# Import ground truth
from src.ground_truth import (
    compute_deletion_damage,
    compute_merge_damage,
    compute_future_attention_similarity,
)


def extract_kv_and_queries(model, tokenizer, texts, max_length=512):
    """
    Extract KV cache states, attention weights, AND queries from real model inference.

    Returns:
        dict with 'keys', 'values', 'queries', 'attention_weights'
    """
    model.eval()
    device = next(model.parameters()).device

    all_keys = []
    all_values = []
    all_queries = []
    all_attention = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting KV states and queries"):
            if not text or not text.strip():
                continue

            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

            if inputs['input_ids'].shape[1] == 0:
                continue

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, output_attentions=True, use_cache=True, output_hidden_states=True)

            past_kv = outputs.past_key_values
            if past_kv is None or len(past_kv) == 0:
                continue

            # Extract keys, values
            layer_keys = [kv[0].cpu() for kv in past_kv]
            layer_values = [kv[1].cpu() for kv in past_kv]
            layer_attentions = [attn.cpu() for attn in outputs.attentions]

            # Extract queries: Q = hidden_states @ W_q
            # For simplicity, we'll approximate queries from attention patterns
            # Alternative: hook into model to get actual Q before matmul
            # For now, use keys as proxy for queries (same sequence)
            layer_queries = layer_keys  # Approximation

            all_keys.append(layer_keys)
            all_values.append(layer_values)
            all_queries.append(layer_queries)
            all_attention.append(layer_attentions)

    return {
        'keys': all_keys,
        'values': all_values,
        'queries': all_queries,
        'attention_weights': all_attention,
    }


def sample_pairs(seq_len, num_pairs, strategy='random', seed=42):
    """
    Sample pairs of positions to evaluate.

    Strategies:
    - random: uniform random pairs
    - nearby: pairs within small distance
    - mixed: combination
    """
    np.random.seed(seed)

    if strategy == 'random':
        pairs = []
        for _ in range(num_pairs):
            i, j = np.random.choice(seq_len, size=2, replace=False)
            pairs.append((i, j))
        return np.array(pairs)

    elif strategy == 'nearby':
        # Sample pairs within distance 10
        pairs = []
        max_dist = min(10, seq_len // 4)
        for _ in range(num_pairs * 2):  # oversample
            i = np.random.randint(0, seq_len - max_dist)
            j = i + np.random.randint(1, max_dist + 1)
            if j < seq_len:
                pairs.append((i, j))
            if len(pairs) >= num_pairs:
                break
        return np.array(pairs[:num_pairs])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_benchmark_single_head(
    keys, values, queries, attentions,
    head_idx, layer_idx, dataset_name, sample_id,
    num_pairs=500, future_horizon=64, pair_strategy='random', seed=42
):
    """
    Run benchmark for a single (layer, head, sample).

    Returns:
        DataFrame with one row per pair, columns for all metrics + ground truth
    """
    # keys: [batch=1, heads, seq, dim]
    # Extract for this head
    K = keys[0, head_idx, :, :]  # [seq, dim]
    V = values[0, head_idx, :, :]  # [seq, dim]
    Q = queries[0, head_idx, :, :]  # [seq, dim]
    A = attentions[0, head_idx, :, :]  # [seq, seq]

    seq_len, d_model = K.shape
    device = K.device

    # Sample pairs
    if seq_len < 2:
        return None

    pairs_np = sample_pairs(seq_len, num_pairs=min(num_pairs, seq_len * (seq_len - 1) // 2),
                           strategy=pair_strategy, seed=seed)
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

    # Define geometries to test
    geometries = [
        EuclideanMetric(),
        CosineDistanceMetric(),
        CosineSimilarityMetric(),
        QueryMahalanobisOracleMetric(),
        SphericalRadialMetric({'lambda_radial': 1.0}),
        SphericalOnlyMetric(),
        RadialOnlyMetric(),
        ExponentialResponseLogMetric({'p_norm': 2}),
        FisherSymmetricMetric(),
    ]

    # Compute all geometry metrics
    for geom in geometries:
        try:
            # Precompute if needed
            precomputed = geom.precompute(K, V, queries=Q)

            # Compute pairwise distances
            distances = geom.compute_pairwise(
                K, V, pairs, queries=Q, **precomputed
            )

            results[geom.name] = distances.cpu().numpy().tolist()
        except Exception as e:
            print(f"Warning: {geom.name} failed: {e}")
            results[geom.name] = [np.nan] * num_actual_pairs

    # Compute ground truth: future attention similarity
    try:
        future_sims = []
        for pair_idx in range(num_actual_pairs):
            i_pos = pairs_np[pair_idx, 0]
            j_pos = pairs_np[pair_idx, 1]
            max_pos = max(i_pos, j_pos)
            future_start = max_pos + 1

            if future_start + future_horizon <= seq_len:
                attn_col_i = A[future_start:future_start+future_horizon, i_pos]
                attn_col_j = A[future_start:future_start+future_horizon, j_pos]

                sim = torch.nn.functional.cosine_similarity(
                    attn_col_i.unsqueeze(0),
                    attn_col_j.unsqueeze(0),
                    dim=1
                ).item()
                future_sims.append(sim)
            else:
                future_sims.append(np.nan)

        results['future_attention_similarity'] = future_sims
    except Exception as e:
        print(f"Warning: future_attention_similarity failed: {e}")
        results['future_attention_similarity'] = [np.nan] * num_actual_pairs

    # Compute ground truth: deletion damage (averaged over all queries)
    try:
        positions = pairs[:, 0]  # damage for position i
        damage_tensor = compute_deletion_damage(K, V, Q, positions)  # [num_queries, num_pairs]
        damage_avg = damage_tensor.mean(dim=0).cpu().numpy()  # [num_pairs]
        results['deletion_damage_i'] = damage_avg.tolist()

        # Also for position j
        positions_j = pairs[:, 1]
        damage_tensor_j = compute_deletion_damage(K, V, Q, positions_j)
        damage_avg_j = damage_tensor_j.mean(dim=0).cpu().numpy()
        results['deletion_damage_j'] = damage_avg_j.tolist()
    except Exception as e:
        print(f"Warning: deletion_damage failed: {e}")
        results['deletion_damage_i'] = [np.nan] * num_actual_pairs
        results['deletion_damage_j'] = [np.nan] * num_actual_pairs

    # Compute ground truth: merge damage
    try:
        merge_damage_tensor = compute_merge_damage(K, V, Q, pairs)  # [num_queries, num_pairs]
        merge_damage_avg = merge_damage_tensor.mean(dim=0).cpu().numpy()  # [num_pairs]
        results['merge_damage'] = merge_damage_avg.tolist()
    except Exception as e:
        print(f"Warning: merge_damage failed: {e}")
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
        torch_dtype=torch.float16,
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

    # Extract KV states
    print(f"\nExtracting KV states...")
    kv_data = extract_kv_and_queries(model, tokenizer, texts, max_length=args.max_length)

    # Run benchmark
    print(f"\nRunning geometry benchmark...")
    all_results = []

    layer_idx = args.layer
    num_heads = kv_data['keys'][0][layer_idx][0].shape[0]
    heads_to_test = [args.head] if args.head is not None else range(num_heads)

    for sample_id, (keys_sample, values_sample, queries_sample, attns_sample) in enumerate(
        zip(kv_data['keys'], kv_data['values'], kv_data['queries'], kv_data['attention_weights'])
    ):
        keys_layer = keys_sample[layer_idx]
        values_layer = values_sample[layer_idx]
        queries_layer = queries_sample[layer_idx]
        attns_layer = attns_sample[layer_idx]

        for head_idx in heads_to_test:
            df = run_benchmark_single_head(
                keys_layer, values_layer, queries_layer, attns_layer,
                head_idx, layer_idx, args.dataset, sample_id,
                num_pairs=args.num_pairs,
                future_horizon=args.future_horizon,
                pair_strategy=args.pair_strategy,
                seed=args.seed + sample_id * 100 + head_idx,
            )

            if df is not None:
                all_results.append(df)

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

    for gt in ground_truths:
        if gt not in results_df.columns:
            continue

        print(f"\nGround truth: {gt}")
        print("-" * 40)

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
                print(f"  {geom_col:30s} | Pearson: {pearson_r:+.3f} | Spearman: {spearman_r:+.3f}")
            except:
                pass

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
