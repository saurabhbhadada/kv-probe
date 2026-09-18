# Geometry Benchmark Implementation Summary

## Overview

Extended the KV-probe codebase with a comprehensive geometry benchmark to measure functional redundancy between KV-cache entries using multiple distance metrics.

**Research Question:** Which geometry best predicts whether two KV-cache entries are functionally redundant?

---

## Files Created/Modified

### New Files Created (11 total):

**Core Geometries:**
1. `src/geometries/base.py` - Abstract `GeometryMetric` base class
2. `src/geometries/euclidean.py` - Euclidean, Cosine, Manhattan metrics
3. `src/geometries/mahalanobis.py` - Query-induced Mahalanobis distance
4. `src/geometries/spherical_radial.py` - Spherical × radial geometry
5. `src/geometries/exponential.py` - Exponential-response geometry
6. `src/geometries/fisher.py` - Fisher/softmax-sensitive geometry

**Ground Truth:**
7. `src/ground_truth/__init__.py` - Ground truth module
8. `src/ground_truth/redundancy.py` - Deletion damage, merge damage, attention similarity

**Benchmark Runner:**
9. `scripts/run_geometry_benchmark.py` - Main benchmark script

**Tests:**
10. `tests/test_geometries.py` - Synthetic validation tests

**Modified Files:**
11. `src/geometries/__init__.py` - Updated exports

---

## Implemented Metrics

### 1. Query-Induced Mahalanobis Distance

**Formula:**
```
d_Q(i,j) = ||Q delta_k||_2 / sqrt(T)
where delta_k = k_i - k_j
      Q = [num_queries, d_model] query matrix
```

**Two Modes:**
- `oracle`: Uses all future queries (ground truth)
- `causal`: Uses only past queries (realistic)

**Key Insight:** Measures distance in the query-weighted subspace. Keys differing only in directions orthogonal to queries have near-zero distance.

**Implementation:** `QueryMahalanobisOracleMetric`, `QueryMahalanobisCausalMetric`

**Approximations:** None. Uses exact empirical form.

---

### 2. Spherical × Radial Geometry

**Formula:**
```
d_SR^2 = acos(clamp(u_i^T u_j, -1, 1))^2 + lambda * (log(r_i) - log(r_j))^2

where:
  r_i = ||k_i||
  u_i = k_i / ||k_i||
  lambda = radial weight (configurable)
```

**Variants:**
- `SphericalRadialMetric` - Combined angular + radial
- `SphericalOnlyMetric` - Angular distance only
- `RadialOnlyMetric` - Magnitude distance only (log or linear)

**Key Insight:** Separates direction from magnitude. Useful for RoPE models where position affects magnitude differently than semantics.

**Implementation:** Clamps dot product to [-1, 1] before acos for numerical stability.

**Approximations:** None.

---

### 3. Exponential-Response Geometry

**Formula:**
```
d_exp(i,j) = sqrt(mean_q [(exp(q^T k_i / T) - exp(q^T k_j / T))^2])

Log-space variant:
d_exp_log(i,j) = sqrt(mean_q [(q^T k_i / T - q^T k_j / T)^2])
```

**Implementation:** Two variants:
- `ExponentialResponseMetric` - Works in exp space with clipping
- `ExponentialResponseLogMetric` - Log-space (more stable)

**Numerical Stability:**
- Clips logits to [-20, 20] to prevent overflow
- Log-space variant avoids exp entirely
- Adds epsilon (1e-8) before sqrt

**Approximations:** Logit clipping is an approximation when logits exceed ±20.

---

### 4. Fisher / Softmax-Sensitive Geometry

**Perturbation Definition:** Replacing key k_i with k_j.

**Formula (efficient closed form):**
```
d_F^2(i,j) = mean_q [(logit_j - logit_i)^2 * a_i * (1 - a_i)]

where:
  logit_i = q^T k_i / sqrt(d)
  a_i = softmax(logits)[i]
```

**Variants:**
- `FisherSwapMetric` - Swap k_i with k_j
- `FisherMergeMetric` - Replace both with (k_i + k_j)/2
- `FisherSymmetricMetric` - Average sensitivity from both positions

**Key Insight:** Measures how much attention distribution changes. High attention weights (a_i ≈ 1) → undefined, but a_i(1-a_i) → 0, so impact is small.

**Mathematical Simplification:** Avoids constructing full Jacobian matrix J = diag(a) - aa^T by using:
```
||J * e_i * delta_z||^2 = (delta_z)^2 * a_i * (1 - a_i)
```

**Approximations:** First-order perturbation analysis (linear approximation).

---

## Ground Truth Metrics

### A. Future Attention-Pattern Similarity

**Reused from existing code:** Compares attention columns A[:,i] vs A[:,j] over future queries.

**Formula:**
```
similarity(i,j) = cosine(A[future_start:future_end, i], A[future_start:future_end, j])
```

---

### B. Deletion Damage

**Formula (closed-form):**
```
damage_j = a_j / (1 - a_j) * ||v_j - o||

where:
  o = sum_i a_i v_i (original output)
```

**Numerical Protection:** Clamps a_j to [0, 0.9999] to prevent division by zero when a_j → 1.

**Implementation:** Computes per-query damage, then averages.

---

### C. Merge Damage

**Perturbation:** Replace positions i and j with merged token.

**Merge Strategy:** Simple average: k_merged = (k_i + k_j)/2, v_merged = (v_i + v_j)/2

**Formula:**
```
damage(i,j) = ||output_original - output_merged||
```

**Implementation:** Recomputes full attention with modified KV cache.

**Performance Note:** This is O(seq_len) per pair and may be slow for large sequences.

---

## Evaluation Metrics

For each (geometry, ground_truth) pair, computes:

1. **Pearson correlation** - Linear relationship
2. **Spearman correlation** - Rank-order relationship
3. **AUROC** - Classification performance (planned, not yet implemented)
4. **AUPRC** - Precision-recall curve (planned, not yet implemented)

---

## Output Schema

### Detailed CSV (per-pair results):

```
dataset, sample_id, layer, head, token_i, token_j,
euclidean, cosine_distance, cosine_similarity,
mahalanobis_oracle, spherical_radial_lambda1.0, spherical_only, radial_only_log,
exp_response_log_L2, fisher_symmetric,
future_attention_similarity, deletion_damage_i, deletion_damage_j, merge_damage
```

### Aggregate Results (console output):

```
Ground truth: future_attention_similarity
----------------------------------------
  euclidean                      | Pearson: +0.345 | Spearman: +0.312
  cosine_similarity              | Pearson: +0.421 | Spearman: +0.398
  mahalanobis_oracle             | Pearson: +0.512 | Spearman: +0.487
  ...
```

---

## Commands

### Quick Smoke Test (1 sample, 1 head, 100 pairs):

```bash
python scripts/run_geometry_benchmark.py \
    --model pythia-1b \
    --dataset wikitext \
    --num-samples 1 \
    --layer 6 \
    --head 0 \
    --num-pairs 100 \
    --output results/smoke_test.csv
```

**Expected runtime:** ~2-3 minutes on GPU

---

### Full Experiment (100 samples, all heads, 500 pairs):

```bash
python scripts/run_geometry_benchmark.py \
    --model pythia-1b \
    --dataset wikitext \
    --num-samples 100 \
    --layer 6 \
    --num-pairs 500 \
    --future-horizon 64 \
    --pair-strategy random \
    --output results/geometry_benchmark_full.csv
```

**Expected runtime:** ~30-60 minutes on GPU

---

### Run Tests:

```bash
cd /path/to/kv-probe
pytest tests/test_geometries.py -v
```

---

## Performance Bottlenecks

1. **Merge damage computation** - O(seq_len * num_pairs * num_queries)
   - Most expensive ground truth metric
   - Consider sampling fewer pairs or shorter sequences

2. **Fisher metric** - Requires computing full softmax for all queries
   - Still efficient due to vectorization

3. **Mahalanobis causal mode** - Per-pair loop over queries
   - Slower than oracle mode
   - Consider batching if possible

4. **Query extraction** - Currently approximates queries as keys
   - For true queries, would need model hooks (more complex)

---

## Assumptions & Limitations

1. **Query Approximation:** Current implementation uses keys as proxy for queries. For exact queries, would need to hook into model's Q projection.

2. **Single Temperature:** All attention-based metrics use sqrt(d) temperature. Real models may use different values.

3. **No RoPE Handling:** Metrics compute on post-RoPE keys. Pre-RoPE analysis would require model hooks.

4. **Pair Sampling:** Currently random or nearby. Could add attention-weighted sampling, high-similarity pairs, etc.

5. **Ground Truth Assumes Independence:** Deletion/merge damage assumes other tokens' attention renormalizes independently. In practice, there may be second-order effects.

6. **Memory:** Stores all pairs in memory. For very large experiments, may need batching.

---

## Next Steps

1. **Add AUROC/AUPRC:** Define "safe-to-merge" threshold (e.g., damage <= 10th percentile)

2. **Add Pre-RoPE Analysis:** Hook model to extract pre-RoPE keys

3. **Multi-Model Comparison:** Test on Llama, Qwen, GPT-NeoX

4. **Layer Sweep:** Compare geometry effectiveness across layers

5. **Visualization:** Heatmaps of correlation per layer/head

6. **Compression Experiment:** Use best geometry to guide actual KV cache pruning

---

## File Tree

```
kv-probe/
├── src/
│   ├── geometries/
│   │   ├── base.py                 # Abstract GeometryMetric
│   │   ├── euclidean.py            # Euclidean, Cosine, Manhattan
│   │   ├── mahalanobis.py          # Query-induced Mahalanobis
│   │   ├── spherical_radial.py     # Spherical × radial
│   │   ├── exponential.py          # Exponential-response
│   │   ├── fisher.py               # Fisher / softmax-sensitive
│   │   ├── padic.py                # Original p-adic (unchanged)
│   │   └── __init__.py             # Updated exports
│   └── ground_truth/
│       ├── redundancy.py           # Deletion/merge damage
│       └── __init__.py
├── scripts/
│   ├── run_geometry_benchmark.py   # Main benchmark runner
│   ├── probe_kv_structure.py       # Original probe (unchanged)
│   └── ...
├── tests/
│   ├── test_geometries.py          # New geometry tests
│   └── test_padic_ops.py           # Original tests (unchanged)
└── GEOMETRY_BENCHMARK_SUMMARY.md   # This file
```

---

## Research Discipline

**Neutral Implementation:** All metrics treated equally. No optimization bias toward any geometry.

**Falsifiable:** Every metric can fail. Poor performance is a valid result.

**Extensible:** Easy to add new geometries by inheriting from `GeometryMetric`.

**Reproducible:** Fixed seeds, deterministic pair sampling, version-controlled code.

---

## Contact / Issues

For bugs or questions, see existing issue tracker or documentation.
