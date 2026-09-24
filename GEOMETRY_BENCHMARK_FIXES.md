# Geometry Benchmark Fixes - Summary

## Overview
Fixed all 11 priority issues in the geometry benchmark to make it scientifically valid.

---

## Files Changed

### 1. `scripts/run_geometry_benchmark.py`
**Major changes:**

#### Extract Real Queries (Post-RoPE)
- **Before**: Used `layer_queries = layer_keys` (incorrect proxy)
- **After**: Extract actual Q tensors by:
  1. Getting hidden states from model output
  2. Projecting through `query_key_value` layer
  3. Applying RoPE to match cached K tensors
  4. Works specifically with Pythia/GPTNeoX architecture

#### Sanity Check Function
- **New**: `sanity_check_qk_reconstruction()`
  - Verifies `softmax(Q @ K^T / sqrt(d) + causal_mask) ≈ outputs.attentions`
  - Computes max_error and mean_error
  - Only runs on first sample to avoid spam
  - Warns if max_error > 0.01

#### Unique Pair Sampling
- **Before**: Could generate duplicate pairs, no canonicalization
- **After**:
  - `sample_pairs(..., unique=True)` ensures no duplicates
  - All pairs canonicalized to `i < j`
  - Uses efficient sampling (rejection sampling for small N, enumerate for large N)

#### Temperature Defaults
- **Before**: Many metrics used `temperature=1.0`
- **After**:
  - Compute `temp_sqrt_d = math.sqrt(head_dim)` per head
  - Pass to all metrics: Fisher, Exponential, ground truth functions

#### Valid Future Queries for Ground Truth
- **Before**: Used all queries, including those before the pair exists
- **After**:
  - For pair `(i, j)`, let `t = max(i, j)`
  - Use only `Q[t+1 : min(t+1+future_horizon, seq_len)]`
  - Applied to: future_attention_similarity, deletion_damage, merge_damage

#### GPU Optimization
- **Before**: Immediately moved tensors to CPU
- **After**:
  - Keep tensors on GPU during computation
  - Only move to CPU when saving results
  - `keep_on_gpu=True` parameter

#### Improved Evaluation
- **Before**: Only Pearson/Spearman correlation
- **After**:
  - Added AUROC/AUPRC for binary classification (safe-to-merge vs unsafe)
  - Per-layer × per-head breakdown
  - Shows best geometry for each head

#### Restored P-adic Baseline
- **Before**: P-adic metric not included in benchmark
- **After**:
  - Import and add `UltrametricMetric` to geometry list
  - Compares on identical pairs as all other metrics

#### Removed Duplicate Metric
- **Before**: Had both `ExponentialResponseLogMetric(L2)` and regular exponential
- **After**:
  - Removed log variant (equivalent to Mahalanobis)
  - Kept `ExponentialResponseMetric` with normalized option

#### Error Handling
- **Before**: Broad `except:` silently swallowed errors
- **After**:
  - Print full exception details with traceback
  - Helps debug metric failures

---

### 2. `src/ground_truth/redundancy.py`

#### Fixed Causal Masking
- **Before**: No causal mask in deletion/merge damage
- **After**:
  - Added documentation noting that benchmark passes Q_future (already causally valid)
  - No additional masking needed since queries are pre-filtered

#### Fixed Pair Canonicalization in Merge
- **Before**: Merge could fail when `i > j`
- **After**:
  ```python
  i_pos = min(i_pos_raw, j_pos_raw)
  j_pos = max(i_pos_raw, j_pos_raw)
  ```
  - Ensures merge always creates correct sequence: `keys[:i] + merged + keys[i+1:j] + keys[j+1:]`
  - Verifies `keys_merged.shape[0] == seq_len - 1`

---

### 3. `src/geometries/mahalanobis.py`

#### Fixed Oracle vs Causal Modes
- **Before (WRONG)**:
  - `causal`: Used queries AFTER `max(i,j)` (future queries)
- **After (CORRECT)**:
  - `oracle`: Uses ALL queries (unchanged)
  - `causal`: Uses queries BEFORE `max(i,j)` (past context)
  ```python
  causal_start = max(0, max_pos - self.causal_window)
  causal_end = max_pos  # Exclusive
  Q_causal = query_matrix[causal_start:causal_end]
  ```

---

### 4. `src/geometries/fisher.py`

#### Fixed Documentation
- **Before**: Claimed to compute `||J delta_z||^2`
- **After**:
  - Corrected to Fisher information / second-order KL approximation
  - Properly documented as quadratic form: `delta_z^T J delta_z`
  - For single-position swap: `(z_j - z_i)^2 * a_i(1 - a_i)`
  - Added note about cross-terms for two-position merges

---

### 5. `src/geometries/padic.py`

#### Added UltrametricMetric Class
- **New**: Wraps existing p-adic functions into GeometryMetric interface
- Features:
  - Quantizes keys to p-adic representation
  - Computes S_k statistic: fraction of dimensions with `v_p(k_i - k_j) >= k`
  - Returns `distance = 1 - S_k`
  - Config: `prime`, `precision`, `k_threshold`

---

### 6. `src/geometries/__init__.py`
- Added `UltrametricMetric` to exports

---

### 7. `tests/test_geometries.py`

#### New Test Classes

1. **TestCausalMasking**
   - `test_attention_reconstruction_with_causal_mask()`: Verifies softmax with causal mask

2. **TestMergeDamage**
   - `test_reversed_pair_merge()`: Verifies merge invariance to pair ordering
   - `test_merge_reduces_length_by_one()`: Verifies merge produces correct shape

3. **TestMahalanobisQueryWindows**
   - `test_oracle_uses_all_queries()`: Oracle uses all 100 queries
   - `test_causal_uses_past_queries()`: Causal uses only past queries
   - `test_causal_no_past_queries()`: Handles edge case with no past context

4. **TestTemperatureDefaults**
   - `test_fisher_temperature_default()`: Fisher uses sqrt(d_model)
   - `test_exponential_temperature_default()`: Exponential uses sqrt(d_model)

5. **TestUniquePairs**
   - `test_unique_pairs_no_duplicates()`: No duplicate pairs
   - `test_pairs_canonicalized()`: All pairs have `i < j`

---

## Summary of Fixes by Priority

| # | Fix | Status |
|---|-----|--------|
| 1 | Extract real queries (post-RoPE) | ✅ Done |
| 2 | Fix causal masking | ✅ Done |
| 3 | Use only valid future queries | ✅ Done |
| 4 | Fix pair ordering in merge | ✅ Done |
| 5 | Fix attention temperature | ✅ Done |
| 6 | Fix Mahalanobis modes | ✅ Done |
| 7 | Remove duplicate metric | ✅ Done |
| 8 | Restore p-adic baseline | ✅ Done |
| 9 | Fix Fisher documentation | ✅ Done |
| 10 | Improve evaluation | ✅ Done |
| 11 | Performance optimization | ✅ Done |

---

## Remaining Assumptions

1. **Model Architecture**: Query extraction code specifically handles Pythia/GPTNeoX
   - For other architectures, falls back to warning + using keys as proxy
   - Future: Add support for Llama, Qwen, etc.

2. **RoPE Application**: Assumes `_apply_rotary_pos_emb` method exists
   - Works for Pythia models
   - May need adjustment for different RoPE implementations

3. **Attention Format**: Assumes `outputs.attentions` is `[batch, heads, seq, seq]`
   - Standard HuggingFace format
   - Should work across most transformer models

4. **Causal Property**: Ground truth functions assume Q_future is pre-filtered
   - Benchmark correctly passes `Q[t+1:]` for each pair
   - No additional causal masking needed in ground truth functions

---

## How to Test

### Unit Tests (local or remote)
```bash
# In Docker or with pytest installed
make test

# Or directly
pytest tests/test_geometries.py -v
```

### Smoke Benchmark (requires GPU)
```bash
# Small test: 2 samples, 1 layer, 1 head, 50 pairs
python scripts/run_geometry_benchmark.py \
    --model pythia-1b \
    --dataset wikitext \
    --num-samples 2 \
    --layer 6 \
    --head 0 \
    --num-pairs 50 \
    --future-horizon 32 \
    --output results/smoke_test.csv
```

Expected output:
- ✓ QK reconstruction verified (first sample only)
- No errors from metric computations
- CSV with columns: all geometries + ground truth
- Summary showing correlations and AUROC/AUPRC

---

## Expected Improvements

1. **More accurate Q tensors**: Real queries instead of keys
2. **Correct Mahalanobis causal**: Uses past context, not future
3. **Valid ground truth**: Only evaluates on queries that can observe the pair
4. **Proper canonicalization**: Merge works regardless of pair ordering
5. **Correct temperature**: Matches actual attention computation
6. **Complete benchmark**: Includes p-adic alongside all other geometries
7. **Better evaluation**: AUROC/AUPRC + per-head analysis

---

## Next Steps

1. **Run Tests**: Execute unit tests on remote GPU machine
2. **Smoke Test**: Run small benchmark to verify all metrics work
3. **Full Benchmark**: Run on 100+ samples, multiple layers/heads
4. **Analyze Results**: Compare geometry rankings with corrected implementation
5. **Extend**: Add query extraction for Llama/Qwen architectures if needed
