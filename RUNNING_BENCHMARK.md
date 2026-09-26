# Running the Geometry Benchmark

All commands run inside Docker. Make sure you have built the Docker image first:
```bash
make build
```

## 1. Run Unit Tests

```bash
make test-geometry
```

## 2. Run Smoke Test (3 samples)

Monitor GPU memory while running:
```bash
# Terminal 1: Run benchmark
make benchmark-smoke

# Terminal 2: Monitor GPU
make shell
# Inside container:
watch -n 1 nvidia-smi
```

## 3. Run Full Benchmark (Single Head)

Default (100 samples, layer 10, head 0):
```bash
make benchmark-full
```

Custom parameters:
```bash
make benchmark-full LAYER=5 HEAD=2 SAMPLES=200
```

## 4. Benchmark All Heads in One Layer

Process all heads in a single run (more efficient than looping):
```bash
# Default: layer 5, 200 samples
make benchmark-layer

# Custom layer and samples
make benchmark-layer LAYER=10 SAMPLES=100
```

Output: `results/benchmark_L{LAYER}_all_heads.csv` with rows for all heads (0-15 for pythia-410m)

## 5. Run Multiple Individual Heads (if needed)

```bash
for head in 0 1 2 3; do
  make benchmark-full LAYER=10 HEAD=$head SAMPLES=100
done
```

## 6. Direct Python Command (if needed)

If you need more control, use `make exec`:
```bash
make exec CMD="python3 scripts/run_geometry_benchmark.py \
  --model pythia-410m \
  --dataset wikitext \
  --num-samples 100 \
  --layer 10 \
  --head 0 \
  --num-pairs 500 \
  --output results/custom.csv"
```

## 7. View Results

```bash
# From host
head -20 results/smoke_test.csv

# Or inside container
make shell
head -20 results/benchmark_L10_H0.csv
```

## Output Columns

Results CSV contains:
- **Geometry distances**: euclidean, cosine_distance, fisher_symmetric, mahalanobis_oracle, etc.
- **Ground truth**: future_attention_similarity, merge_damage, deletion_damage
- **Correlations**: Spearman, Pearson, AUROC, AUPRC (printed to console)
