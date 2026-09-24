# Running the Geometry Benchmark

## 1. Run Unit Tests

```bash
pytest tests/test_geometries.py -v
```

## 2. Run Smoke Test (3 samples)

```bash
python scripts/run_geometry_benchmark.py \
  --model pythia-410m \
  --dataset wikitext \
  --num-samples 3 \
  --layer 5 \
  --head 0 \
  --num-pairs 100 \
  --future-horizon 64 \
  --min-future-queries 32 \
  --output results/smoke_test.csv
```

## 3. Run Full Benchmark (Single Layer/Head)

```bash
python scripts/run_geometry_benchmark.py \
  --model pythia-410m \
  --dataset wikitext \
  --num-samples 100 \
  --layer 10 \
  --head 0 \
  --num-pairs 500 \
  --future-horizon 64 \
  --min-future-queries 32 \
  --output results/full_benchmark_L10_H0.csv
```

## 4. Run Across Multiple Heads

To benchmark multiple heads, run separate commands for each head or modify the script to loop over heads:

```bash
for head in 0 1 2 3; do
  python scripts/run_geometry_benchmark.py \
    --model pythia-410m \
    --dataset wikitext \
    --num-samples 100 \
    --layer 10 \
    --head $head \
    --num-pairs 500 \
    --output results/benchmark_L10_H${head}.csv
done
```

## 5. Model Names

The script automatically prepends `EleutherAI/` to model names, so use:
- `pythia-410m` (not `EleutherAI/pythia-410m`)
- `pythia-1b`
- `pythia-2.8b`

## 6. CLI Flags Reference

- `--model`: Model name (e.g., `pythia-410m`)
- `--dataset`: Dataset name (e.g., `wikitext`)
- `--num-samples`: Number of text samples to process
- `--layer`: Layer index to analyze
- `--head`: Head index to analyze (or None for all heads)
- `--num-pairs`: Number of KV pairs to evaluate per head
- `--future-horizon`: Number of future queries for ground truth
- `--min-future-queries`: Minimum future queries required (default: 32)
- `--output`: Output CSV file path

## 7. View Results

Results are saved as CSV files with columns:
- Geometry distances for each metric
- Ground truth redundancy scores
- Correlation statistics (Spearman, Pearson, AUROC, AUPRC)
