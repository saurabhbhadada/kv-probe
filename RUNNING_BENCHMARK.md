# Running the Geometry Benchmark

## 1. Run Unit Tests

```bash
pytest tests/test_geometries.py -v
```

## 2. Run Smoke Test (3 samples)

```bash
python scripts/run_geometry_benchmark.py \
  --model_name EleutherAI/pythia-410m \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --num_samples 3 \
  --layers 0 5 10 \
  --heads 0 1 2 3 \
  --output_dir results/smoke_test
```

## 3. Run Full Benchmark

```bash
python scripts/run_geometry_benchmark.py \
  --model_name EleutherAI/pythia-410m \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --num_samples 100 \
  --layers 0 5 10 15 19 \
  --heads all \
  --output_dir results/full_benchmark \
  --min_future_queries 32
```

## 4. View Results

Results are saved to the output directory as CSV files with aggregated statistics across samples and heads.
