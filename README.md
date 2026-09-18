# KV-Probe

Research project for probing transformer KV cache structure using different distance geometries.

**Goal**: Determine which geometric distance metrics best predict attention patterns in transformer key-value caches.

## Overview

This project systematically probes transformer KV caches to understand their geometric structure by testing correlation between different distance metrics and attention behavior.

**Currently Implemented:**
- P-adic (p=2, 3, 5) - Ultrametric distance based on prime factorization
- Euclidean (L2) - Standard distance metric baseline
- Cosine similarity - Angular distance baseline

## Quick Start

**→ See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed remote GPU setup**

```bash
# On your remote GPU machine:
git clone <repo> kv-probe && cd kv-probe
make build              # Build Docker (10-15 mins, one-time)

# Run quick validation test
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/smoke_test_probe.py"

# Run full probe experiment
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/probe_kv_structure.py \
    --model pythia-1b \
    --dataset wikitext \
    --num-samples 2000 \
    --layer 6 \
    --num-pairs 1000 \
    --output results/probe_layer6.json"

# Analyze results
python scripts/summarize_probe_results.py results/probe_*.json
```

## Research Questions

1. **Do transformer KV caches exhibit exploitable geometric structure?**
   - Does any distance metric predict attention patterns better than random?

2. **Which geometry best captures KV structure?**
   - Comparing different distance metrics systematically

3. **Is structure layer-specific or model-specific?**
   - Does it vary across layers (early vs middle vs late)?
   - Does it vary across models (Pythia vs Llama vs Qwen)?

4. **Is structure domain-specific?**
   - Code vs natural language vs mathematics

## Current Results

**Experiment Status:** 🔄 In Progress

Initial p-adic (p=2) experiments showed query-induced correlation **lower** than Euclidean distance. Now expanding to test additional geometries systematically.

📊 **[Full experimental details](docs/EXPERIMENTS.md)** - Hypothesis, methodology, analysis

## Documentation

📚 **Core Documentation**
- **[Experimental Log](docs/EXPERIMENTS.md)** - Hypothesis, results, and analysis 📊
- **[Probe Guide](docs/PROBE_GUIDE.md)** - Step-by-step instructions for running probes 🔬
- **[Quick Start](docs/QUICKSTART.md)** - Environment setup

## Project Structure

```
kv-probe/
├── src/
│   └── geometries/         # Distance metric implementations
│       └── padic.py        # P-adic ultrametric distance
├── scripts/
│   ├── probe_kv_structure.py      # Main probe experiment
│   ├── smoke_test_probe.py        # Quick validation
│   ├── summarize_probe_results.py # Result analysis
│   └── download_model.py          # Download models
├── tests/
│   └── test_padic_ops.py          # Geometry tests
└── docs/                           # Documentation
```

## Key Features

- **Multi-geometry probing** - Test different distance metrics
- **Per-head analysis** - Analyze each attention head independently
- **Statistical rigor** - Bootstrap confidence intervals, Spearman correlation
- **Null controls** - Permutation tests, random baselines
- **Extensible** - Easy to add new geometries

## License

MIT
