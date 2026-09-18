# Quick Start Guide - Remote GPU Setup

Get your first p-adic compression experiment running in 1 hour.

---

## Step 1: Transfer to Remote Machine

### Option A: Using rsync

```bash
# From your local machine
cd /Users/saurabhbhadada/Desktop/current/research
rsync -avz --exclude '.git' kv-probe/ user@your-gpu-server:~/kv-probe/
```

### Option B: Using git (recommended)

```bash
# On local machine
cd kv-probe
git init
git add .
git commit -m "Initial p-adic transformers setup"
git remote add origin <your-repo-url>
git push -u origin main

# On remote machine
ssh user@your-gpu-server
git clone <your-repo-url> kv-probe
cd kv-probe
```

---

## Step 2: Build Docker Environment

```bash
# SSH into GPU machine
ssh user@your-gpu-server
cd ~/kv-probe

# Setup directories and config
make setup

# Build Docker image (10-15 minutes, one-time)
make build

# Verify GPUs are accessible
make gpu-info
```

**Expected output**: Should show your NVIDIA GPU information

---

## Step 3: Download Pythia-1B

```bash
# Download pretrained model (~2GB, 5 minutes)
make download-pythia
```

This will:
- Download Pythia-1B from HuggingFace
- Cache in `checkpoints/pretrained/`
- Run test generation to verify it works

---

## Step 4: Run Baseline Experiment

```bash
# Run baseline evaluation (FP16, no compression)
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_baseline.py \
    --model pythia-1b \
    --dataset wikitext \
    --context-lengths 2048 \
    --output results/baseline.json"
```

**What this does**:
- Computes perplexity on WikiText-103
- Measures peak GPU memory
- Saves results to JSON

**Expected time**: 5-10 minutes

**Expected results**:
```
Perplexity: ~12-13
Peak memory: ~60-70 MB (for KV cache)
```

---

## Step 5: Run Compression Experiment

```bash
# Run with 8-bit 2-adic compression
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py \
    --model pythia-1b \
    --dataset wikitext \
    --compression simple_2x \
    --precision 8 \
    --context-lengths 2048 \
    --output results/compression_2x.json"
```

**What this does**:
- Same perplexity evaluation
- Applies 2-adic compression to KV cache
- Measures compression overhead and ratio

**Expected time**: 5-10 minutes

**Expected results**:
```
Perplexity: ~13-14 (slight increase)
Peak memory: ~30-40 MB (2x reduction!)
Compression ratio: ~2.0x
```

---

## Step 6: Visualize Results

```bash
# Start Jupyter Lab
make jupyter

# In browser: http://your-server:8888
# Open and run: notebooks/01_visualize_compression_results.ipynb
```

**What you get**:
- Comparison table (baseline vs compressed)
- Perplexity plots
- Memory usage plots
- Accuracy-memory tradeoff visualization
- CSV export for paper

---

## Next Steps: Extended Experiments

### Multiple Context Lengths

```bash
# Baseline with longer contexts
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_baseline.py \
    --context-lengths 2048,4096,8192 \
    --output results/baseline_long.json"

# Compression with longer contexts
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py \
    --compression simple_2x \
    --context-lengths 2048,4096,8192 \
    --output results/compression_2x_long.json"
```

**Why**: Compression benefits scale with context length!

### Adaptive Compression

```bash
# Adaptive precision (16/8/4 bits)
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py \
    --compression adaptive \
    --context-lengths 2048,4096,8192,16384 \
    --output results/compression_adaptive.json"
```

**Expected**: 3-4x compression with better accuracy preservation!

---

## Running in Background

### Using tmux (recommended)

```bash
# Create session
tmux new -s padic-exp

# Run experiment
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py ..."

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t padic-exp
```

### Using nohup

```bash
# Run in background
nohup make exec CMD="python scripts/run_baseline.py ..." > logs/baseline.log 2>&1 &

# Monitor progress
tail -f logs/baseline.log
```

---

## Common Commands

```bash
# Development
make shell                   # Interactive shell in container
make jupyter                 # Start Jupyter (port 8888)
make test                    # Run unit tests
make gpu-info                # Show GPU information

# Model management
make download-pythia         # Download Pythia-1B
make download-tinyllama      # Download TinyLlama-1.1B

# Cleanup
make clean                   # Remove Docker resources
make down                    # Stop containers
```

---

## Troubleshooting

### "CUDA out of memory"

```bash
# Use single GPU
CUDA_VISIBLE_DEVICES=0 make exec CMD="..."

# Limit evaluation samples
python scripts/run_baseline.py --max-samples 100 ...

# Use smaller context
--context-lengths 1024,2048
```

### "Docker not found"

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

### "nvidia-docker not working"

```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
make gpu-info  # Test
```

### "Module not found: src.kernels"

```bash
# Run from within container
make shell
cd /workspace/kv-probe
python scripts/run_baseline.py ...
```

---

## File Locations

```
kv-probe/
├── checkpoints/pretrained/  # Downloaded models
├── results/                 # Experiment outputs (JSON, plots)
├── notebooks/               # Jupyter analysis notebooks
├── logs/                    # Log files
└── datasets/                # Datasets (for Track 2)
```

---

## Expected Timeline

| Task | Time |
|------|------|
| Transfer to remote | 1-2 mins |
| Build Docker | 10-15 mins |
| Download Pythia-1B | 5 mins |
| Baseline experiment | 5-10 mins |
| Compression experiment | 5-10 mins |
| Visualization | 5 mins |
| **Total to first results** | **~40-60 mins** |

---

## What's Next

**After first results (Week 1)**:
1. Run multiple context lengths (2k, 4k, 8k)
2. Test adaptive compression
3. Generate comparison plots
4. Document initial findings

**Week 2**: Long context & MMLU benchmarks

**Weeks 3+**: Track 2 (full model training)

See `docs/EXPERIMENTAL_PLAN.md` for complete roadmap.

---

## Quick Command Summary

```bash
# Complete first experiment:
make build
make download-pythia
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_baseline.py --context-lengths 2048 --output results/baseline.json"
CUDA_VISIBLE_DEVICES=0 make exec CMD="python scripts/run_compression.py --compression simple_2x --context-lengths 2048 --output results/compression_2x.json"
make jupyter  # Analyze in notebook
```

**You're ready to go! 🚀**
