# KV-Probe Development Environment
# Using CUDA 12.1 (compatible with driver 580.x / CUDA 13.0)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    cmake \
    libpari-dev \
    pari-gp \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6 support (use cu124 which is latest stable)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install JAX with CUDA 12 support
RUN pip3 install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Triton for custom kernels
RUN pip3 install triton

# Install core ML libraries
RUN pip3 install \
    transformers \
    accelerate \
    deepspeed \
    bitsandbytes \
    datasets \
    tokenizers \
    sentencepiece \
    safetensors

# Install mathematics libraries (cypari2 removed - not needed for current implementation)
RUN pip3 install sympy mpmath

# Install evaluation frameworks
RUN pip3 install \
    lm-eval \
    evaluate \
    sacrebleu

# Install experiment tracking
RUN pip3 install wandb tensorboard

# Install utilities
RUN pip3 install \
    numpy \
    scipy \
    pandas \
    tqdm \
    pyyaml \
    omegaconf \
    einops \
    matplotlib \
    seaborn \
    plotly \
    jupyterlab \
    ipywidgets

# Install development tools
RUN pip3 install \
    pytest \
    black \
    isort \
    flake8 \
    mypy

# Set working directory
WORKDIR /workspace/kv-probe

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Set default command
CMD ["/bin/bash"]
