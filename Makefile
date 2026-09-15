.PHONY: help build up down shell jupyter tensorboard train eval test download clean logs probe-smoke probe-exp1 probe-quick

# Project configuration
PROJECT_NAME := padic-transformers
DOCKER_COMPOSE := docker-compose
DOCKER_RUN := $(DOCKER_COMPOSE) run --rm padic-dev

# Default target
help:
	@echo "P-adic Transformers - Make Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (create dirs, env file)"
	@echo "  make build          - Build Docker image"
	@echo ""
	@echo "Container Management:"
	@echo "  make up             - Start container in background"
	@echo "  make down           - Stop container"
	@echo "  make shell          - Open interactive bash shell"
	@echo "  make logs           - View container logs"
	@echo ""
	@echo "Development:"
	@echo "  make jupyter        - Start Jupyter Lab (port 8888)"
	@echo "  make tensorboard    - Start TensorBoard (port 6006)"
	@echo "  make test           - Run pytest tests"
	@echo ""
	@echo "Data & Training:"
	@echo "  make download       - Download datasets"
	@echo "  make train CONFIG=<path>  - Train model with config"
	@echo "  make eval CKPT=<path>     - Evaluate checkpoint"
	@echo ""
	@echo "Experiments:"
	@echo "  make probe-smoke    - Run smoke test for p-adic probe"
	@echo "  make probe-exp1     - Run Experiment 1 (2000 samples, 1000 pairs)"
	@echo "  make probe-quick    - Quick probe test (200 samples, 100 pairs)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Clean up Docker resources"
	@echo "  make clean-data     - Remove downloaded datasets"
	@echo "  make clean-all      - Clean everything"
	@echo ""

# Setup
setup:
	@echo "Setting up project structure..."
	@mkdir -p datasets checkpoints results logs notebooks/exploration
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo "Created .env file - please edit with your API keys"; \
	fi
	@echo "Setup complete!"

# Docker operations
build:
	@echo "Building Docker image..."
	$(DOCKER_COMPOSE) build

up:
	@echo "Starting container in background..."
	$(DOCKER_COMPOSE) up -d
	@echo "Container started. Use 'make logs' to view logs"

down:
	@echo "Stopping container..."
	$(DOCKER_COMPOSE) down

shell:
	@echo "Opening interactive shell..."
	$(DOCKER_RUN) bash

logs:
	$(DOCKER_COMPOSE) logs -f

# Development tools
jupyter:
	@echo "Starting Jupyter Lab on http://localhost:8888"
	$(DOCKER_COMPOSE) run --rm --service-ports padic-dev \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	$(DOCKER_COMPOSE) run --rm --service-ports padic-dev \
		tensorboard --logdir=/workspace/padic-transformers/results --host=0.0.0.0

test:
	@echo "Running tests..."
	$(DOCKER_RUN) pytest tests/ -v

# Data operations
download:
	@echo "Downloading datasets..."
	$(DOCKER_RUN) python scripts/download_data.py

download-pile:
	@echo "Downloading The Pile..."
	$(DOCKER_RUN) python scripts/download_data.py --dataset pile

download-stack:
	@echo "Downloading The Stack..."
	$(DOCKER_RUN) python scripts/download_data.py --dataset stack

download-math:
	@echo "Downloading math datasets..."
	$(DOCKER_RUN) python scripts/download_data.py --dataset proofpile,openwebmath

# Model operations
download-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Listing available models:"; \
		$(DOCKER_RUN) python scripts/download_model.py --list; \
	else \
		echo "Downloading model: $(MODEL)"; \
		$(DOCKER_RUN) python scripts/download_model.py $(MODEL) --test; \
	fi

download-pythia:
	@echo "Downloading Pythia-1B..."
	$(DOCKER_RUN) python scripts/download_model.py pythia-1b --test

download-tinyllama:
	@echo "Downloading TinyLlama-1.1B..."
	$(DOCKER_RUN) python scripts/download_model.py tinyllama --test

# Training operations
train:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG not specified. Usage: make train CONFIG=configs/1b_hybrid.yaml"; \
		exit 1; \
	fi
	@echo "Training with config: $(CONFIG)"
	$(DOCKER_RUN) python scripts/train.py --config $(CONFIG)

train-1b:
	@echo "Training 1B hybrid model..."
	$(DOCKER_RUN) python scripts/train.py --config configs/1b_hybrid.yaml

train-resume:
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: CKPT not specified. Usage: make train-resume CKPT=checkpoints/step_1000.pt"; \
		exit 1; \
	fi
	@echo "Resuming training from: $(CKPT)"
	$(DOCKER_RUN) python scripts/train.py --resume $(CKPT)

# Evaluation operations
eval:
	@if [ -z "$(CKPT)" ]; then \
		echo "Error: CKPT not specified. Usage: make eval CKPT=checkpoints/best_model.pt"; \
		exit 1; \
	fi
	@echo "Evaluating checkpoint: $(CKPT)"
	$(DOCKER_RUN) python scripts/evaluate.py --checkpoint $(CKPT)

eval-all:
	@echo "Running full benchmark suite..."
	$(DOCKER_RUN) python scripts/evaluate.py --checkpoint $(CKPT) --benchmarks all

eval-mmlu:
	@echo "Evaluating on MMLU..."
	$(DOCKER_RUN) python scripts/evaluate.py --checkpoint $(CKPT) --benchmarks mmlu

eval-math:
	@echo "Evaluating on math benchmarks..."
	$(DOCKER_RUN) python scripts/evaluate.py --checkpoint $(CKPT) --benchmarks math,gsm8k

# Experiment 1: P-adic structure probe
probe-smoke:
	@echo "Running smoke test for p-adic probe..."
	$(DOCKER_RUN) python3 scripts/smoke_test_probe.py

probe-exp1:
	@if [ -z "$(LAYER)" ]; then \
		LAYER=6; \
	else \
		LAYER=$(LAYER); \
	fi; \
	echo "Running Experiment 1: P-adic structure probe on WikiText"; \
	echo "  Model: pythia-1b, Layer: $$LAYER"; \
	echo "  Samples: 2000, Pairs per head: 1000, Bootstrap: 1000"; \
	mkdir -p results; \
	$(DOCKER_RUN) python3 scripts/probe_kv_structure.py \
		--layer $$LAYER \
		--num-samples 2000 \
		--num-pairs 1000 \
		--n-bootstrap 1000 \
		--output results/probe_wikitext_layer$${LAYER}_1M.json

probe-quick:
	@if [ -z "$(LAYER)" ]; then \
		LAYER=6; \
	else \
		LAYER=$(LAYER); \
	fi; \
	echo "Running quick probe test (200 samples, 100 pairs)..."; \
	echo "  Layer: $$LAYER"; \
	mkdir -p results; \
	$(DOCKER_RUN) python3 scripts/probe_kv_structure.py \
		--layer $$LAYER \
		--num-samples 200 \
		--num-pairs 100 \
		--n-bootstrap 100 \
		--output results/probe_quick_layer$${LAYER}.json

# Leaderboard submission
submit-openllm:
	@echo "Submitting to Open LLM Leaderboard..."
	$(DOCKER_RUN) python scripts/submit_to_leaderboard.py --platform openllm --checkpoint $(CKPT)

# Utilities
clean:
	@echo "Cleaning Docker resources..."
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

clean-data:
	@echo "Removing downloaded datasets..."
	@read -p "This will delete all data in datasets/. Continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf datasets/*; \
		echo "Datasets removed"; \
	fi

clean-checkpoints:
	@echo "Removing checkpoints..."
	@read -p "This will delete all checkpoints. Continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf checkpoints/*; \
		echo "Checkpoints removed"; \
	fi

clean-all: clean clean-data clean-checkpoints
	@echo "Full cleanup complete"

# Custom commands
exec:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: CMD not specified. Usage: make exec CMD='python script.py'"; \
		exit 1; \
	fi
	$(DOCKER_RUN) $(CMD)

# GPU info
gpu-info:
	@echo "GPU Information:"
	$(DOCKER_RUN) nvidia-smi

# Python dependencies
install-deps:
	@echo "Installing additional Python dependencies..."
	$(DOCKER_RUN) pip install -r requirements.txt
