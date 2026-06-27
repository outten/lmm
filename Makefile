# outten/lmm — Makefile
#
# Common workflows. Run `make help` for a list.

SHELL := /bin/bash
PY   ?= python3
PIP  ?= python3 -m pip

OLLAMA     ?= ollama
MODEL      ?= outten/lmm
TAG        ?= latest
BASE_MODEL ?= llama3.2

DATA       := training/data/sft.jsonl
RUN_DIR    := training/runs/$(TAG)
GGUF_DIR   := $(RUN_DIR)/gguf
LLAMA_CPP  := $(HOME)/src/llama.cpp

.PHONY: help install install-dev sandbox-test build run eval \
        download-benchmarks train convert push clean

help:           ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:        ## Install runtime deps for the sandbox
	$(PIP) install -U matplotlib sympy numpy pandas scipy requests pyyaml
	@echo "Install mermaid-cli with: npm install -g @mermaid-js/mermaid-cli"

install-dev:    ## Install dev deps (training + eval)
	$(PIP) install -U unsloth transformers trl peft accelerate datasets bitsandbytes

sandbox-test:   ## Run the sandbox test suite
	$(PY) -m unittest discover tests -v

build:          ## Build the Ollama model
	$(OLLAMA) create $(MODEL):$(TAG) -f Modelfile

run:            ## Chat with the model + run code blocks
	$(PY) sandbox/ollama_cli.py --model $(MODEL):$(TAG)

download-benchmarks:  ## Download GSM8K + MATH test splits
	$(PY) eval/harness/download_benchmarks.py --all

eval:           ## Run the smoke benchmark
	$(PY) -m eval.harness.runner \
		--benchmark eval/benchmarks/smoke.jsonl \
		--model $(MODEL):$(TAG) \
		--output eval/results/$(TAG).json

eval-gsm8k:     ## Run GSM8K (after `make download-benchmarks`)
	$(PY) -m eval.harness.runner \
		--benchmark eval/benchmarks/gsm8k_test.jsonl \
		--model $(MODEL):$(TAG) \
		--output eval/results/$(TAG)-gsm8k.json \
		--timeout 60

train:          ## Fine-tune the model (requires GPU)
	$(PY) training/scripts/sft_train.py \
		--config training/configs/$(TAG).yaml

convert:        ## HF -> GGUF -> Ollama
	$(PY) training/scripts/convert_to_gguf.py \
		$(RUN_DIR)/merged \
		--out $(GGUF_DIR) \
		--llama-cpp-dir $(LLAMA_CPP) \
		--quant Q4_K_M

push:           ## Push the model to the Ollama registry
	$(OLLAMA) push $(MODEL):$(TAG)

clean:          ## Remove generated artifacts
	rm -rf eval/results training/runs/*/gguf