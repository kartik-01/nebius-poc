VENV ?= .venv
BIN := $(VENV)/bin

# Local checkout runs the test suite on CPU. Cluster jobs use the CUDA build that
# ships inside the training container, so pulling CUDA wheels here wastes disk.
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

# The smoke path is opt-in because it downloads weights. It uses the 0.5B sibling of
# the real base model: same tokenizer and prompt format, a fraction of the bytes.
SMOKE_MODEL ?= Qwen/Qwen2.5-0.5B
SMOKE_ROOT ?= results/raw/smoke
SMOKE_QUESTIONS ?= 8

.PHONY: install test lint prepare-data smoke discover prefetch-smoke

install:
	python3 -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install --index-url $(TORCH_INDEX) torch
	$(BIN)/python -m pip install -e '.[dev]'

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

prepare-data:
	$(BIN)/python -m nebius_poc.data --config configs/train_sft.yaml

discover:
	./scripts/discover_cluster.sh

prefetch-smoke:
	$(BIN)/python scripts/prefetch_assets.py --smoke-only --boundary-check-only \
	  --hf-home $${HF_HOME:-hf_cache} --out results/raw

# Exercises train -> evaluate -> compare -> merge end to end on a small model. Runs
# the ranking objective because it is the harder of the two code paths.
smoke:
	rm -rf $(SMOKE_ROOT)
	$(BIN)/python -m nebius_poc.train --config configs/train_ranking.yaml \
	  --model $(SMOKE_MODEL) --limit 4 --batch-size 2 --max-steps 2 \
	  --results-root $(SMOKE_ROOT)
	@set -e; adapter=$$(ls -d $(SMOKE_ROOT)/*_train-*/adapter); \
	for variant in base tuned; do \
	  test $$variant = base && with="" || with="--adapter $$adapter"; \
	  $(BIN)/python -m nebius_poc.evaluate --config configs/train_ranking.yaml \
	    --model $(SMOKE_MODEL) $$with --label smoke-$$variant --split validation \
	    --limit $(SMOKE_QUESTIONS) --batch-size 2 --results-root $(SMOKE_ROOT); \
	done; \
	$(BIN)/python -m nebius_poc.report \
	  --base $(SMOKE_ROOT)/*_evaluate-smoke-base_*/forced_choice.jsonl \
	  --tuned $(SMOKE_ROOT)/*_evaluate-smoke-tuned_*/forced_choice.jsonl \
	  --resamples 1000 --out $(SMOKE_ROOT)/accuracy.json; \
	$(BIN)/python -m nebius_poc.merge_adapter --config configs/train_ranking.yaml \
	  --model $(SMOKE_MODEL) --adapter $$adapter --output $(SMOKE_ROOT)/merged \
	  --results-root $(SMOKE_ROOT)
	@echo "smoke complete: $(SMOKE_ROOT)"
