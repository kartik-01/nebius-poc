VENV ?= .venv
BIN := $(VENV)/bin

# Local checkout runs the test suite on CPU. Cluster jobs use the CUDA build that
# ships inside the training container, so pulling CUDA wheels here wastes disk.
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: install test lint prepare-data smoke

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

smoke:
	@echo "not implemented yet"; exit 1
