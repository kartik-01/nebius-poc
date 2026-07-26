VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test lint prepare-data smoke

install:
	python3 -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e '.[dev]'

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

prepare-data:
	$(BIN)/python -m nebius_poc.data --config configs/train_sft.yaml

smoke:
	@echo "not implemented yet"; exit 1
