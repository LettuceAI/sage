PYTHON ?= python
PIP    ?= pip

.PHONY: help install test lint format type rust check clean

help:
	@echo "Targets:"
	@echo "  install      install package with dev extras (editable)"
	@echo "  test         run python test suite"
	@echo "  lint         run ruff lint"
	@echo "  format       run ruff format in-place"
	@echo "  type         run mypy"
	@echo "  rust         cargo check + cargo test for the rust crate"
	@echo "  check        full check: lint + type + test + rust"
	@echo "  clean        remove build artifacts"

install:
	$(PIP) install -e '.[train,export,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

type:
	$(PYTHON) -m mypy sage training

rust:
	cd rust/sage-rs && cargo check && cargo test

check: lint type test rust

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	cd rust/sage-rs && cargo clean
