.PHONY: install format lint typecheck test coverage ingest baseline train predict leaderboard explain check

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e '.[dev]'

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src/gridmind --cov-report=term-missing --cov-report=html

ingest:
	$(PYTHON) -m gridmind.cli ingest

baseline:
	$(PYTHON) -m gridmind.cli baseline

train:
	$(PYTHON) -m gridmind.cli train

predict:
	$(PYTHON) -m gridmind.cli predict

leaderboard:
	$(PYTHON) -m gridmind.cli leaderboard

explain:
	$(PYTHON) -m gridmind.cli explain

check: lint typecheck coverage
