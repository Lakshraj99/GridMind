.PHONY: install format lint typecheck test coverage quality check api dashboard docker-up docker-down \
	clean check-secrets ingest baseline train predict leaderboard explain

PYTHON ?= python3

install:
	$(PYTHON) -m pip install --upgrade pip
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
	$(PYTHON) -m pytest --cov=src/gridmind --cov-config=pyproject.toml \
		--cov-report=term-missing --cov-fail-under=85

coverage:
	$(PYTHON) -m coverage erase
	find . -maxdepth 2 -type f -name '.coverage*' -delete
	$(PYTHON) -m pytest --cov=src/gridmind --cov-config=pyproject.toml \
		--cov-report=term-missing --cov-report=html --cov-fail-under=85

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

quality: lint typecheck coverage

check: quality

api:
	gridmind-api

dashboard:
	gridmind-dashboard

docker-up:
	docker compose up --build

docker-down:
	docker compose down

check-secrets:
	$(PYTHON) scripts/check_tracked_secrets.py

clean:
	find . -maxdepth 2 -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} +
	find . -maxdepth 2 -type f \( -name '.coverage' -o -name '.coverage.*' -o -name 'coverage.xml' \) -delete
	rm -rf htmlcov
