PY := python

.PHONY: help install install-dev test test-unit test-integration test-fast test-slow test-cov lint format type-check security-check clean clean-all

help:
	@echo "Targets: install, install-dev, test, test-unit, test-integration, test-fast, test-slow, test-cov, lint, format, type-check, security-check, clean, clean-all"

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install black flake8 mypy bandit safety

test:
	pytest -v

test-unit:
	pytest -m unit -v

test-integration:
	pytest -m integration -v

test-fast:
	pytest -m "not slow" -v

test-slow:
	pytest -m slow -v

test-cov:
	pytest --cov=src --cov-report=html:htmlcov -v

lint:
	flake8 src tests

format:
	black src tests

type-check:
	mypy src

security-check:
	bandit -r src || true
	safety check || true

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ .mypy_cache .coverage htmlcov

clean-all: clean
	rm -rf models data/preprocessed logs

