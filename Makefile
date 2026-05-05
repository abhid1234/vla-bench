.PHONY: install eval-mock test fmt lint clean

install:
	uv pip install -e ".[dev]"

eval-mock:
	uv run python -m vla_bench.cli eval --model mock --env mock-libero --tasks 5 --rollouts 10

test:
	uv run pytest

fmt:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
