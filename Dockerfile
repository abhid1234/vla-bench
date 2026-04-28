# Reproducible eval environment for vla-bench.
# CPU base for the mock harness; swap to nvidia/cuda for real models.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["vla-bench"]
CMD ["eval", "--model", "mock", "--env", "mock-libero", "--tasks", "5", "--rollouts", "10"]
