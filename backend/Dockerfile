# syntax=docker/dockerfile:1

FROM python:3.14.2-slim-bookworm AS app
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv by copying the binary (recommended for Docker)
# (avoids relying on pip wheels existing for every new Python point release)
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install deps (prod only)
RUN if [ -f uv.lock ]; then uv sync --locked --no-dev; else uv sync --no-dev; fi

# Copy app code
COPY app ./app
COPY tests ./tests

RUN mkdir -p /app/out/logs /app/out/manifests /app/out/artifacts /app/out/benchmarks /app/out/capsule /app/out/config /app/out/provenance /app/out/offline /app/out/analysis

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
