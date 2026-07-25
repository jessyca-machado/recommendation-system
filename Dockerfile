# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group deep

COPY src ./src
COPY scripts ./scripts
COPY config ./config

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group deep


FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1002 appuser \
    && mkdir -p /app \
    && chown -R appuser:appuser /app

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser config ./config

RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/artifacts/metrics \
    /app/artifacts/models && \
    chown -R appuser:appuser /app/artifacts /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

USER appuser

CMD ["train"]
