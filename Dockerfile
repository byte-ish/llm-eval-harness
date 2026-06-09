# syntax=docker/dockerfile:1.7
# Multi-stage build for llm-eval-harness.
# Stage 1 (builder): install dependencies + project with uv.
# Stage 2 (runtime): slim image with non-root user, no build tooling.

FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.4.0 /uv /bin/uv

# Resolve and install dependencies first (cache-friendly: changes to source
# don't bust the dep cache).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project itself.
COPY harness/ harness/
COPY run_evals.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user — never run the harness as root.
RUN groupadd --system harness && useradd --system --gid harness --create-home harness

COPY --from=builder --chown=harness:harness /app /app
COPY --chown=harness:harness evals/ evals/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER harness

ENTRYPOINT ["python", "/app/run_evals.py"]
