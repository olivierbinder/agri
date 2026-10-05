# https://docs.docker.com/engine/reference/builder/

# %% BUILDER

# Only the main dependencies: the API needs neither gradio (Dockerfile.ui) nor shap,
# matplotlib, plotly, seaborn (extras in pyproject.toml). Bytecode is precompiled here
# so the runtime stage starts faster (Cloud Run cold start) — no PYTHONDONTWRITEBYTECODE.
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first — cacheable layer, invalidated only when these files change.
# nvidia-nccl-cu12 is a Linux-only dependency of xgboost (GPU collectives, ~400 MB):
# the API predicts on CPU, so skip the install (the lock file is left untouched).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project \
    --no-install-package nvidia-nccl-cu12

# Source code
COPY src/ ./src/
RUN uv sync --frozen --no-default-groups --no-install-package nvidia-nccl-cu12

# %% RUNTIME

# Same base image, so the venv built above is reusable as-is (same Python, same paths).
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_URI="/app/deploy/model"

# libgomp1: required at runtime by xgboost/sklearn (OpenMP).
# Also drop the parts of the base image we never use (uv/pip/caches already in builder).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/lib/python3.12/test \
              /usr/local/lib/python3.12/idlelib \
              /usr/local/lib/python3.12/ensurepip \
              /usr/local/lib/python3.12/site-packages/pip \
              /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
              /usr/local/lib/python3.12/site-packages/wheel \
              /usr/local/lib/python3.12/site-packages/wheel-*.dist-info \
    && find /usr/local/lib/python3.12 -name '__pycache__' -prune -exec rm -rf {} +

WORKDIR /app

# The venv holds the precompiled dependencies; src/ is needed because the project is
# installed in editable mode (agri.pth points at /app/src).
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Bundled Champion model (see `just docker-export-model`) — no MLflow registry needed at runtime
COPY deploy/model/ ./deploy/model/

# Real 2013 yields, for the API's "actual vs predicted" comparison
COPY deploy/reference/ ./deploy/reference/

EXPOSE 8000

# API only — the Gradio UI is its own Cloud Run service (see Dockerfile.ui).
# Shell form (not exec form) so $PORT expands — Cloud Run injects its own port at
# runtime; defaults to 8000 for local `docker run` / docker-compose.
CMD uvicorn agri.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
