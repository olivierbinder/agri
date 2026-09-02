# https://docs.docker.com/engine/reference/builder/

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MODEL_URI="/app/deploy/model"

# libgomp1: required at runtime by xgboost/sklearn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first — cacheable layer, invalidated only when these files change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# Source code
COPY src/ ./src/
RUN uv sync --frozen --no-default-groups

# Bundled Champion model (see `just docker-export-model`) — no MLflow registry needed at runtime
COPY deploy/model/ ./deploy/model/

EXPOSE 8000

# API only — the Streamlit frontend is deployed separately (Streamlit Community Cloud).
# Shell form (not exec form) so $PORT expands — Render assigns its own port at runtime;
# defaults to 8000 for local `docker run` / docker-compose.
CMD uvicorn agri.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
