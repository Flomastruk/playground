FROM python:3.12-bookworm

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

COPY pyproject.toml ./

ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN uv sync --no-dev