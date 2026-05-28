FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first so this layer is cached unless deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source after deps are installed
COPY configs/ ./configs/
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

# MODE=train  → run the training pipeline (default)
# MODE=serve  → start the FastAPI prediction server
ARG MODE=train
ENV APP_MODE=${MODE}

CMD if [ "$APP_MODE" = "serve" ]; then \
        uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000; \
    else \
        python src/pipeline_runner.py; \
    fi
