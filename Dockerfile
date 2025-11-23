FROM python:3.12-slim-bookworm AS app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ADD . /app

RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "python", "src/model_trainer.py"]
