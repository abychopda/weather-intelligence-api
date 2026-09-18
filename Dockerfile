# 1. Start from a small Linux image that already has Python 3.13.
FROM python:3.13-slim

# 2. Add uv, the tool used to install our dependencies.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. All following commands run inside the /app folder.
WORKDIR /app

# 4. Install dependencies first. Docker caches this step, so rebuilding after
#    a code change is fast as long as the dependencies stay the same.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 5. Copy in the application code and the trained models.
COPY app ./app
COPY models ./models

# 6. Start the API. Render tells the app which port to use through the PORT
#    environment variable; when running locally we fall back to 8000.
EXPOSE 8000
CMD uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
