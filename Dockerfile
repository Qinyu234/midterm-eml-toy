FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install sympy numpy scipy pytest

COPY . .

# Default: run full SCORE suite (override via docker compose / docker run).
CMD ["python", "scripts/run_score.py", "--targets", "all", "--max-steps", "500", "--eps", "1e-2"]
