# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV UV_SYSTEM_PYTHON=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create venv and activate it
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install setuptools INSIDE the venv first (fixes pkg_resources for webrtcvad)
RUN uv pip install --python /app/.venv/bin/python setuptools

# Install all dependencies into the venv
COPY src/requirements.txt .
RUN uv pip install --python /app/.venv/bin/python -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    SDL_AUDIODRIVER=dummy \
    SDL_VIDEODRIVER=dummy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    libasound2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src/ .

EXPOSE 8000 8001

CMD ["/app/.venv/bin/python", "main.py"]