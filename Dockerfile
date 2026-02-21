FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    libportaudio2 \
    libasound2 \
    portaudio19-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

RUN mkdir -p /app/data/uploads /app/data/embeddings /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser

COPY src/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel

RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20240930

RUN pip install --no-cache-dir -r requirements.txt


COPY . .

ENV UPLOAD_DIR=/app/data/uploads
ENV STORAGE_DIR=/app/data/embeddings
ENV HF_HOME=/home/appuser/.cache
ENV PYTHONPATH=/app/src

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]