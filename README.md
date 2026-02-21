# AI Call Center — README

This repository contains an AI-powered call center demo with components for speech-to-text, dialect detection, retrieval-augmented generation (RAG) using a vector DB, and text-to-speech audio output. The project provides both a FastAPI-based web interface and utilities for generating and managing audio assets.

## Key Capabilities

- Real-time and batch audio processing (STT using Whisper-compatible workflows).
- Arabic dialect detection and adaptation (`src/AI/dialect_detector.py`).
- RAG integration with Qdrant via `src/RAGcontrollers/` for context-aware responses.
- Orchestration and live LLM interaction in `src/AI/gemini_live_manager.py`.
- TTS/audio generation helpers in `src/generate_audio_files.py` and `src/AI/audio_files/`.
- FastAPI routes and websocket endpoint for live interaction in `src/routers/` (see `websocket_endpoint.py`).

## Quickstart

Prerequisites:

- Python 3.10+
- FFmpeg (system-wide) for audio transformations
- A running Qdrant instance if you use the RAG features (can be run via Docker)

Install Python dependencies (from project root):

```bash
pip install -r src/requirements.txt
```

Copy and edit environment variables (see `env.example`):

```bash
copy env.example src\.env
```

Required environment variables (examples exist in `env.example`):

- `GEMINI_API_KEY` — API key for Gemini LLMs
- `QDRANT_HOST` / `QDRANT_PORT` — Vector DB connection
- `VECTOR_COLLECTION_NAME` — Qdrant collection name
- `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` — embedding settings

See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) for route and payload details.

## Running Locally

Run the FastAPI server (from `src/`):

```bash
cd src
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Use the websocket endpoint for live audio sessions: check `src/routers/websocket_endpoint.py` for details.

Generate audio assets (example):

```bash
python src/generate_audio_files.py
```

Run Qdrant (optional) using Docker Compose (project root):

```bash
docker-compose up --build
```

## Development Notes

- Core AI modules live in `src/AI/`:
    - `dialect_detector.py` — dialect detection utilities
    - `gemini_live_manager.py` — handles calls to Gemini and live orchestration
    - `session.py` — session/state management for conversations

- RAG controllers in `src/RAGcontrollers/` handle embeddings, indexing, and queries against Qdrant.
- FastAPI route handlers are in `src/routers/` (voice, websocket, and base routes).

## Project Structure (short)

- `src/` — application source
    - `AI/` — AI modules and helpers
    - `RAGcontrollers/` — Qdrant and embedding logic
    - `routers/` — FastAPI endpoints (including websocket support)
    - `helpers/` — config and utilities (`src/helpers/config.py`)
    - `generate_audio_files.py` — utility for producing TTS audio assets
    - `main.py` — FastAPI application entrypoint

## Logs, Data and Assets

- Generated audio files and examples are stored under `src/AI/audio_files/` and `src/assets/`.

## Notes & Next Steps

- Update `src/requirements.txt` if you add packages (a `pip freeze > src/requirements.txt` was used during development).
- Verify API keys and Qdrant connectivity before starting the server.

For full API details and integration examples, see [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

---

If you'd like, I can also:

- add a small example script showing how to connect to the websocket endpoint,
- update `env.example` with clearer variable descriptions, or
- run the server and smoke-test the main endpoints locally.

