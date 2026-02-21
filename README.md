# AI Call Center — Gemini Live Demo

A real-time, Arabic-first AI call center powered by the **Gemini Live API**. The system handles full speech-to-speech conversations with dialect adaptation, retrieval-augmented generation (RAG) from a Qdrant vector database, and a WebRTC audio interface served through FastRTC and Gradio.

---

## Architecture Overview

```
Browser (WebRTC)
      │  audio in/out
      ▼
FastRTC / ReplyOnPause
      │
      ├─► welcome_startup()  — plays welcome WAV + pre-opens Gemini session
      │
      └─► process_call()
            ├─ inject pending RAG context (from previous turn)
            ├─ _stream_with_hold_music()
            │      ├─ Gemini Live send_and_receive  ← starts immediately
            │      └─ hold_music.wav fills silence until first audio chunk
            ├─ dialect detection (keyword-based)
            └─ dynamic RAG search → queued for next turn

Gemini Live API  ◄──── GeminiLiveManager (persistent WebSocket per caller)

Qdrant Vector DB ◄──── VectorDB (LRU-cached embeddings via Gemini Embeddings API)
```

---

## Key Features

### Real-time Speech-to-Speech
- One **persistent Gemini Live WebSocket** per browser caller — no STT → LLM → TTS pipeline overhead.
- Audio streamed back in **50 ms chunks** (1200 samples @ 24 kHz) for minimal first-token latency.
- Input tail padding reduced to **100 ms** (was 250 ms) to lower turn-around time.

### Welcome Message on Connect
- A pre-recorded WAV (`welcome_egyptian.wav`) plays **immediately on WebRTC connect**, before the user speaks.
- 100 ms of silence is sent first to prime the browser AudioContext (fixes Chrome/Safari first-chunk silence).

### Hold Music During Processing
- `hold_music.wav` plays **concurrently** while Gemini processes each utterance.
- Gemini processing starts at the same instant — no extra latency added.
- Hold music stops the moment the first real audio chunk arrives and switches over seamlessly.
- File is optional: if absent, the server logs a warning and plays silence instead.

### Arabic Dialect Adaptation
- Keyword-based dialect detection per turn (Egyptian, Gulf, Sudanese, Levantine, MSA).
- Gemini voice automatically selected per dialect (`DIALECT_VOICES` map).
- Output audio transcription enabled in Gemini config for logging/debugging.

### RAG — Retrieval-Augmented Generation

| Stage | What happens |
|---|---|
| **Session seed** | On connect, `profile` doc-type chunks + KB keyword search run in parallel (`asyncio.gather`) and are injected into the Gemini system prompt. |
| **Per-turn dynamic RAG** | After each user utterance, a semantic search runs against Qdrant using the transcript. Results are queued and injected as a text message **before the next turn's audio**, so Gemini has fresh context when replying. |
| **Post-upload invalidation** | Uploading a new document immediately closes all open Gemini sessions. They are rebuilt with the new data on the next user turn. |

#### Document Upload
`POST /upload-and-index` accepts:
- `file` — PDF or DOCX
- `doc_type` *(optional Form field)* — `profile`, `company_kb`, `faq`, etc.

If `doc_type` is omitted, the filename is inspected for keywords (`cv`, `resume`, `سيرة`, `سيره`, `profile`, `بيانات`, `شخصية`) to auto-tag as `profile`; otherwise defaults to `company_kb`.

Chunks are created using **semchunk** (semantic, word-level splitting, chunk size 800) instead of fixed-size splitting.

### Session & Connection Management
- Each browser tab gets its own session keyed by `webrtc_id` — no cross-contamination between concurrent callers.
- A background task runs every 60 s to close and remove sessions where `is_connected() == False`.
- All sessions are closed gracefully on server shutdown.

### Embedding Cache
- LRU cache (`cachetools.LRUCache`, max 500 entries) for Gemini embedding vectors.
- Common Arabic support queries are **pre-warmed at server startup** so the first real embedding call hits cache.

---

## Project Structure

```
src/
├── main.py                        # FastAPI app, lifespan hooks (cache pre-warm, session cleanup)
├── generate_audio_files.py        # Utility to generate welcome/hold WAV files via TTS
├── AI/
│   ├── gemini_live_manager.py     # GeminiLiveManager — persistent Live API session per caller
│   ├── dialect_detector.py        # Keyword-based Arabic dialect detection
│   ├── session.py                 # CallSession state (dialect lock, history)
│   └── audio_files/
│       ├── welcome_egyptian.wav   # Played on WebRTC connect
│       └── hold_music.wav         # Played while Gemini processes each turn
├── RAGcontrollers/
│   ├── VectorDB.py                # Qdrant client, LRU embedding cache, search, insert
│   ├── DataProcessing.py          # Docling parsing + semchunk semantic splitting
│   ├── DataController.py          # File storage helpers
│   └── seed_qdrant.py             # One-shot seeding script
├── routers/
│   ├── websocket_endpoint.py      # FastRTC stream, welcome/hold logic, process_call, dynamic RAG
│   ├── base.py                    # /upload-and-index REST endpoint
│   └── voice.py                   # Additional voice REST routes
└── helpers/
    └── config.py                  # Settings loaded from .env
```

---

## Quickstart

### Prerequisites

- Python 3.11
- Docker (for Qdrant)
- FFmpeg (system-wide, for audio format conversion)

### 1. Clone & install

```bash
git clone https://github.com/AhmedHossam61/ai_call_center.git
cd ai_call_center
python -m pip install -r src/requirements.txt
```

### 2. Environment variables

```bash
copy env.example src\.env   # Windows
# cp env.example src/.env   # Linux/macOS
```

Edit `src/.env`:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `QDRANT_HOST` | Qdrant host (default `localhost`) |
| `QDRANT_PORT` | Qdrant port (default `6333`) |
| `VECTOR_COLLECTION_NAME` | Qdrant collection name |
| `EMBEDDING_MODEL` | Gemini embedding model (e.g. `models/text-embedding-004`) |
| `EMBEDDING_DIMENSION` | Embedding vector size (must match model) |

### 3. Start Qdrant

```bash
docker-compose up -d qdrant
```

### 4. Generate audio assets

```bash
cd src
python generate_audio_files.py
```

Place your own `hold_music.wav` in `src/AI/audio_files/` if desired (any sample rate, mono or stereo WAV).

### 5. Run the server

```bash
cd src
python main.py
```

| URL | Purpose |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — REST API |
| `http://localhost:8001` | Gradio WebRTC UI — live call |

### 6. Upload knowledge base documents

Use Swagger or curl:

```bash
curl -X POST http://localhost:8000/upload-and-index \
  -F "file=@my_resume.pdf" \
  -F "doc_type=profile"
```

After upload, existing Gemini sessions are invalidated and rebuilt with the new data on the next call.

---

## Audio Files Reference

| File | Role | Required |
|---|---|---|
| `welcome_egyptian.wav` | Played immediately on WebRTC connect | Yes (skipped if missing) |
| `hold_music.wav` | Looped while Gemini processes each turn | No (silence if missing) |

---

## API Reference

See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) for full route and payload details.

---

## Deployment (Docker)

```bash
docker-compose up --build
```

Set all required env vars in `.env` before building. The Dockerfile runs from `src/` and exposes ports 8000 (API) and 8001 (Gradio UI).

