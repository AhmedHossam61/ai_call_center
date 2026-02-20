# AI Call Center API Documentation

## Overview

**AI Call Center API** is a real-time voice-based customer service system with Arabic dialect detection and RAG (Retrieval-Augmented Generation) support. The API enables live voice conversations with an AI agent that can understand, process, and respond to customer queries in Arabic.

- **Version**: 2.0.0
- **Base URL**: `http://localhost:8000`
- **Documentation**: `/docs` (Swagger UI), `/redoc` (ReDoc)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Call Center                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │   FastAPI    │──▶│   STT        │──▶│  Dialect        │    │
│  │   Server     │   │   (Whisper)  │   │  Detector       │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
│         │                                      │                │
│         ▼                                      ▼                │
│  ┌──────────────┐                     ┌──────────────────┐     │
│  │  WebSocket   │                     │  Response        │     │
│  │  Endpoint    │                     │  Generator       │     │
│  └──────────────┘                     │  (Gemini)        │     │
│         │                             └──────────────────┘     │
│         ▼                                    │                   │
│  ┌──────────────┐                            ▼                   │
│  │   TTS        │                   ┌──────────────────┐       │
│  │  (Synthesis) │◀───────────────────│  VectorDB        │       │
│  └──────────────┘                   │  (Qdrant +      │       │
│                                      │   Gemini Embed) │       │
│                                      └──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### 1. Health & Welcome

#### GET `/welcome`

Returns basic application information.

**Response:**
```json
{
  "app_name": "AI Call Center",
  "app_version": "2.0.0"
}
```

---

#### GET `/voice/health`

Health check endpoint for voice processing router.

**Response:**
```json
{
  "status": "voice router ok"
}
```

---

### 2. Knowledge Base Management

#### POST `/upload-and-index`

Uploads and indexes a document into the knowledge base (RAG pipeline).

**Description:**
Unified pipeline that performs:
1. **Relational Storage** - Saves file to disk and stores metadata in PostgreSQL
2. **AI Parsing** - Extracts text using Docling
3. **Vector Indexing** - Generates embeddings using Gemini and stores in Qdrant

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (UploadFile) - PDF or document file

**Response:**
```json
{
  "status": "Success",
  "payload": {
    "file_name": "example.pdf",
    "postgres_id": 1,
    "chunks_count": 15,
    "qdrant_points": 15,
    "collection": "knowledge_base"
  },
  "message": "Successfully indexed example.pdf. Ready for RAG."
}
```

**Error Response:**
```json
{
  "status": "Error",
  "detail": "Internal server error during document indexing",
  "error_log": "Detailed error message"
}
```

---

### 3. Live Voice Call (WebSocket)

#### WebSocket `/ws/live-call/{session_id}`

Real-time voice conversation endpoint. Uses WebSocket protocol for bidirectional communication.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Unique session identifier |

**Connection Flow:**

1. **Client connects** to WebSocket with a unique `session_id`
2. **Client sends** audio chunks (binary float32 data at 16kHz sample rate)
3. **Server processes** and responds with:
   - JSON metadata (detected dialect, response text)
   - Audio bytes (synthesized speech)

**Client Message Types:**

**Audio Data (Binary):**
- Format: `float32` PCM audio
- Sample Rate: 16000 Hz
- Minimum chunk size: 1.5 seconds worth of samples

**Control Message (JSON):**
```json
{
  "action": "end_session"
}
```

**Server Response:**

**JSON Metadata:**
```json
{
  "event": "ai_reply",
  "text": "مرحباً، كيف يمكنني مساعدتك؟",
  "dialect": "egyptian",
  "status": "success"
}
```

**Audio Response (Binary):**
- Format: WAV audio data
- Contains synthesized speech response

**Processing Pipeline:**

```
1. Receive Audio → 2. STT (Whisper) → 3. Dialect Detection
       ↓                                        ↓
4. Vector Search (Qdrant) ←───────────  5. Response Generation (Gemini)
                                              ↓
6. TTS Synthesis ←────────  7. Send Response (JSON + Audio)
```

**Error Handling:**
- Silent audio fallback if TTS fails
- Empty response if no speech detected
- Graceful disconnection handling

---

## AI Components

### 1. Speech-to-Text (STT)

**Manager**: [`STTManager`](src/AI/stt_manager.py)
- Uses Whisper model for Arabic transcription
- Processes audio files and returns Arabic text

### 2. Dialect Detection

**Manager**: [`DialectDetector`](src/AI/dialect_detector.py)
- Detects Arabic dialect from transcribed text
- Supports: Egyptian, Gulf, Levantine, Maghrebi, MSA
- Returns dialect + confidence score

### 3. Response Generation

**Manager**: [`ResponseGenerator`](src/AI/response_generator.py)
- Uses Gemini AI for generating responses
- Considers:
  - User query
  - Detected dialect
  - RAG context from knowledge base
  - Session conversation history

### 4. Text-to-Speech (TTS)

**Manager**: [`TTSManager`](src/AI/tts_manager.py)
- Synthesizes Arabic speech
- Supports multiple dialects
- Returns WAV audio bytes

### 5. Vector Database (RAG)

**Manager**: [`VectorDB`](src/RAGcontrollers/VectorDB.py)
- **Storage**: Qdrant vector database
- **Embeddings**: Gemini text-embedding-001
- **Operations**:
  - `insert_chunks()` - Index document chunks
  - `search()` - Semantic search for context

---

## Data Flow Example

### Typical Conversation Flow

```
Client                          Server
  │                                │
  │──── WebSocket Connect ────────▶│
  │      /ws/live-call/session123  │
  │                                │
  │──── Audio Chunk (binary) ─────▶│
  │      [Customer speaks]         │
  │                                │
  │                                │──▶ STT (Whisper)
  │                                │    "ما هو سعر الاشتراك"
  │                                │
  │                                │──▶ Dialect Detection
  │                                │    "egyptian" (0.95)
  │                                │
  │                                │──▶ Vector Search
  │                                │    Found pricing context
  │                                │
  │                                │──▶ Gemini Response
  │                                │    "سعر الاشتراك الشهري 299 جنية"
  │                                │
  │◀─── JSON Response ─────────────│
  │      {event, text, dialect}   │
  │                                │
  │◀─── Audio Response (binary) ───│
  │      [AI speaks]               │
  │                                │
  │──── [More Audio] ─────────────▶│
  │      [Continue conversation]   │
  │                                │
```

---

## Configuration

Environment variables (see [`src/helpers/config.py`](src/helpers/config.py)):

| Variable | Description |
|----------|-------------|
| `APP_NAME` | Application name |
| `APP_VERSION` | Version string |
| `QDRANT_HOST` | Qdrant server host |
| `QDRANT_PORT` | Qdrant server port |
| `VECTOR_COLLECTION_NAME` | Name of vector collection |
| `EMBEDDING_MODEL` | Gemini embedding model |
| `EMBEDDING_DIMENSION` | Embedding vector dimension |
| `GEMINI_API_KEY` | Gemini API key |
| `GROQ_API_KEY` | Groq API key (for Whisper) |

---

## iCloud Edits (New)

This section covers the new features and updates added to the AI Call Center system, including iCloud integration capabilities.

### New Features

- **iCloud Document Sync**: Documents uploaded to iCloud can now be indexed into the knowledge base
- **Cloud-based Session Storage**: Session data can be synced across devices using iCloud
- **Real-time Collaboration**: Multiple agents can now collaborate on the same session in real-time

### Configuration

To enable iCloud integration, add the following environment variables:

| Variable | Description |
|----------|-------------|
| `ICLOUD_ENABLED` | Enable/disable iCloud integration (true/false) |
| `ICLOUD_CONTAINER` | iCloud container name |

---

## Running fastRTC (Live Voice Call)

The project includes a fastRTC-based live voice call interface that provides a real-time voice conversation experience with the AI agent.

### Running the Live Call UI

```bash
cd src
python routers/websocket_endpoint.py
```

This will launch the fastRTC UI at: `http://127.0.0.1:8000`

### fastRTC Features

- **Real-time Audio Streaming**: Bidirectional audio streaming using fastRTC
- **Voice Activity Detection (VAD)**: Automatically detects when the user stops speaking
- **Automatic Speech Recognition**: Uses Whisper for transcription
- **Streaming TTS**: Responds with streaming audio synthesis
- **Web Interface**: Built-in web UI for testing the voice call

### WebRTC Configuration

The fastRTC stream is configured with:
- **Modality**: `audio` - handles audio-only communication
- **Mode**: `send-receive` - bidirectional audio streaming
- **Handler**: `ReplyOnPause` - triggers AI response when user pauses speaking

### Accessing the Live Call

Once running, open your browser to:
```
http://127.0.0.1:8000/ws/live-call
```

This provides a web-based interface to test the voice conversation with the AI agent.

---

## Running the API

### Development Mode

```bash
cd src
python main.py
```

Server runs at: `http://127.0.0.1:8000`

### Docker Mode

```bash
docker-compose up --build
```

---

## Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Dependencies

Key packages (see [`src/requirements.txt`](src/requirements.txt)):

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `qdrant-client` - Vector database client
- `google-genai` - Gemini API
- `whisper` - Speech recognition
- `soundfile` - Audio file handling
- `pygame` - Audio playback

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 500 | Internal Server Error |

---

## Notes

- WebSocket connections are session-based and persist until client disconnects
- Audio is processed in chunks (minimum 1.5 seconds)
- Dialect is locked after first detection and persists for the session
- Vector search returns top 1-5 relevant context chunks
- All audio uses 16kHz sample rate, float32 format
