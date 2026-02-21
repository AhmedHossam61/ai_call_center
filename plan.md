# AI Call Center — Full Enhancement Implementation Plan

> **Purpose**: This document is a complete, actionable roadmap for the next
> model / developer that will continue the project. Each section describes
> the current state, the problem it solves, the exact changes to make, and
> how to validate the result.

---

## Table of Contents

1. [Welcome Audio on Connect (DONE)](#1-welcome-audio-on-connect)
2. [Latency / Delay Optimisations](#2-latency--delay-optimisations)
3. [RAG Enhancements](#3-rag-enhancements)
4. [Session & Connection Management](#4-session--connection-management)
5. [Testing Strategy](#5-testing-strategy)
6. [Deployment Checklist](#6-deployment-checklist)

---

## 1. Welcome Audio on Connect

### Current state (before this PR)
The welcome WAV was played inside `process_call()`, which is only triggered by
`ReplyOnPause` **after** the user speaks and pauses. The user heard silence until
they said something first.

### What was changed (this PR)
`ReplyOnPause` exposes a `startup_fn` parameter that is called **once per WebRTC
connection** (before any user audio arrives). A new async generator
`welcome_startup()` in `routers/websocket_endpoint.py`:

1. Yields 100 ms of silence to prime the browser AudioContext (prevents the
   first-chunk silence bug on Chrome/Safari).
2. Streams `AI/audio_files/welcome_msa.wav` in 100 ms chunks to the caller.

The welcome logic was removed from `process_call()` to avoid double-play and to
keep the main handler focused on Gemini turn-taking.

### Files changed
| File | Change |
|------|--------|
| `src/routers/websocket_endpoint.py` | Added `welcome_startup()`, removed welcome/priming from `process_call()`, updated `Stream` to pass `startup_fn=welcome_startup` |

### Validation
- Start the server; open the Gradio UI at `http://localhost:8001`.
- Click **Start** (begin WebRTC session).
- You should **immediately** hear the welcome message without speaking.

---

## 2. Latency / Delay Optimisations

### 2.1 Pre-open Gemini Live Session on Connect

**Problem**: The first user utterance triggers `get_or_create_live_session()`,
which does two Qdrant queries + a Gemini Live WebSocket handshake **during** the
user's first pause. This adds ~1–3 s of visible latency on the first turn.

**Fix**: Pre-open the Gemini Live session inside `welcome_startup()` (or in a
background task launched right after the WebRTC handshake) so it is ready before
the user speaks.

```python
# In welcome_startup() — after playing the welcome WAV:
SESSION_ID = "default_web_user"
session = get_or_create_session(SESSION_ID)
asyncio.ensure_future(get_or_create_live_session(SESSION_ID, session))
```

**Files**: `src/routers/websocket_endpoint.py`

---

### 2.2 Parallel RAG Seed at Session Creation

**Problem**: `get_or_create_live_session()` fetches `profile` and then `kb`
sequentially with `await asyncio.to_thread(...)`.

**Fix**: Run both queries concurrently with `asyncio.gather`.

```python
# Before
profile = await asyncio.to_thread(vector_db.get_by_doc_type, "profile", 20)
kb      = await asyncio.to_thread(vector_db.search, "...", 5)

# After
profile, kb = await asyncio.gather(
    asyncio.to_thread(vector_db.get_by_doc_type, "profile", 20),
    asyncio.to_thread(vector_db.search, "خدمة عملاء دعم فني شحن رصيد باقات", 5),
)
```

**Files**: `src/routers/websocket_endpoint.py`

---

### 2.3 Embedding Cache Pre-warm at Server Startup

**Problem**: `VectorDB.prewarm_cache()` exists but is never called from the
server startup, so the first embedding call (during the first RAG query) incurs
a cold-start round-trip to the Gemini Embeddings API.

**Fix**: Call `prewarm_cache()` in the FastAPI `lifespan` startup hook.

```python
# src/main.py  — inside lifespan(), before yield
from RAGcontrollers.VectorDB import vector_db as _vdb
await asyncio.to_thread(_vdb.prewarm_cache)
```

**Files**: `src/main.py`

---

### 2.4 Reduce Gemini Live Audio Tail Padding

**Problem**: `gemini_live_manager.py` appends 250 ms of silence after every
utterance:

```python
tail = np.zeros(int(INPUT_SAMPLE_RATE * 0.25), dtype=np.int16)
```

This tells the Live API to wait 250 ms before generating a reply, adding
perceivable lag.

**Fix**: Reduce to 100 ms (or 0 ms if Gemini handles it cleanly).

```python
tail = np.zeros(int(INPUT_SAMPLE_RATE * 0.10), dtype=np.int16)  # 100 ms
```

Validate by checking that Gemini does not cut off the last syllable. If it does,
keep at 150 ms.

**Files**: `src/AI/gemini_live_manager.py`

---

### 2.5 Enable Gemini Output Audio Transcription

**Problem**: `_build_config()` includes no transcription config, so
`last_turn.output_text` is usually empty, making logging/debugging hard.
Transcription is free for the Live API and does not add latency.

**Fix**: Add output transcription to the session config.

```python
return {
    "response_modalities": ["AUDIO"],
    "system_instruction": system_instruction,
    "speech_config": { ... },
    "output_audio_transcription": {},   # ← add this
}
```

**Files**: `src/AI/gemini_live_manager.py`

---

### 2.6 Streamed Chunk Size Tuning

**Problem**: `OUTPUT_CHUNK_SIZE = 2400` (100 ms @ 24 kHz) is a reasonable
default, but delivering smaller chunks (e.g. 1200 = 50 ms) reduces the
perceived first-token latency because the browser receives and plays audio
sooner.

**Fix**: Lower `OUTPUT_CHUNK_SIZE` to `1200` and measure the improvement.
Revert if it causes stuttering.

```python
OUTPUT_CHUNK_SIZE = 1200  # 50 ms @ 24 kHz
```

**Files**: `src/AI/gemini_live_manager.py`

---

## 3. RAG Enhancements

### 3.1 Per-Turn Dynamic RAG Retrieval

**Problem**: RAG context is injected **once** into the Gemini system prompt at
session creation time using a generic seed query. The context never updates
during the conversation even though the user may ask about topics not covered by
the initial seed.

**Fix**: After each user utterance (detected via `in_text` transcript), run a
dynamic Qdrant search and send the result as a **user-turn text message** before
the audio response:

```python
# In process_call(), after receiving in_text:
if in_text:
    extra_context = await vector_db.search_async(in_text, limit=3)
    if extra_context:
        ctx_msg = "معلومة إضافية:\n" + "\n".join(extra_context)
        await live._session.send(input=ctx_msg)
```

This uses the Gemini Live API's ability to receive text turns interleaved with
audio.

**Files**: `src/routers/websocket_endpoint.py`, `src/AI/gemini_live_manager.py`

---

### 3.2 Semantic Chunking

**Problem**: `DataProcessing` uses `RecursiveCharacterTextSplitter` with fixed
`chunk_size=1000`. Arabic text often has sentences that exceed or are far below
this limit, leading to broken context at chunk boundaries.

**Fix**: Switch to `semchunk` (already in `requirements.txt`) or LangChain's
`SemanticChunker` for meaning-aware splitting:

```python
from semchunk import chunk

chunks = chunk(
    text=markdown_content,
    chunk_size=800,
    token_counter=lambda t: len(t.split()),  # word-level approximation
)
```

Alternatively, use `langchain_experimental.text_splitter.SemanticChunker` with a Gemini
embedding function as the similarity model.

**Files**: `src/RAGcontrollers/DataProcessing.py`

---

### 3.3 Metadata-Enriched Chunks

**Problem**: Chunks lack `doc_type` and `doc_id` when uploaded through the REST
endpoint (only the seed script tags them). This means `get_by_doc_type()` finds
nothing for documents uploaded at runtime.

**Fix**: When saving a file via the upload API, prompt the user for `doc_type`
(e.g. `company_kb`, `profile`, `faq`) and store it in chunk metadata:

```python
# In base.py upload handler:
doc_type = form_data.get("doc_type", "company_kb")
for chunk in processed_chunks:
    chunk["metadata"]["doc_type"] = doc_type
```

**Files**: `src/routers/base.py`, `src/RAGcontrollers/DataProcessing.py`

---

### 3.4 Hybrid Search (Dense + BM25)

**Problem**: Pure cosine-similarity search sometimes misses exact keyword
matches important in customer-service queries (e.g. specific plan names,
product codes).

**Fix**: Enable Qdrant's built-in sparse-vector support (BM25) and query with
`fusion=RRF` (Reciprocal Rank Fusion).

```python
# Requires Qdrant ≥ 1.7 and re-indexing with sparse vectors
from qdrant_client.http.models import SparseVectorParams, SparseIndexParams

# At collection creation:
self.qdrant_client.create_collection(
    collection_name=...,
    vectors_config={"dense": models.VectorParams(size=expected_dim, distance=COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
)

# At query time, build both dense and sparse vectors and use query_points with RRF.
```

**Files**: `src/RAGcontrollers/VectorDB.py`

---

### 3.5 Result Re-ranking

**Problem**: The top-k Qdrant results are passed to Gemini in their raw
retrieval order, which may not match relevance for the specific question.

**Fix**: After Qdrant retrieval, re-rank with a cross-encoder. A lightweight
Arabic-compatible option is `cross-encoder/ms-marco-MiniLM-L-6-v2` (with
transliteration) or the `Cohere` Rerank API.

```python
# Pseudo-code
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query_text, doc) for doc in extracted_texts]
scores = reranker.predict(pairs)
ranked = [doc for _, doc in sorted(zip(scores, extracted_texts), reverse=True)]
```

**Files**: `src/RAGcontrollers/VectorDB.py`

---

### 3.6 Embedding Cache Eviction by LRU

**Problem**: `_embed_cache` evicts the oldest entry (insertion order) when
full. A Least-Recently-Used (LRU) policy keeps frequently accessed embeddings
and discards stale ones.

**Fix**: Replace `dict` with `functools.lru_cache` or `cachetools.LRUCache`.

```python
from cachetools import LRUCache
self._embed_cache: LRUCache = LRUCache(maxsize=500)
```

**Files**: `src/RAGcontrollers/VectorDB.py`

---

## 4. Session & Connection Management

### 4.1 Per-Connection Session IDs

**Problem**: All callers share `SESSION_ID = "default_web_user"`. If two
browser tabs connect simultaneously, they share conversation history and the same
Gemini Live session, causing cross-contamination.

**Fix**: `ReplyOnPause` with `needs_args=True` passes the Gradio `webrtc_id`
as an additional argument. Use it as the session key.

```python
stream = Stream(
    handler=ReplyOnPause(process_call, startup_fn=welcome_startup, needs_args=True),
    ...
)

async def process_call(audio_data, webrtc_id: str = "default"):
    SESSION_ID = webrtc_id
    ...
```

**Files**: `src/routers/websocket_endpoint.py`

---

### 4.2 Session Cleanup on Disconnect

**Problem**: When the browser closes the WebRTC connection, `live_sessions` and
`sessions` are never cleaned up, causing memory leaks in long-running servers.

**Fix**: Use the FastAPI `lifespan` or a background task that periodically
removes sessions where `mgr.is_connected() == False`.

```python
# Background cleanup (add to lifespan):
async def cleanup_sessions():
    while True:
        await asyncio.sleep(60)
        for sid in list(live_sessions):
            if not live_sessions[sid].is_connected():
                await live_sessions.pop(sid).close()
                sessions.pop(sid, None)
asyncio.ensure_future(cleanup_sessions())
```

**Files**: `src/main.py` or `src/routers/websocket_endpoint.py`

---

## 5. Testing Strategy

Since the project currently has no automated tests, the following tests should
be added to a `tests/` directory using `pytest` + `pytest-asyncio`.

| Test | What it validates |
|------|-------------------|
| `test_welcome_startup.py` | `welcome_startup()` yields at least one audio chunk |
| `test_session.py` | `get_or_create_session` creates a fresh `CallSession` |
| `test_vector_db.py` | `VectorDB.search()` returns a list when Qdrant is up |
| `test_data_processing.py` | `process_single_file()` produces non-empty chunks for a sample PDF |
| `test_gemini_live_manager.py` | `_build_config()` returns required keys; resample/convert helpers work |

Mock Qdrant with `unittest.mock.patch` and Gemini with a stub client to avoid
network calls in CI.

---

## 6. Deployment Checklist

- [ ] Run `python src/generate_audio_files.py` to generate fresh WAV files after
  changing the welcome text or voice.
- [ ] Set `GEMINI_API_KEY`, `QDRANT_HOST`, `QDRANT_PORT`, `VECTOR_COLLECTION_NAME`,
  `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` in `.env` (see `env.example`).
- [ ] Start Qdrant: `docker-compose up -d qdrant`
- [ ] Seed knowledge base: upload documents via `POST /base/upload`.
- [ ] Start the app: `docker-compose up --build app`
- [ ] Verify welcome plays on connect: open `http://localhost:8001` in a browser.
- [ ] Check Swagger: `http://localhost:8000/docs`
