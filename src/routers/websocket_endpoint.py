"""routers/websocket_endpoint.py

Gemini Live API demo handler.

Replaces the STT → LLM → TTS pipeline with a single persistent
Gemini Live WebSocket session per browser caller.

Key ideas:
- fastrtc ReplyOnPause already segments utterances → we disable Live VAD and
  send activity_start/activity_end.
- Stream 24 kHz PCM back to the browser immediately.
- Optional pre-recorded welcome WAV.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastrtc import Stream, ReplyOnPause, AdditionalOutputs

from AI.session import CallSession
from AI.dialect_detector import DialectDetector
from AI.gemini_live_manager import GeminiLiveManager, OUTPUT_SAMPLE_RATE
from RAGcontrollers.VectorDB import VectorDB


# ── Constants ────────────────────────────────────────────────────────────────
AUDIO_DIR = Path(__file__).parent.parent / "AI" / "audio_files"
WELCOME_FILE = AUDIO_DIR / "welcome_egyptian.wav"
HOLD_FILE    = AUDIO_DIR / "hold_music.wav"


# ── Singletons ───────────────────────────────────────────────────────────────
base_router = APIRouter()
vector_db = VectorDB()

dialect_detector = DialectDetector()  # keyword-only logging

sessions: dict[str, CallSession] = {}
live_sessions: dict[str, GeminiLiveManager] = {}
pending_context: dict[str, str] = {}  # per-session RAG context ready for next turn


# ── Pre-load welcome WAV ─────────────────────────────────────────────────────
welcome_audio: tuple[int, np.ndarray] | None = None
hold_music_audio: tuple[int, np.ndarray] | None = None


def _load_wav(path: Path, label: str) -> tuple[int, np.ndarray] | None:
    """Load a mono int16 WAV. Returns (sample_rate, samples) or None."""
    if not path.exists():
        print(f"⚠️  {label} not found at {path} — skipping")
        return None
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        print(f"✅ {label} loaded: {len(audio)/sr:.1f}s @ {sr} Hz")
        return (sr, audio)
    except Exception as e:
        print(f"❌ {label} load error: {e}")
        return None


print("\n" + "=" * 60)
print("📁 Loading audio files...")
print("=" * 60)
welcome_audio    = _load_wav(WELCOME_FILE, "Welcome")
hold_music_audio = _load_wav(HOLD_FILE,    "Hold music")
print("=" * 60 + "\n")


# ── Hold-music helper ─────────────────────────────────────────────────────────

async def _stream_with_hold_music(
    live: GeminiLiveManager,
    audio: np.ndarray,
    sample_rate: int,
):
    """Run send_and_receive in a background task.
    Yields hold-music chunks until the first Gemini audio chunk arrives,
    then transparently switches to streaming Gemini audio.
    No extra latency is added — Gemini processing starts immediately.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def _feed():
        try:
            async for item in live.send_and_receive(audio, sample_rate):
                await queue.put(item)
        finally:
            await queue.put(None)  # sentinel

    feed_task = asyncio.ensure_future(_feed())

    HOLD_CHUNK = OUTPUT_SAMPLE_RATE // 10  # 100 ms
    hold_pos = 0
    gemini_started = False

    try:
        while True:
            # Phase 1: hold music until first Gemini chunk
            if not gemini_started:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    if hold_music_audio is not None:
                        h_sr, h_audio = hold_music_audio
                        chunk = h_audio[hold_pos : hold_pos + HOLD_CHUNK]
                        if len(chunk) == 0:        # loop
                            hold_pos = 0
                            chunk = h_audio[:HOLD_CHUNK]
                        hold_pos = (hold_pos + HOLD_CHUNK) % max(len(h_audio), 1)
                        if len(chunk):
                            yield (h_sr, chunk.reshape(1, -1))
                    await asyncio.sleep(0.05)  # 50 ms poll
                    continue

                if item is None:   # Gemini returned nothing
                    break
                gemini_started = True
                yield item        # first real chunk

            # Phase 2: drain remaining Gemini chunks (blocking)
            else:
                item = await queue.get()
                if item is None:
                    break
                yield item
    finally:
        feed_task.cancel()

async def welcome_startup(webrtc_id: str = "default_web_user"):
    """Async generator yielded by ReplyOnPause.startup_fn on each new WebRTC
    connection.  Sends a 100 ms silence to prime the browser AudioContext, then
    streams the pre-recorded welcome WAV — all before the user speaks a word.
    """
    # Prime browser AudioContext (fixes first-reply silence on Chrome/Safari)
    silence = np.zeros(OUTPUT_SAMPLE_RATE // 10, dtype=np.int16)
    yield (OUTPUT_SAMPLE_RATE, silence.reshape(1, -1))

    if welcome_audio is None:
        return

    print("🎵 Playing welcome message on connection...")
    w_sr, w_audio = welcome_audio
    chunk_size = OUTPUT_SAMPLE_RATE // 10  # 100 ms chunks
    for i in range(0, len(w_audio), chunk_size):
        chunk = w_audio[i : i + chunk_size]
        if len(chunk):
            yield (w_sr, chunk.reshape(1, -1))
    print("   ✓ Welcome complete\n")

    # Pre-open Gemini Live session so it's ready before the user speaks
    _pre_session = get_or_create_session(webrtc_id)
    asyncio.ensure_future(get_or_create_live_session(webrtc_id, _pre_session))


def get_or_create_session(session_id: str) -> CallSession:
    if session_id not in sessions:
        sessions[session_id] = CallSession(session_id)
    return sessions[session_id]


async def get_or_create_live_session(session_id: str, session: CallSession) -> GeminiLiveManager | None:
    # Reuse an existing session if it's still connected
    mgr = live_sessions.get(session_id)
    if mgr and mgr.is_connected():
        return mgr

    # If it exists but is disconnected, drop it and recreate
    if mgr:
        try:
            await mgr.close()
        except Exception:
            pass
        live_sessions.pop(session_id, None)

    print(f"\n🔗 Creating Gemini Live session for [{session_id}]...")

    # Seed system prompt with BOTH: resume/profile + company KB
    profile, kb = [], []

    try:
        profile, kb = await asyncio.gather(
            asyncio.to_thread(vector_db.get_by_doc_type, "profile", 20),
            asyncio.to_thread(vector_db.search, "خدمة عملاء دعم فني شحن رصيد باقات", 5),
        )
        if profile:
            print(f"✅ Profile: Found {len(profile)} segment(s).")
        else:
            print("⚠️  Profile: No segments found (resume not tagged/indexed as doc_type=profile).")
        if kb:
            print(f"✅ KB: Found {len(kb)} segment(s).")
        else:
            print("⚠️  KB: No segments found.")
    except Exception as e:
        print(f"⚠️  RAG seed failed: {e}")

    rag_context = ""
    if profile:
        rag_context += "=== ملف العميل (السيرة الذاتية) ===\n" + "\n\n".join(profile) + "\n\n"
    if kb:
        rag_context += "=== قاعدة معرفة الشركة ===\n" + "\n\n".join(kb)

    if rag_context:
        print("📚 RAG: injected profile + KB into session")
    else:
        print("⚠️  RAG: context is empty (Gemini will not know you yet)")

    mgr = GeminiLiveManager(dialect=session.active_dialect, rag_context=rag_context)
    if not await mgr.connect():
        print("❌ Gemini Live session failed — cannot process audio")
        return None

    live_sessions[session_id] = mgr
    return mgr

# ── Main call handler ─────────────────────────────────────────────────────────

async def process_call(audio_data: tuple[int, np.ndarray], webrtc_id: str = "default_web_user"):
    """Called by fastrtc ReplyOnPause each time the user finishes speaking."""

    SESSION_ID = webrtc_id
    session = get_or_create_session(SESSION_ID)

    try:
        sr, audio = audio_data

        # 1) Ensure Live session exists
        live = await get_or_create_live_session(SESSION_ID, session)
        if live is None:
            return

        # 1b) Inject any RAG context queued from the previous turn
        if SESSION_ID in pending_context and live.is_connected():
            try:
                await live._session.send(input=pending_context.pop(SESSION_ID))
                print(f"📚 Pre-turn RAG context injected")
            except Exception as pre_err:
                print(f"⚠️  Pre-turn RAG inject failed: {pre_err}")
                pending_context.pop(SESSION_ID, None)

        # 2) Send audio → stream Gemini audio response (hold music fills the gap)
        print(f"🎵🎵🎵  User spoke ({len(audio)/sr:.1f}s) — sending to Gemini Live...")
        chunks = 0
        async for out_sr, out_audio in _stream_with_hold_music(live, audio, sr):
            yield (out_sr, out_audio)
            chunks += 1
        print(f"   ✓ Streamed {chunks} audio chunks back to browser")

        # 3) Transcripts for UI/logging (optional)
        in_text = live.last_turn.input_text
        out_text = live.last_turn.output_text

        if in_text:
            fast = getattr(dialect_detector, "_fast_detect", None)
            if callable(fast):
                res = fast(in_text)
                if res:
                    dialect, conf = res
                    session.lock_dialect(dialect, conf)
                    print(f"🌍 Dialect (keyword): {dialect} ({conf:.0%})")
            print(f"📝 Input transcript: {in_text}")

            # Dynamic RAG: search for context relevant to this utterance,
            # store it as pending so it's injected BEFORE the next turn's audio.
            try:
                extra_context = await vector_db.search_async(in_text, limit=3)
                if extra_context:
                    pending_context[SESSION_ID] = "معلومة إضافية:\n" + "\n".join(extra_context)
                    print(f"📚 Dynamic RAG: queued {len(extra_context)} segment(s) for next turn")
            except Exception as rag_err:
                print(f"⚠️  Dynamic RAG failed: {rag_err}")

        if out_text:
            print(f"📝 Output transcript: {out_text}")

        yield AdditionalOutputs({"text": out_text or "", "event": "ai_reply"})

    except Exception as e:
        print(f"\n❌ process_call error: {e}")
        import traceback

        traceback.print_exc()


# ── FastRTC Stream ───────────────────────────────────────────────────────────

stream = Stream(
    handler=ReplyOnPause(process_call, startup_fn=welcome_startup, needs_args=True),
    modality="audio",
    mode="send-receive",
    additional_outputs_handler=lambda: {"text": "", "event": ""},
)
