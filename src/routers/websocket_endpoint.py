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
WELCOME_FILE = AUDIO_DIR / "welcome_gulf.wav"


# ── Singletons ───────────────────────────────────────────────────────────────
base_router = APIRouter()
vector_db = VectorDB()

dialect_detector = DialectDetector()  # keyword-only logging

sessions: dict[str, CallSession] = {}
live_sessions: dict[str, GeminiLiveManager] = {}


# ── Pre-load welcome WAV ─────────────────────────────────────────────────────
welcome_audio: tuple[int, np.ndarray] | None = None


def load_welcome_audio() -> None:
    global welcome_audio
    print("\n" + "=" * 60)
    print("📁 Loading welcome message...")
    print("=" * 60)

    if not WELCOME_FILE.exists():
        print("⚠️  Welcome file not found — skipping greeting")
        print("=" * 60 + "\n")
        return

    try:
        with wave.open(str(WELCOME_FILE), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

            # stereo → mono
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

            welcome_audio = (sr, audio)
            print(f"✅ Welcome loaded: {len(audio)/sr:.1f}s @ {sr} Hz")
    except Exception as e:
        print(f"❌ Welcome load error: {e}")

    print("=" * 60 + "\n")


load_welcome_audio()


# ── Session helpers ──────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> CallSession:
    if session_id not in sessions:
        s = CallSession(session_id)
        # Flags used by this router (kept out of CallSession to avoid breaking other imports)
        s.welcome_played = False
        s.audio_initialized = False
        sessions[session_id] = s
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
        profile = await asyncio.to_thread(vector_db.get_by_doc_type, "profile", 20)
        if profile:
            print(f"✅ Profile: Found {len(profile)} segment(s).")
        else:
            print("⚠️  Profile: No segments found (resume not tagged/indexed as doc_type=profile).")
    except Exception as e:
        print(f"⚠️  Profile fetch failed: {e}")

    try:
        kb = await asyncio.to_thread(vector_db.search, "خدمة عملاء دعم فني شحن رصيد باقات", 5)
        if kb:
            print(f"✅ KB: Found {len(kb)} segment(s).")
        else:
            print("⚠️  KB: No segments found.")
    except Exception as e:
        print(f"⚠️  KB seed failed (Qdrant down?): {e}")

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

async def process_call(audio_data: tuple[int, np.ndarray]):
    """Called by fastrtc ReplyOnPause each time the user finishes speaking."""

    SESSION_ID = "default_web_user"
    session = get_or_create_session(SESSION_ID)

    try:
        sr, audio = audio_data

        # 1) Prime browser AudioContext (fixes first-reply silence)
        if not getattr(session, "audio_initialized", False):
            session.audio_initialized = True
            print("\n🔧 Priming AudioContext...")
            silence = np.zeros(OUTPUT_SAMPLE_RATE // 10, dtype=np.int16).reshape(1, -1)  # 100ms
            yield (OUTPUT_SAMPLE_RATE, silence)
            await asyncio.sleep(0.1)
            print("   ✓ AudioContext primed\n")

        # 2) Play pre-recorded welcome once
        if not getattr(session, "welcome_played", False) and welcome_audio:
            session.welcome_played = True
            print("🎵 Playing welcome message...")
            w_sr, w_audio = welcome_audio
            chunk_size = OUTPUT_SAMPLE_RATE // 10  # 100ms
            for i in range(0, len(w_audio), chunk_size):
                chunk = w_audio[i : i + chunk_size]
                if len(chunk):
                    yield (w_sr, chunk.reshape(1, -1))
            print("   ✓ Welcome complete\n")

        # 3) Ensure Live session exists
        live = await get_or_create_live_session(SESSION_ID, session)
        if live is None:
            return

        # 4) Send audio → stream Gemini audio response
        print(f"🎙️  User spoke ({len(audio)/sr:.1f}s) — sending to Gemini Live...")
        chunks = 0
        async for out_sr, out_audio in live.send_and_receive(audio, sr):
            yield (out_sr, out_audio)
            chunks += 1
        print(f"   ✓ Streamed {chunks} audio chunks back to browser")

        # 5) Transcripts for UI/logging (optional)
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
        if out_text:
            print(f"📝 Output transcript: {out_text}")

        yield AdditionalOutputs({"text": out_text or "", "event": "ai_reply"})

    except Exception as e:
        print(f"\n❌ process_call error: {e}")
        import traceback

        traceback.print_exc()


# ── FastRTC Stream ───────────────────────────────────────────────────────────

stream = Stream(
    handler=ReplyOnPause(process_call),
    modality="audio",
    mode="send-receive",
    additional_outputs_handler=lambda: {"text": "", "event": ""},
)
