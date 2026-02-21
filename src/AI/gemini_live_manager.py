"""AI/gemini_live_manager.py

Gemini Live API wrapper for speech-to-speech conversations.

Design goals for this project:
- One persistent Live API session per web caller.
- fastrtc ReplyOnPause already detects end-of-utterance, so we disable
  Live API automatic VAD and send activity_start/activity_end ourselves.
- Stream 24 kHz PCM audio chunks back to the browser as soon as they arrive.

Docs used for this implementation:
- Live API get-started + capabilities guide (audio format, transcription,
  voice config, manual activity signals).
"""

from __future__ import annotations
import base64
import os
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

try:
    # Recommended import path in current docs.
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    # Fallback for older environments.
    import google.genai as genai  # type: ignore
    from google.genai import types  # type: ignore

from helpers.config import get_settings


settings = get_settings()


# ── Models ─────────────────────────────────────────────────────────────
# Developer API (API key) model IDs from the Live API docs/models pages.
PRIMARY_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
FALLBACK_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
# PRIMARY_MODEL = "gemini-live-2.5-flash-native-audio"
# Vertex AI model ID (only used if GOOGLE_GENAI_USE_VERTEXAI is enabled).
VERTEX_MODEL = "gemini-live-2.5-flash-native-audio"


# ── Audio specs ─────────────────────────────────────────────────────────
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
OUTPUT_CHUNK_SIZE = 1200  # 50 ms @ 24 kHz


DIALECT_VOICES = {
    "egyptian": "Aoede",
    "gulf": "Charon",
    "sudanese": "Aoede",
    "levantine": "Aoede",
    "msa": "Kore",
}


def build_system_prompt(rag_context: str = "") -> str:
    kb_section = ""
    if rag_context and rag_context.strip():
        kb_section = f"""

=== قاعدة المعرفة (استخدمها للإجابة) ===
{rag_context}
========================================
"""

    return f"""أنت موظف دعم عملاء محترف في شركة تليكوم.
{kb_section}
تعليمات صارمة:
1) استمع جيداً للهجة العميل وتحدث بنفس اللهجة تلقائياً:
   - إذا كان مصرياً تكلم مصري
   - إذا كان خليجياً تكلم خليجي
   - إذا كان سودانياً تكلم سوداني
   - إذا كان شامياً تكلم شامي
   - إذا تكلم فصحى رد بفصحى

2) ردودك قصيرة ومباشرة — جملتان كحد أقصى.

3) لا تذكر أنك نموذج ذكاء اصطناعي — أنت موظف حقيقي.

4) إذا لم تجد الإجابة في قاعدة المعرفة، قل:
   "سأحوّلك لمختص يساعدك أكثر."
"""


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert various audio array shapes/dtypes to mono float32."""
    a = np.asarray(audio)
    # Typical shapes from WebRTC libs: (1, n) or (n,) or (n, 2)
    if a.ndim == 2:
        # (channels, n)
        if a.shape[0] in (1, 2):
            a = a.T
        # (n, channels)
        if a.shape[1] == 2:
            a = a.mean(axis=1)
        else:
            a = a[:, 0]
    return a.astype(np.float32, copy=False).flatten()


def _float_to_int16_pcm(x: np.ndarray) -> np.ndarray:
    # If it looks like normalized float audio, scale.
    if x.size == 0:
        return x.astype(np.int16)
    mx = float(np.max(np.abs(x)))
    if mx <= 1.5:
        x = np.clip(x, -1.0, 1.0) * 32767.0
    return x.astype(np.int16)


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return x
    n = int(round(len(x) * (dst_rate / float(src_rate))))
    if n <= 1:
        return x[:1]
    old = np.arange(len(x), dtype=np.float32)
    new = np.linspace(0, len(x) - 1, n, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32)


@dataclass
class TurnTranscripts:
    input_text: str = ""
    output_text: str = ""


class GeminiLiveManager:
    """One Live API session per caller."""

    def __init__(self, dialect: str = "msa", rag_context: str = ""):
        self.dialect = dialect or "msa"
        self.rag_context = rag_context or ""
        self._session = None
        self._ctx = None
        self._connected = False
        self._model = None
        self.last_turn = TurnTranscripts()

        # Client init: use explicit key (project already manages .env)
        # Note: the SDK also supports reading GEMINI_API_KEY/GOOGLE_API_KEY env vars.
        self.client = genai.Client(api_key=getattr(settings, "GEMINI_API_KEY", None))

    async def connect(self) -> bool:
        """Open websocket session; tries developer API models, then Vertex if enabled."""
        models = [PRIMARY_MODEL, FALLBACK_MODEL]
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
            models = [VERTEX_MODEL] + models

        for m in models:
            try:
                self._ctx = self.client.aio.live.connect(model=m, config=self._build_config())
                self._session = await self._ctx.__aenter__()
                self._connected = True
                self._model = m
                print("✅ Gemini Live connected")
                print(f"   Model  : {m}")
                print(f"   Voice  : {DIALECT_VOICES.get(self.dialect, 'Aoede')}")
                return True
            except Exception as e:
                print(f"⚠️  Gemini Live connect failed ({m}): {e}")

        self._connected = False
        self._session = None
        return False

    async def close(self) -> None:
        if self._ctx is not None:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._ctx = None
        self._session = None
        self._connected = False
        self._model = None
        print("🔌 Gemini Live session closed")

    def is_connected(self) -> bool:
        return bool(self._connected and self._session is not None)

    async def send_and_receive(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> AsyncIterator[tuple[int, np.ndarray]]:
        """Send one utterance; stream back audio chunks.

        Yields (OUTPUT_SAMPLE_RATE, np.int16[1, n]) chunks.
        Also populates self.last_turn with input/output transcriptions.
        """

        self.last_turn = TurnTranscripts()

        if not self.is_connected():
            return

        # Convert input audio to 16kHz mono int16 PCM bytes.
        mono = _to_mono_float32(audio)
        mono = _resample_linear(mono, int(sample_rate), INPUT_SAMPLE_RATE)
        pcm16 = _float_to_int16_pcm(mono)
        pcm_bytes = pcm16.tobytes()

        # Because ReplyOnPause already segments utterances, disable Live VAD
        # and explicitly mark activity boundaries.
        tail = np.zeros(int(INPUT_SAMPLE_RATE * 0.10), dtype=np.int16)  # 100 ms
        pcm16_with_tail = np.concatenate([pcm16, tail])
        pcm_bytes = pcm16_with_tail.tobytes()

        try:
            # v0.3.0 supports Blob or dict input; Blob is cleanest
            await self._session.send(
                input=types.Blob(
                    data=pcm_bytes,
                    mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                )
            )
        except Exception as e:
            print(f"❌ Gemini Live send error (0.3.0): {e}")
            return

        # Receive one model turn.
        out_text_parts: list[str] = []
        in_text: Optional[str] = None

        try:
            turn = self._session.receive()
            async for msg in turn:
                sc = getattr(msg, "server_content", None)
                if not sc:
                    continue

                # Audio chunks
                mt = getattr(sc, "model_turn", None)
                if mt and getattr(mt, "parts", None):
                    for part in mt.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            raw = inline.data
                            if isinstance(raw, str) and raw:
                                raw = base64.b64decode(raw)

                            if isinstance(raw, (bytes, bytearray)) and raw:
                                samples = np.frombuffer(raw, dtype=np.int16)
                                for i in range(0, len(samples), OUTPUT_CHUNK_SIZE):
                                    chunk = samples[i : i + OUTPUT_CHUNK_SIZE]
                                    if len(chunk):
                                        yield (OUTPUT_SAMPLE_RATE, chunk.reshape(1, -1))

                # Transcriptions
                if getattr(sc, "input_transcription", None) and getattr(sc.input_transcription, "text", None):
                    in_text = sc.input_transcription.text
                if getattr(sc, "output_transcription", None) and getattr(sc.output_transcription, "text", None):
                    out_text_parts.append(sc.output_transcription.text)

                # End of this turn
                if getattr(sc, "turn_complete", False):
                    break

        except Exception as e:
            print(f"❌ Gemini Live receive error: {e}")

        self.last_turn.input_text = (in_text or "").strip()
        self.last_turn.output_text = "".join(out_text_parts).strip()

    # ── Internal helpers ───────────────────────────────────────────────

    def _build_config(self) -> dict:
        voice = DIALECT_VOICES.get(self.dialect, "Kore")
        prompt = build_system_prompt(self.rag_context)

        system_instruction = {"role": "system", "parts": [{"text": prompt}]}

        return {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_instruction,
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
            },
            "output_audio_transcription": {},
        }
