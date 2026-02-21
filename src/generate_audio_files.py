"""
Generate welcome messages and hold music using Gemini TTS.

Uses the same voice (Aoede) and style prompts as the live TTS,
ensuring consistent tone throughout the call.

Usage:
    python AI/generate_audio_files.py

Requirements:
    - GEMINI_API_KEY in .env
    - Same voice configured in tts_manager.py

All files saved to AI/audio_files/
"""

import asyncio
import sys
import wave
import io
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import google.genai as genai
from google.genai import types
from helpers.config import get_settings

settings = get_settings()

# Output directory
AUDIO_DIR = Path(__file__).parent / "audio_files"
AUDIO_DIR.mkdir(exist_ok=True)

# Gemini TTS configuration (must match tts_manager.py)
GEMINI_MODEL = "gemini-2.5-pro-preview-tts"
GEMINI_VOICE = "Aoede"   # Change this if you changed it in tts_manager.py
SAMPLE_RATE  = 24000

# Style prompts per dialect (must match tts_manager.py)
STYLE_PROMPTS = {
    "egyptian": "Speak with a warm, friendly Egyptian Cairo accent. Natural conversational Egyptian intonation.",
    "gulf": "Speak with a clear Gulf Arabic accent. Respectful and professional Khaleeji tone.",
    "levantine": "Speak with a natural Levantine Arabic accent. Warm Syrian or Lebanese intonation.",
    "moroccan": "Speak with a natural Moroccan Darija accent. Friendly and clear tone.",
    "msa": "Speak in clear Modern Standard Arabic (Fusha). Professional and neutral tone.",
}

# Welcome messages per dialect
WELCOME_MESSAGES = {
    "egyptian": "أهلاً وسهلاً، أنا أحمد من شركة telnova Solutions . ازيك؟ عايز أساعدك في إيه؟",
    "gulf": "يا هلا وسهلا، أنا أحمد من telnova Solutions. كيف حالك؟ شلون أقدر أخدمك؟",
    "levantine": "أهلاً وسهلاً، أنا أحمد من telnova Solutions. كيفك؟ شو ممكن ساعدك؟",
    "moroccan": "مرحبا، أنا أحمد من telnova Solutions. كيداير؟ شنو نقدر نعاونك؟",
    "msa": "أهلاً وسهلاً، أنا أحمد من telnova Solutions. كيف يمكنني مساعدتك اليوم؟",
}

# Hold music message (neutral MSA)
HOLD_MESSAGE = "شكراً لانتظارك، سيتم الرد عليك في أقرب وقت ممكن."


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer.read()


async def generate_gemini_tts(text: str, dialect: str, output_path: Path):
    """Generate TTS audio using Gemini and save as WAV."""
    print(f"  Generating: {output_path.name}...")

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Build prompt with style instruction
        style = STYLE_PROMPTS.get(dialect, STYLE_PROMPTS["msa"])
        full_prompt = f"{style}\n\nText to speak: {text}"

        # Call Gemini TTS (synchronous SDK)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=GEMINI_VOICE
                        )
                    )
                )
            )
        )

        # Extract raw PCM bytes
        pcm_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                pcm_bytes = part.inline_data.data
                break

        if not pcm_bytes:
            print(f"    ❌ No audio data returned")
            return

        # Convert PCM → WAV
        wav_bytes = pcm_to_wav(pcm_bytes, SAMPLE_RATE)

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(wav_bytes)

        # Calculate duration
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        duration = len(samples) / SAMPLE_RATE

        print(f"    ✓ {output_path.name} ({duration:.1f}s, voice: {GEMINI_VOICE})")

    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("=" * 70)
    print("🎙️  AI Call Center Audio Generator (Gemini TTS)")
    print("=" * 70)
    print(f"Voice: {GEMINI_VOICE}")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Output: {AUDIO_DIR.absolute()}\n")

    # Check API key
    if not settings.GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env")
        print("   Add: GEMINI_API_KEY=your_key_here")
        return

    # Generate welcome messages for all dialects
    print("📢 Generating welcome messages...")
    for dialect, text in WELCOME_MESSAGES.items():
        output = AUDIO_DIR / f"welcome_{dialect}.wav"
        await generate_gemini_tts(text, dialect, output)

    # Generate hold music (MSA, neutral)
    print("\n🎶 Generating hold music...")
    output = AUDIO_DIR / "hold_music.wav"
    await generate_gemini_tts(HOLD_MESSAGE, "msa", output)

    print("\n" + "=" * 70)
    print("✅ All audio files generated successfully!")
    print("=" * 70)
    print(f"\nFiles saved to: {AUDIO_DIR.absolute()}")
    print(f"\n📊 Summary:")
    print(f"   - 5 welcome messages (one per dialect)")
    print(f"   - 1 hold music file")
    print(f"   - Voice: {GEMINI_VOICE}")
    print(f"   - Sample rate: {SAMPLE_RATE} Hz")
    
    print("\n💡 To customize:")
    print(f"   1. Edit WELCOME_MESSAGES in this script")
    print(f"   2. Change GEMINI_VOICE to match your tts_manager.py")
    print(f"   3. Run: python AI/generate_audio_files.py")

    print("\n🚀 Next step:")
    print("   Run: python AI/call_center_agent.py")


if __name__ == "__main__":
    asyncio.run(main())