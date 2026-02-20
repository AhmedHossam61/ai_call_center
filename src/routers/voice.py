from fastapi import APIRouter

router = APIRouter(prefix="/voice", tags=["voice"])

@router.get("/health")
def health_check():
    return {"status": "voice router ok"}

@router.get("/audio-spec")
def audio_spec():
    return {"input_sample_rate": 16000, "output_sample_rate": 24000, "encoding": "pcm_s16le"}