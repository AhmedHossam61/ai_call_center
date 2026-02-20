import threading
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import base, voice
from routers.websocket_endpoint import stream
from routers import websocket_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    - Startup: nothing heavy (Gemini Live sessions open lazily on first call)
    - Shutdown: close any open Gemini Live sessions
    """
    print("✅ Server ready — Gemini Live sessions will open on first call")
    yield

    # Close all Live sessions gracefully
    try:
        from routers.websocket_endpoint import live_sessions

        for session_id, mgr in list(live_sessions.items()):
            print(f"🔌 Closing Gemini Live session: {session_id}")
            try:
                await mgr.close()
            except Exception:
                pass
        print("👋 All sessions closed")
    except Exception:
        pass


app = FastAPI(
    title="AI Call Center API — Gemini Live Demo",
    description="Real-time Arabic call center powered by Gemini Live API",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FastRTC WebRTC endpoints onto FastAPI
stream.mount(app, path="/ws/live-call")

# REST routers
app.include_router(voice.router, tags=["Voice"])
app.include_router(base.base_router, tags=["Knowledge Base / Upload"])
app.include_router(websocket_endpoint.base_router, tags=["Live Call"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "message": "AI Call Center API is active"}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀  AI CALL CENTER — GEMINI LIVE DEMO")
    print("=" * 60)
    print("📖  Swagger UI  →  http://localhost:8000/docs")
    print("🎙️   FastRTC UI  →  check terminal for gradio.live link")
    print("=" * 60 + "\n")

    # ── FastRTC Gradio UI on port 8001 ──────────────────────────
    # share=True generates a public HTTPS link so the browser
    # can access the microphone (HTTP blocks mic by default)
    def run_ui():
        stream.ui.launch(
            server_name="0.0.0.0",
            server_port=8001,
            share=True,
            quiet=False,
        )

    threading.Thread(target=run_ui, daemon=True).start()

    # ── FastAPI / Swagger on port 8000 ──────────────────────────
    uvicorn.run(app, host="0.0.0.0", port=8000)
