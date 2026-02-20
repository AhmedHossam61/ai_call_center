# AI Call Center 📞🤖

An advanced, AI-powered call center solution featuring real-time Speech-to-Text (STT), Retrieval Augmented Generation (RAG) for localized knowledge, Arabic dialect detection, and high-quality Text-to-Speech (TTS) response generation.

---

## 🌟 Key Features

- **🗣️ Real-time Arabic STT**: Uses OpenAI's **Whisper** model for accurate transcription of Arabic speech.
- **🌍 Dialect Detection**: Automatically detects and adapts to different Arabic dialects (e.g., Egyptian, MSA) using **Google Gemini**.
- **🔍 RAG Support (Knowledge Base)**: Integrated with **Qdrant** Vector DB to provide context-aware responses based on provided documents.
- **💬 Smart Response Generation**: Powered by **Gemini 2.5 Flash** for natural, helpful, and dialect-appropriate interactions.
- **🔊 High-Quality TTS**: Delivers smooth voice responses via **Edge-TTS**.
- **🚀 Dual Interface**: Supports both a **CLI-based call handler** for direct interaction and a **FastAPI** web server for integration.

---

## 🏗️ Architecture Overview

The system operates in a pipeline:
1.  **Input**: Audio is captured via `sounddevice` / microphone.
2.  **STT**: `whisper` transcribes the audio to text.
3.  **RAG**: `VectorDB` searches for relevant context in the Qdrant database.
4.  **Processor**: `CallCenterAgent` orchestrates dialect detection and response generation using Gemini.
5.  **Output**: `edge-tts` generates speech, which is played back to the user.

---

## 🛠️ Technology Stack

- **Core Logic**: Python 3.10+
- **AI Models**:
    - **LLM/Embeddings**: Google Gemini API (`gemini-2.5-flash`, `text-embedding-004`)
    - **STT**: OpenAI Whisper (Local)
- **Database**: Qdrant (Vector Search)
- **Web Framework**: FastAPI
- **Audio Processing**: SoundDevice, SoundFile, Pydub, Edge-TTS

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- [FFmpeg](https://ffmpeg.org/) (Required for audio processing)
- [Qdrant](https://qdrant.tech/documentation/quick-start/) (Running instance for RAG)

### 2. Install Dependencies
```bash
pip install -r src/requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the `src/` directory (refer to `env.example`):
```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
VECTOR_COLLECTION_NAME=knowledge_base
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768
```

---

## 🚀 How to Run

### Interactive CLI Agent
To start the AI Call Center in your terminal:
```bash
python src/AI/call_center_agent.py
```
*Follow the on-screen prompts to "Press Enter" and start speaking.*

### FastAPI Server
To run the API server for voice services:
```bash
cd src
uvicorn main:app --reload
```
The server will be available at `http://127.0.0.1:8000`.

---

## 📁 Project Structure

- `src/AI/`: Core AI logic (Agent, Session management, Dialect detection).
- `src/RAGcontrollers/`: Vector database (Qdrant) integration and embedding logic.
- `src/routers/`: FastAPI routes for voice and system health.
- `src/helpers/`: Configuration and utility functions.
- `src/main.py`: Entry point for the FastAPI application.

---

## 📝 License
This project is for demonstration and development purposes. Please ensure compliance with Google Gemini and OpenAI Whisper licensing when using in production.
