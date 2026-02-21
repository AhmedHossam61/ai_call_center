from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import asyncio
from helpers.config import get_settings
from RAGcontrollers.DataController import DataController
from RAGcontrollers.DataProcessing import DataProcessing
from RAGcontrollers.VectorDB import VectorDB
import uuid


base_router = APIRouter()
settings = get_settings()

data_controller = DataController()
processor = DataProcessing()
vector_db = VectorDB()


@base_router.get("/welcome")
def get_info():
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION
    }


@base_router.post("/upload-and-index")
async def upload_and_index_file(
    file: UploadFile = File(...),
    doc_type: str = Form(default=None),
):
    """
    Unified Pipeline:
    Relational Storage (Postgres) -> AI Parsing (Docling) -> Vector Indexing (Qdrant)
    """
    try:
        file_info = await data_controller.save_file(file)
        db_id = file_info.get("id")
        file_path = file_info.get("path")

        if not file_path:
            raise HTTPException(status_code=500, detail="File path not returned from controller")

        raw_chunks = processor.process_single_file(file_path, file.filename)
        
        doc_id = uuid.uuid4().hex
        original_name = file.filename
        stored_name = file_info.get("filename")  # comes from DataController
        file_path = file_info.get("path")

        # Use caller-supplied doc_type or auto-detect from filename
        _RESUME_KEYWORDS = ["cv", "resume", "سيرة", "سيره", "profile", "بيانات", "شخصية"]
        if not doc_type:
            doc_type = "profile" if any(k in (original_name or "").lower() for k in _RESUME_KEYWORDS) else "company_kb"

        for chunk in raw_chunks:
            chunk["metadata"]["doc_id"] = doc_id
            chunk["metadata"]["doc_type"] = doc_type
            chunk["metadata"]["source_original"] = original_name
            chunk["metadata"]["source_stored"] = stored_name
            chunk["metadata"]["file_path"] = file_path

        if not raw_chunks:
            return {"status": "Warning", "message": "No text chunks extracted"}

        for chunk in raw_chunks:
            chunk["metadata"]["db_id"] = db_id

        total_indexed = vector_db.insert_chunks(raw_chunks)

        # Invalidate all open live sessions so they re-build with fresh RAG on next call
        try:
            from routers.websocket_endpoint import live_sessions
            stale = list(live_sessions.keys())
            for sid in stale:
                mgr = live_sessions.pop(sid, None)
                if mgr:
                    asyncio.create_task(mgr.close())
            if stale:
                print(f"🔄 Invalidated {len(stale)} live session(s) — RAG will refresh on next call")
        except Exception as inv_err:
            print(f"⚠️  Session invalidation error: {inv_err}")

        return {
            "status": "Success",
            "payload": {
                "text": chunk["text"],
                "source": chunk["metadata"].get("source_file"),  # existing
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "dialect": chunk["metadata"].get("dialect", "all"),

                # ✅ new fields
                "doc_id": chunk["metadata"].get("doc_id"),
                "doc_type": chunk["metadata"].get("doc_type"),
                "source_original": chunk["metadata"].get("source_original"),
                "source_stored": chunk["metadata"].get("source_stored"),
                "file_path": chunk["metadata"].get("file_path"),
            },
            "message": f"Successfully indexed {file.filename}. Ready for RAG."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {
            "status": "Error",
            "detail": "Internal server error during document indexing",
            "error_log": str(e)
        }