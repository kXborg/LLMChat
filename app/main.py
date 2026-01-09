from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os
from openai import OpenAI, BadRequestError, NotFoundError
from fastapi import UploadFile, File, HTTPException

from tavily import TavilyClient
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Import utility modules
from app.utils import (
    # RAG utilities
    init_qdrant_collection,
    extract_text_from_pdf,
    index_document,
    search_documents,
    get_user_documents,
    delete_document,
    perform_rag_search,
    document_metadata,
    # Vision utilities
    probe_vision_capability,
    get_vision_capability_from_request,
    get_max_history_turns_for_model,
    validate_attachments_for_model,
    # Message utilities
    build_messages,
    normalize_messages_for_vllm,
    # Search utilities
    perform_web_search,
    # Image utilities
    compress_image,
    # LLM utilities
    parse_allowed_tokens_from_error,
)

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()

# CORS (ok to keep even if same-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent

# Optionally serve static files under /static if folder exists
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Uploads directory (always ensure and mount)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Serve index.html at root
@app.get("/")
async def index():
    # Serve the app's index.html located alongside this file
    return FileResponse(str(BASE_DIR / "index.html"))


# OpenAI-compatible chat completions client (e.g., vLLM server)
openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
openai_api_base = os.getenv("OPENAI_API_BASE", "")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    default_headers={"ngrok-skip-browser-warning": "true"},
)

# Tavily Web Search client
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# Embedding model for RAG
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))  # characters per chunk
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))  # overlap between chunks
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # number of chunks to retrieve
RAG_COLLECTION_NAME = "documents"

# Initialize Qdrant client (in-memory)
qdrant_client = QdrantClient(":memory:")

# Create collection on startup
init_qdrant_collection(qdrant_client, RAG_COLLECTION_NAME)

user_histories = {}
vision_capability_overrides = {}  # Store user overrides for vision capability
SYSTEM_PROMPT = "You are a helpful assistant."

# Chat behavior knobs (override via env vars if needed)
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
CONTEXT_MARGIN = int(os.getenv("CONTEXT_MARGIN", "16"))  # safety headroom tokens
SUFFIX_MARGIN_TOKENS = int(os.getenv("SUFFIX_MARGIN_TOKENS", "24"))
TRUNCATION_SUFFIX = os.getenv("TRUNCATION_SUFFIX", "… Would you like me to continue?")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
VISION_MODELS_ENV = {m.strip() for m in os.getenv("VISION_MODELS", "").split(",") if m.strip()}
INLINE_LOCAL_UPLOADS = os.getenv("INLINE_LOCAL_UPLOADS", "1") in {"1", "true", "yes", "on"}

# Max history turns per model type (user+assistant pairs)
MAX_HISTORY_TURNS_TEXT = int(os.getenv("MAX_HISTORY_TURNS_TEXT", "6"))
MAX_HISTORY_TURNS_VISION = int(os.getenv("MAX_HISTORY_TURNS_VISION", "2"))

# Image compression settings
IMAGE_SIZE_THRESHOLD = int(os.getenv("IMAGE_SIZE_THRESHOLD", str(500 * 1024)))  # 500 KB
IMAGE_MAX_SIZE_THRESHOLD = int(os.getenv("IMAGE_MAX_SIZE_THRESHOLD", str(1 * 1024 * 1024)))  # 1 MB
IMAGE_MAX_DIMENSION = int(os.getenv("IMAGE_MAX_DIMENSION", "2048"))  # Max width or height
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))  # JPEG/WebP quality (1-100)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    attachments: Optional[List["Attachment"]] = None
    model: Optional[str] = None
    vision_enabled: Optional[bool] = None  # User toggle override for vision capability
    web_search: Optional[bool] = False  # Enable web search for this request
    rag_enabled: Optional[bool] = False  # Enable RAG for this request


class ChatResponse(BaseModel):
    reply: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: int | None = None
    tokens_per_sec: float | None = None


class Attachment(BaseModel):
    filename: str
    url: str
    mime_type: str
    text: Optional[str] = None


# Wrapper functions to adapt utility functions to main.py's context
def _get_vision_capability_from_request(user_id, model_id, vision_enabled_override=None):
    """Wrapper to call vision utility with proper context."""
    return get_vision_capability_from_request(
        openai_api_base, user_id, model_id, vision_capability_overrides, vision_enabled_override
    )


def _get_max_history_turns_for_model(model_id, user_id="", vision_enabled_override=None):
    """Wrapper to call vision utility with proper context."""
    return get_max_history_turns_for_model(
        openai_api_base, model_id, MAX_HISTORY_TURNS_TEXT, MAX_HISTORY_TURNS_VISION,
        vision_capability_overrides, user_id, vision_enabled_override
    )


def _validate_attachments_for_model(model_id, attachments, user_id="", vision_enabled_override=None):
    """Wrapper to call vision utility with proper context."""
    return validate_attachments_for_model(
        openai_api_base, model_id, attachments, vision_capability_overrides,
        user_id, vision_enabled_override
    )


def _perform_rag_search(user_id, query):
    """Wrapper to call RAG utility with proper context."""
    return perform_rag_search(
        qdrant_client, RAG_COLLECTION_NAME, RAG_EMBEDDING_MODEL, user_id, query, RAG_TOP_K
    )


def _perform_web_search(query, max_results=5):
    """Wrapper to call search utility with proper context."""
    return perform_web_search(tavily_client, query, max_results)


def _build_messages(user_id, user_message, attachments=None, model_id=DEFAULT_MODEL, 
                   vision_enabled_override=None, web_search=False, rag_enabled=False):
    """Wrapper to call message utility with proper context."""
    return build_messages(
        user_id, user_message, SYSTEM_PROMPT, user_histories, attachments, model_id,
        UPLOAD_DIR, INLINE_LOCAL_UPLOADS, _get_max_history_turns_for_model,
        _get_vision_capability_from_request, _perform_web_search, _perform_rag_search,
        vision_enabled_override, web_search, rag_enabled
    )


def _normalize_messages_for_vllm(messages, model_id, user_id="", vision_enabled_override=None):
    """Wrapper to call message utility with proper context."""
    return normalize_messages_for_vllm(
        messages, model_id, _get_vision_capability_from_request, user_id, vision_enabled_override
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model_name = (req.model or DEFAULT_MODEL).strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="No model selected. Please pick a model from the dropdown.")
    _validate_attachments_for_model(model_name, req.attachments, req.user_id, req.vision_enabled)
    messages = _build_messages(
        req.user_id, 
        req.message, 
        req.attachments, 
        model_name, 
        req.vision_enabled, 
        req.web_search or False, 
        req.rag_enabled or False)
    messages = _normalize_messages_for_vllm(messages, model_name, req.user_id, req.vision_enabled)
    max_tokens = DEFAULT_MAX_TOKENS
    # Reserve some tokens for suffix if we hit the length limit
    effective_max_tokens = max(16, max_tokens - SUFFIX_MARGIN_TOKENS)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    except BadRequestError as e:
        allowed = parse_allowed_tokens_from_error(str(e))
        if allowed is None:
            # Fallback: halve and retry once
            effective_max_tokens = max(16, effective_max_tokens // 2)
        else:
            effective_max_tokens = max(16, min(effective_max_tokens, allowed - max(0, SUFFIX_MARGIN_TOKENS)))

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    except (NotFoundError, Exception) as e:
        # Fallback to web search when model is unavailable
        if tavily_client:
            search_results = _perform_web_search(req.message)
            if search_results:
                fallback_answer = f"**Model is not available, results are from web:**\n\n{search_results}"
                return ChatResponse(
                    reply=fallback_answer,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    duration_ms=None,
                    tokens_per_sec=None,
                )
        # If no web search available or failed, raise the original error
        if isinstance(e, NotFoundError):
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found on server. Choose an available model.")
        raise HTTPException(status_code=500, detail=f"Model unavailable and web search fallback failed: {str(e)}")

    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    answer = resp.choices[0].message.content or ""
    if finish_reason == "length" and TRUNCATION_SUFFIX:
        answer = f"{answer}{TRUNCATION_SUFFIX}"

    # Usage metrics (if provided by server)
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    # tokens/sec not precisely known; approximate via server timing not available
    # Let duration be None for non-streaming unless later instrumented server-side

    if req.user_id not in user_histories:
        user_histories[req.user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": answer},
        ]
    else:
        user_histories[req.user_id].append({"role": "user", "content": req.message})
        user_histories[req.user_id].append({"role": "assistant", "content": answer})

    # Prune stored history to respect max turns per selected model
    hist = user_histories.get(req.user_id)
    if hist:
        system_msg = []
        tail = []
        if isinstance(hist[0], dict) and hist[0].get("role") == "system":
            system_msg = [hist[0]]
            tail = hist[1:]
        else:
            tail = hist
        max_tail_msgs = _get_max_history_turns_for_model(model_name, req.user_id, req.vision_enabled) * 2
        trimmed_tail = tail[-max_tail_msgs:]
        user_histories[req.user_id] = [*system_msg, *trimmed_tail]

    return ChatResponse(
        reply=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_ms=None,
        tokens_per_sec=(completion_tokens / 1.0) if completion_tokens else None,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    model_name = (req.model or DEFAULT_MODEL).strip()
    if not model_name:
        def gen_err():
            yield "No model selected. Please pick a model from the dropdown."
        return StreamingResponse(gen_err(), media_type="text/plain")
    _validate_attachments_for_model(model_name, req.attachments, req.user_id, req.vision_enabled)
    messages = _build_messages(
        req.user_id, 
        req.message, 
        req.attachments, 
        model_name, 
        req.vision_enabled, 
        req.web_search or False, 
        req.rag_enabled or False)
    messages = _normalize_messages_for_vllm(messages, model_name, req.user_id, req.vision_enabled)

    def token_generator():
        buffer = []
        max_tokens = DEFAULT_MAX_TOKENS
        effective_max_tokens = max(16, max_tokens - SUFFIX_MARGIN_TOKENS)
        last_finish = None
        import time
        start_t = time.perf_counter()

        def do_stream(toks):
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=toks,
                temperature=0.8,
                top_p=0.95,
                stream=True,
            )

        try:
            stream = do_stream(effective_max_tokens)
        except BadRequestError as e:
            allowed = parse_allowed_tokens_from_error(str(e))
            if allowed is None:
                effective_max_tokens = max(16, effective_max_tokens // 2)
            else:
                effective_max_tokens = max(16, min(effective_max_tokens, allowed - max(0, SUFFIX_MARGIN_TOKENS)))
            try:
                stream = do_stream(effective_max_tokens)
            except BadRequestError:
                yield "Context limit reached. Please start a new chat or /reset."
                return
        except NotFoundError:
            # Fallback to web search when model is unavailable
            if tavily_client:
                search_results = _perform_web_search(req.message)
                if search_results:
                    yield f"**Model is not available, results are from web:**\n\n{search_results}"
                    return
            yield f"Model '{model_name}' not found on server. Choose an available model."
            return
        except Exception as e:
            # Fallback to web search for any other model errors
            if tavily_client:
                search_results = _perform_web_search(req.message)
                if search_results:
                    yield f"**Model is not available, results are from web:**\n\n{search_results}"
                    return
            yield f"Model unavailable: {str(e)}"
            return

        for chunk in stream:
            choice0 = chunk.choices[0]
            last_finish = getattr(choice0, "finish_reason", last_finish)
            delta = getattr(choice0, "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None) or ""
            if piece:
                buffer.append(piece)
                yield piece

        final_answer = "".join(buffer)
        if last_finish == "length" and TRUNCATION_SUFFIX:
            # Append suffix to the stream output
            final_answer = f"{final_answer}{TRUNCATION_SUFFIX}"
            yield TRUNCATION_SUFFIX
        end_t = time.perf_counter()
        dur_ms = int((end_t - start_t) * 1000)
        # Rough token estimate: 4 chars ≈ 1 token
        approx_tokens = max(1, len(final_answer) // 4)
        tps = approx_tokens / max(0.001, (end_t - start_t))
        # Emit a final metrics line that the client can parse (optional)
        yield f"\n\n[throughput] duration_ms={dur_ms} tokens_per_sec={tps:.2f} approx_tokens={approx_tokens}\n"
        if req.user_id not in user_histories:
            user_histories[req.user_id] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": final_answer},
            ]
        else:
            user_histories[req.user_id].append({"role": "user", "content": req.message})
            user_histories[req.user_id].append({"role": "assistant", "content": final_answer})
        # Prune stored history to respect max turns per selected model
        hist = user_histories.get(req.user_id)
        if hist:
            system_msg = []
            tail = []
            if isinstance(hist[0], dict) and hist[0].get("role") == "system":
                system_msg = [hist[0]]
                tail = hist[1:]
            else:
                tail = hist
            max_tail_msgs = _get_max_history_turns_for_model(model_name, req.user_id, req.vision_enabled) * 2
            trimmed_tail = tail[-max_tail_msgs:]
            user_histories[req.user_id] = [*system_msg, *trimmed_tail]

    return StreamingResponse(token_generator(), media_type="text/plain")


class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5


@app.post("/search")
async def web_search(req: SearchRequest):
    """
    Standalone web search endpoint for testing or direct use.
    """
    if not tavily_client:
        raise HTTPException(status_code=503, detail="Web search is not configured. Set TAVILY_API_KEY.")
    
    try:
        response = tavily_client.search(
            query=req.query,
            search_depth="basic",
            max_results=req.max_results or 5,
            include_answer=True,
        )
        return {
            "query": req.query,
            "answer": response.get("answer"),
            "results": response.get("results", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/status")
def search_status():
    """Check if web search is configured and available."""
    return {
        "available": tavily_client is not None,
        "provider": "tavily" if tavily_client else None,
    }



class RAGUploadRequest(BaseModel):
    user_id: str


@app.post("/rag/upload")
async def rag_upload(user_id: str, file: UploadFile = File(...)):
    """Upload and index a PDF document for RAG."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported for RAG")
    
    # Save file temporarily
    import uuid
    doc_id = str(uuid.uuid4())
    unique_name = f"{doc_id}{ext}"
    dest = UPLOAD_DIR / unique_name
    
    with dest.open("wb") as out:
        content = await file.read()
        out.write(content)
    
    # Extract text from PDF
    text = extract_text_from_pdf(dest)
    if not text.strip():
        dest.unlink()  # Clean up
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Index the document
    chunk_count = index_document(
        qdrant_client, RAG_COLLECTION_NAME, RAG_EMBEDDING_MODEL,
        user_id, doc_id, file.filename, text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
    )
    
    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": chunk_count,
        "text_length": len(text),
    }


@app.get("/rag/documents")
def rag_list_documents(user_id: str):
    """List all documents indexed for a user."""
    docs = get_user_documents(user_id)
    return {"documents": docs}


@app.delete("/rag/documents/{doc_id}")
def rag_delete_document(doc_id: str, user_id: str):
    """Delete a document from the RAG index."""
    success = delete_document(qdrant_client, RAG_COLLECTION_NAME, user_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return {"status": "ok", "doc_id": doc_id}


class RAGSearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: Optional[int] = 5


@app.post("/rag/search")
def rag_search(req: RAGSearchRequest):
    """Search indexed documents (for testing)."""
    results = search_documents(
        qdrant_client, RAG_COLLECTION_NAME, RAG_EMBEDDING_MODEL,
        req.user_id, req.query, req.top_k or RAG_TOP_K
    )
    return {"query": req.query, "results": results}


@app.get("/rag/status")
def rag_status():
    """Check RAG system status."""
    doc_count = len(document_metadata)
    return {
        "available": True,
        "embedding_model": RAG_EMBEDDING_MODEL,
        "collection": RAG_COLLECTION_NAME,
        "document_count": doc_count,
    }



class ResetRequest(BaseModel):
    user_id: str | None = None


@app.post("/reset")
def reset_chat(req: ResetRequest):
    if req.user_id:
        user_histories.pop(req.user_id, None)
        return {"status": "ok", "cleared": "user", "user_id": req.user_id}
    else:
        user_histories.clear()
        return {"status": "ok", "cleared": "all"}


@app.get("/models")
def list_models():
    items = []

    try:
        data = client.models.list()
        for m in getattr(data, "data", []) or []:
            mid = getattr(m, "id", None) or getattr(m, "model", None) or ""
            if not isinstance(mid, str) or not mid:
                continue

            vision = probe_vision_capability(openai_api_base, mid)

            items.append({
                "id": mid,
                "vision": vision,
            })

    except Exception:
        pass

    default_id = DEFAULT_MODEL if DEFAULT_MODEL else (items[0]["id"] if items else None)
    return {
        "models": items,
        "default": default_id,
    }


class VisionOverrideRequest(BaseModel):
    model_id: str
    user_id: str
    vision_enabled: bool


@app.post("/vision-override")
def set_vision_override(req: VisionOverrideRequest):
    """
    Allow user to override vision capability detection for a model.
    Stores the override in session-like dict keyed by user_id:model_id.
    """
    override_key = f"{req.user_id}:{req.model_id}"
    vision_capability_overrides[override_key] = req.vision_enabled
    return {"status": "ok", "override_key": override_key, "vision_enabled": req.vision_enabled}


@app.get("/vision-override")
def get_vision_override(user_id: str, model_id: str):
    """
    Retrieve the override for a specific user+model combo, or None if not set.
    """
    override_key = f"{user_id}:{model_id}"
    override = vision_capability_overrides.get(override_key)
    return {"override_key": override_key, "vision_enabled": override}



ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown",
    ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"
}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")

        # Use unique filename to avoid collisions
        unique = f"{os.urandom(8).hex()}{ext}"
        dest = UPLOAD_DIR / unique
        # Persist file to disk
        with dest.open("wb") as out:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        # Compress image if it exceeds size threshold
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            compress_image(
                dest, uf.content_type or "application/octet-stream",
                IMAGE_SIZE_THRESHOLD, IMAGE_MAX_DIMENSION, IMAGE_QUALITY
            )

        item = {
            "filename": name,
            "url": f"/uploads/{unique}",
            "mime_type": uf.content_type or "application/octet-stream",
        }

        # If text file, attach a text preview
        if ext in {".txt", ".md", ".markdown"}:
            try:
                txt = (UPLOAD_DIR / unique).read_text(encoding="utf-8", errors="ignore")
                item["text"] = txt[:20000]
            except Exception:
                pass

        # PDFs and images are stored and referenced via URL; no extraction here
        saved.append(item)

    return {"files": saved}