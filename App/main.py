import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .catalog import CATALOG, MODELS
from .providers import TIMEOUT_SECONDS, ask_claude, ask_gemini, ask_openai, create_image, provider_error
from .schemas import MAX_BODY_BYTES, ChatRequest, ChatResponse

load_dotenv()
logger = logging.getLogger("iapp.api")


def configured_origins() -> list[str]:
    return [origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]


class RequestLimitsMiddleware:
    """Reject oversized bodies before JSON parsing, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def no_cache_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"cache-control", b"no-store")]
            await send(message)

        if scope["method"] != "POST":
            return await self.app(scope, receive, no_cache_send)
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length", b"")
        if declared:
            try:
                if int(declared) > MAX_BODY_BYTES:
                    return await JSONResponse(status_code=413, content={"detail": "Request exceeds 18 MiB. Remove attachments or start a new chat."})(scope, receive, no_cache_send)
            except ValueError:
                return await JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})(scope, receive, no_cache_send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return await JSONResponse(status_code=413, content={"detail": "Request exceeds 18 MiB. Remove attachments or start a new chat."})(scope, receive, no_cache_send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, no_cache_send)


app = FastAPI(title="IApp API", version="2.0.0")
app.add_middleware(RequestLimitsMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=configured_origins(), allow_credentials=False,
    allow_methods=["POST", "GET"], allow_headers=["Content-Type"], expose_headers=["X-Request-ID"],
)
# Per-worker concurrency protection; this does not provide user authentication.
request_slots = asyncio.Semaphore(4)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, error: RequestValidationError):
    # Default validation responses include rejected input, potentially a private file.
    messages = [item["msg"].removeprefix("Value error, ") for item in error.errors()[:3]]
    return JSONResponse(status_code=422, content={"detail": " ".join(messages)})


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/models")
async def models() -> dict:
    return CATALOG


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.model not in MODELS[request.provider]:
        raise HTTPException(status_code=400, detail="Unsupported model for the selected provider. Refresh the model list and select another model.")
    request_id = uuid.uuid4().hex[:12]
    try:
        await asyncio.wait_for(request_slots.acquire(), timeout=0.1)
    except TimeoutError:
        raise HTTPException(status_code=429, detail="The server is busy. Please try again shortly.") from None
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS + 5):
            if request.mode == "image":
                return await create_image(request)
            adapter = {"GPT": ask_openai, "Claude": ask_claude, "Gemini": ask_gemini}[request.provider]
            return await adapter(request)
    except HTTPException:
        raise
    except Exception as error:
        logger.warning("Provider failure request=%s provider=%s model=%s error_type=%s", request_id, request.provider, request.model, type(error).__name__)
        mapped = provider_error(error)
        mapped.headers = {"X-Request-ID": request_id}
        raise mapped from None
    finally:
        request_slots.release()
