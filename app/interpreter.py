"""Interpreter sessions and stateless translation. Never persist audio or text."""

import asyncio
import html
import logging
import os
import re
import uuid
from typing import Annotated, Literal
from urllib.parse import unquote, urlencode, urlsplit

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictBool, StringConstraints, model_validator

from .providers import require_key

router = APIRouter(prefix="/api/v1/interpreter", tags=["Interpreter"])
logger = logging.getLogger("iapp.interpreter")
request_slots = asyncio.Semaphore(4)
TIMEOUT_SECONDS = 20
MAX_INTERPRETER_BODY_BYTES = 128 * 1024
Language = Literal["en", "es", "ht"]
SpeechProvider = Literal["deepgram", "gladia"]
Term = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160, pattern=r"^[^\x00-\x1f\x7f]+$")]


class VocabularyEntry(BaseModel):
    value: Term
    pronunciations: list[Term] = Field(default_factory=list, max_length=20)
    intensity: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    enabled: StrictBool = True


class SpellingEntry(BaseModel):
    value: Term
    variants: list[Term] = Field(min_length=1, max_length=20)
    enabled: StrictBool = True


class Glossary(BaseModel):
    defaultIntensity: float = Field(default=0.4, ge=0, le=1, allow_inf_nan=False)
    vocabulary: list[VocabularyEntry] = Field(default_factory=list, max_length=100)
    spelling: list[SpellingEntry] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def bounded_terms(self):
        count = sum(len(item.value) + sum(map(len, item.pronunciations)) for item in self.vocabulary)
        count += sum(len(item.value) + sum(map(len, item.variants)) for item in self.spelling)
        if count > 20_000:
            raise ValueError("Interpreter glossary exceeds 20,000 characters.")
        return self


class SessionRequest(BaseModel):
    provider: SpeechProvider
    sample_rate: Literal[8000, 16000, 32000, 44100, 48000] = 16000
    privacy_mode: StrictBool = True
    glossary: Glossary | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_private_glossary(cls, values):
        if isinstance(values, dict) and values.get("privacy_mode", True) is not False:
            return {**values, "glossary": None}
        return values


class SessionResponse(BaseModel):
    provider: SpeechProvider
    url: str
    protocols: list[str] = Field(default_factory=list)
    expires_in: int | None = None


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    source_lang: Language
    target_lang: Language
    privacy_mode: StrictBool = True

    @model_validator(mode="after")
    def usable_text(self):
        if not self.text.strip():
            raise ValueError("Write or record some text to translate.")
        if any(ord(char) < 32 and char not in "\t\n\r" for char in self.text):
            raise ValueError("Translation text must not contain binary control characters.")
        return self


class TranslationResponse(BaseModel):
    translated_text: str
    provider: Literal["deepl", "google", "identity"]


def capabilities() -> dict:
    configured = lambda name: bool(os.getenv(name, "").strip())
    return {
        "languages": ["en", "es", "ht"],
        "transcription": {"deepgram": configured("DEEPGRAM_API_KEY"), "gladia": configured("GLADIA_API_KEY")},
        "translation": {"deepl": configured("DEEPL_API_KEY"), "google": configured("GOOGLE_TRANSLATE_API_KEY")},
    }


def interpreter_error(error: Exception) -> HTTPException:
    status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
    if isinstance(error, (TimeoutError, httpx.TimeoutException)) or status in (408, 504):
        return HTTPException(504, "The interpreter provider took too long. Please retry.")
    if status in (401, 403):
        return HTTPException(502, "The interpreter provider rejected the server credentials or permissions. Check its API key, billing, and permissions in Render.")
    if status == 429:
        return HTTPException(429, "The interpreter provider rate limit or account quota was reached. Wait or check billing.")
    if status in (402, 456):
        return HTTPException(429, "The interpreter provider has insufficient credits or has reached its spending/character limit. Check the provider account's billing and limits before retrying.")
    if status in (400, 413, 422):
        return HTTPException(422, "The interpreter provider rejected this request. Check the selected languages, audio format, or glossary.")
    if isinstance(error, httpx.RequestError):
        return HTTPException(502, "The server could not connect to the interpreter provider. Please retry.")
    return HTTPException(502, "The interpreter provider could not complete the request. Please retry.")


async def bounded_request(operation: str, provider: str, callback):
    request_id = uuid.uuid4().hex[:12]
    try:
        await asyncio.wait_for(request_slots.acquire(), timeout=0.1)
    except TimeoutError:
        raise HTTPException(429, "The interpreter server is busy. Please try again shortly.") from None
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS + 2):
            return await callback()
    except HTTPException:
        raise
    except Exception as error:
        # Never log provider response bodies, credentials, URLs, glossary, or text.
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
        logger.warning("Interpreter failure request=%s operation=%s provider=%s error_type=%s upstream_status=%s", request_id, operation, provider, type(error).__name__, status)
        mapped = interpreter_error(error)
        mapped.headers = {"X-Request-ID": request_id}
        raise mapped from None
    finally:
        request_slots.release()


def gladia_processing(glossary: Glossary | None) -> dict:
    processing = {"translation": False, "custom_vocabulary": False, "custom_spelling": False}
    if glossary is None:
        return processing
    vocabulary = [
        {"value": item.value, "language": "ht",
         **({"pronunciations": item.pronunciations} if item.pronunciations else {}),
         **({"intensity": item.intensity} if item.intensity is not None else {})}
        for item in glossary.vocabulary if item.enabled
    ]
    spelling = {item.value: item.variants for item in glossary.spelling if item.enabled}
    if vocabulary:
        processing.update(custom_vocabulary=True, custom_vocabulary_config={"vocabulary": vocabulary, "default_intensity": glossary.defaultIntensity})
    if spelling:
        processing.update(custom_spelling=True, custom_spelling_config={"spelling_dictionary": spelling})
    return processing


async def create_session(request: SessionRequest) -> SessionResponse:
    if request.provider == "deepgram":
        key = require_key("DEEPGRAM_API_KEY")
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post("https://api.deepgram.com/v1/auth/grant", headers={"Authorization": f"Token {key}"}, json={"ttl_seconds": 30})
            response.raise_for_status()
            data = response.json()
        token, expires = data.get("access_token"), data.get("expires_in")
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token) or len(token) > 8192 or key in token:
            raise ValueError("Invalid temporary session credential")
        # The grant schema makes expires_in optional and numeric, not strictly
        # integer. Do not reject a valid JWT when the optional metadata is absent.
        if expires is not None and (type(expires) not in (int, float) or not 1 <= expires <= 60):
            raise ValueError("Invalid temporary session lifetime")
        params = {
            "model": "nova-3", "language": "multi", "smart_format": "true",
            "punctuate": "true", "numerals": "true", "interim_results": "true",
            "filler_words": "false", "endpointing": "300", "utterance_end_ms": "1200",
            "no_delay": "true", "vad_events": "true", "diarize": "false", "mip_opt_out": "true",
        }
        return SessionResponse(provider="deepgram", url=f"wss://api.deepgram.com/v1/listen?{urlencode(params)}", protocols=["bearer", token], expires_in=int(expires) if expires is not None else None)
    key = require_key("GLADIA_API_KEY")
    payload = {
        "encoding": "wav/pcm", "sample_rate": request.sample_rate, "bit_depth": 16, "channels": 1,
        "model": "solaria-1", "endpointing": 0.3, "maximum_duration_without_endpointing": 10,
        "language_config": {"languages": ["ht"], "code_switching": False},
        "pre_processing": {"audio_enhancer": True},
        "realtime_processing": gladia_processing(None if request.privacy_mode else request.glossary),
        "messages_config": {
            "receive_partial_transcripts": True, "receive_final_transcripts": True,
            "receive_speech_events": False, "receive_errors": True, "receive_lifecycle_events": False,
        },
        "callback": False,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
        response = await client.post("https://api.gladia.io/v2/live", headers={"x-gladia-key": key}, json=payload)
        response.raise_for_status()
        url = response.json().get("url")
    if not isinstance(url, str) or len(url) > 4096 or key in unquote(url) or any(char.isspace() for char in url):
        raise ValueError("Invalid live session URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "wss" or not parsed.hostname or not parsed.hostname.endswith(".gladia.io")
            or parsed.username or parsed.password or parsed.fragment or parsed.port not in (None, 443)):
        raise ValueError("Invalid live session URL")
    return SessionResponse(provider="gladia", url=url)


async def translate_text(request: TranslationRequest) -> TranslationResponse:
    if request.source_lang == request.target_lang:
        return TranslationResponse(translated_text=request.text, provider="identity")
    if "ht" in (request.source_lang, request.target_lang):
        provider = "google"
        url = "https://translation.googleapis.com/language/translate/v2"
        headers = {"x-goog-api-key": require_key("GOOGLE_TRANSLATE_API_KEY")}
        body = {"q": request.text, "source": request.source_lang, "target": request.target_lang, "format": "text"}
    else:
        provider = "deepl"
        key = require_key("DEEPL_API_KEY")
        default_url = "https://api-free.deepl.com/v2/translate" if key.endswith(":fx") else "https://api.deepl.com/v2/translate"
        url = os.getenv("DEEPL_API_URL", "").strip() or default_url
        if url not in {"https://api-free.deepl.com/v2/translate", "https://api.deepl.com/v2/translate"}:
            raise HTTPException(503, "DEEPL_API_URL must be the official DeepL Free or Pro translation endpoint.")
        headers = {"Authorization": f"DeepL-Auth-Key {key}"}
        body = {"text": [request.text], "source_lang": request.source_lang.upper(), "target_lang": "EN-US" if request.target_lang == "en" else "ES"}
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
    text = data["data"]["translations"][0]["translatedText"] if provider == "google" else data["translations"][0]["text"]
    if not isinstance(text, str) or len(text) > 20_000:
        raise ValueError("Invalid translation response")
    text = html.unescape(text) if provider == "google" else text
    if not text.strip():
        raise ValueError("Invalid translation response")
    # Unchanged words/names/numbers are legitimate translations; do not retry them.
    return TranslationResponse(translated_text=text, provider=provider)


@router.get("/capabilities")
async def get_capabilities() -> dict:
    return capabilities()


@router.post("/session", response_model=SessionResponse, response_model_exclude_none=True)
async def session(request: SessionRequest) -> SessionResponse:
    return await bounded_request("session", request.provider, lambda: create_session(request))


@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    provider = "google" if "ht" in (request.source_lang, request.target_lang) else "deepl"
    return await bounded_request("translate", provider, lambda: translate_text(request))
