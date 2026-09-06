import os
from collections.abc import Mapping
from typing import Literal

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


Provider = Literal["Gemini", "GPT", "Claude"]

load_dotenv()

MODELS: Mapping[str, set[str]] = {
    "Gemini": {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
    },
    "GPT": {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"},
    "Claude": {"claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"},
}


class ChatRequest(BaseModel):
    provider: Provider
    model: str
    prompt: str = Field(min_length=1, max_length=100_000)


class ChatResponse(BaseModel):
    text: str


def configured_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]


app = FastAPI(title="IApp API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


def require_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise HTTPException(status_code=503, detail=f"{name} is not configured on the server.")
    return key


def provider_error(error: Exception) -> HTTPException:
    status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status_code in (401, 403):
        return HTTPException(status_code=502, detail="The provider rejected the server API key.")
    if status_code == 429:
        return HTTPException(status_code=429, detail="The provider rate limit or quota was reached.")
    return HTTPException(status_code=502, detail="The AI provider could not complete the request.")


async def ask_openai(prompt: str, model: str) -> str:
    client = AsyncOpenAI(api_key=require_key("OPENAI_API_KEY"))
    response = await client.responses.create(
        model=model,
        input=prompt,
        instructions="You are a helpful and creative assistant integrated in IApp.",
        store=False,
    )
    return response.output_text or "OpenAI did not return text."


async def ask_claude(prompt: str, model: str) -> str:
    client = AsyncAnthropic(api_key=require_key("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model=model,
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text or "Claude did not return text."


async def ask_gemini(prompt: str, model: str) -> str:
    client = genai.Client(api_key=require_key("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(model=model, contents=prompt)
    return response.text or "Gemini did not return text."


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.model not in MODELS[request.provider]:
        raise HTTPException(status_code=400, detail="Unsupported model for the selected provider.")

    try:
        if request.provider == "GPT":
            text = await ask_openai(request.prompt, request.model)
        elif request.provider == "Claude":
            text = await ask_claude(request.prompt, request.model)
        else:
            text = await ask_gemini(request.prompt, request.model)
    except HTTPException:
        raise
    except Exception as error:
        raise provider_error(error) from error

    return ChatResponse(text=text)
