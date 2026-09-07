"""Model identifiers are shared with the frontend via GET /api/v1/models."""

CATALOG = {
    "Gemini": {
        "default_model": "gemini-3.8-flash",
        "models": [
            {"value": "gemini-3.8-flash", "label": "Gemini 3.8 Flash"},
            {"value": "gemini-3.7-flash", "label": "Gemini 3.7 Flash"},
            {"value": "gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
            {"value": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash Lite"},
            {"value": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro Preview"},
        ],
        "image_model": "gemini-3.1-flash-image",
    },
    "GPT": {
        "default_model": "gpt-5.6-terra",
        "models": [
            {"value": "gpt-5.6-sol", "label": "GPT-5.6 Sol"},
            {"value": "gpt-5.6-terra", "label": "GPT-5.6 Terra"},
            {"value": "gpt-5.6-luna", "label": "GPT-5.6 Luna"},
        ],
        "image_model": "gpt-image-2",
    },
    "Claude": {
        "default_model": "claude-opus-5",
        "models": [
            {"value": "claude-opus-5", "label": "Claude Opus 5"},
            {"value": "claude-sonnet-5", "label": "Claude Sonnet 5"},
            {"value": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
        ],
        "image_model": None,
    },
}

MODELS = {provider: {model["value"] for model in config["models"]} for provider, config in CATALOG.items()}
