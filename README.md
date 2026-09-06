# IApp API

The API owns all provider credentials. The browser must never receive an API key.

## Local setup

1. Create `backend/.env` from `.env.example` and fill in the provider keys.
2. Create a Python virtual environment and install dependencies:

   ```powershell
   cd backend
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Start the API:

   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

4. In another terminal, start Vite. Its development proxy forwards `/api` to FastAPI.

## Production configuration

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` in the host's encrypted environment-variable settings. Set `CORS_ORIGINS` to the exact public URL of the frontend, for example `https://calvinrbobadillaf.github.io`.

For a separately hosted API, build the frontend with `VITE_API_BASE_URL=https://api.example.com/api/v1`.
