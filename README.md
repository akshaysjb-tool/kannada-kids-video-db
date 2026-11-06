# Kannada Kids Video Generator API

A small FastAPI application that:
- Accepts a JSON POST request.
- Loads instructions from `system_prompt.txt`.
- Calls a Google AI Studio model using the Google Gen AI Python SDK.
- Parses and validates the generated JSON.
- Stores each generation in SQLite.
- Checks the latest 40 generated titles and retries once if the title is duplicated.
- Provides a simple browser GUI at `/` and API docs at `/docs`.

## 1. Requirements

Python 3.10+ recommended.

## 2. Install

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 3. Configure API key

Copy `.env.example` to `.env` and set your Google AI Studio API key:

```env
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemma-4-26b-a4b-it
```

The default is a hosted Gemma 4 model available through the Gemini API. You can change `GEMINI_MODEL` to a Gemini model available to your account. The model name must be supported by your API key and project.

## 4. Run

```bash
uvicorn main:app --reload
```

Open:
- Browser GUI: http://127.0.0.1:8000/
- Swagger API testing: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## 5. POST request

Endpoint:
`POST http://127.0.0.1:8000/generate`

Example JSON body:

```json
{
  "video_settings": {
    "duration_seconds": 15,
    "aspect_ratio": "9:16",
    "visual_style": "cinematic_realistic",
    "language": "Kannada",
    "platform": ["Instagram Reels", "YouTube Shorts"]
  },
  "characters": {
    "boy_age": 3,
    "girl_age": 4,
    "boy_outfit": "Yellow kurta and white dhoti",
    "girl_outfit": "Pink silk pavada",
    "boy_accessories": "Small traditional watch",
    "girl_accessories": "Jasmine flowers"
  },
  "story_settings": {
    "theme": "AUTO",
    "district": "AUTO",
    "location": "AUTO",
    "regional_dialect": "AUTO",
    "comedy_style": "Unexpected punchline"
  },
  "custom_instructions": "Funny family-friendly conversation with a strong Kannada punchline."
}
```

The response is one JSON object containing the video concept, regional dialect, characters, timed scenes, dialogues, final punchline, master video prompt, audio instructions, Instagram and YouTube metadata, hashtags, SEO keywords, and thumbnail suggestions.

## 6. SQLite history

The database file `video_history.db` is created automatically.

Endpoints:
- `GET /history?limit=20` — list recent generations.
- `GET /history/{id}` — retrieve a complete saved JSON response.

Duplicate check is intentionally simple: it compares normalized titles against the last 40 saved generations and asks the model for one retry if a title matches. It is not semantic duplicate detection; different titles can still describe similar stories.

## 7. Important model note

The application uses `system_instruction` from the TXT file and asks the model to return JSON. The Python code then parses and validates the result. This approach is used for Gemma compatibility rather than relying on every model supporting the same structured-output schema features.

## 8. Production notes

- Keep `.env` private; never commit your API key.
- Add authentication before exposing this API publicly.
- Add rate limiting and request-size limits for public deployments.
- Check the selected model's availability and current pricing in Google AI Studio.
- The API creates the prompt and metadata; it does not call your video-generation provider yet. Pass `complete_video_generation.master_prompt` or individual `scenes[].scene_generation_prompt` to your video tool.


## Deploy on Vercel

1. Push this folder to a GitHub repository.
2. Import the repository in Vercel.
3. Add `GEMINI_API_KEY` and `GEMINI_MODEL` under Project Settings -> Environment Variables.
4. Deploy. Vercel detects the root `main.py` FastAPI `app` automatically.
5. Test `/health`, `/docs`, and `/generate`.

`vercel.json` explicitly bundles `system_prompt.txt` and sets a 300-second function duration. On Vercel, SQLite is placed under `/tmp`, which is writable but temporary and instance-local. Do not rely on it for permanent or cross-instance history. Use a hosted PostgreSQL database for persistent history.


## 8. Connect to Turso Cloud (recommended for Vercel)

The app uses Turso's HTTP-based Python DB-API driver for remote database access. This avoids relying on Vercel's temporary local filesystem. Local development can still use SQLite when Turso credentials are not set.

Add these environment variables to your local `.env` file and to Vercel → Project → Settings → Environment Variables:

```env
TURSO_DATABASE_URL=libsql://your-database-name-your-org.turso.io
TURSO_AUTH_TOKEN=your_turso_auth_token
```

Also keep your existing `GEMINI_API_KEY` and `GEMINI_MODEL` variables. Do not commit `.env` or expose the Turso token.

Install dependencies with `pip install -r requirements.txt`. On Vercel, redeploy after adding the environment variables. The app creates the `generations` table and indexes automatically at startup.

The `GET /history` and `GET /history/{generation_id}` endpoints now read from Turso when the Turso credentials are configured.
