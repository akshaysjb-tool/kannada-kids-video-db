import json
import os
import re
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from starlette.concurrency import run_in_threadpool
from google import genai
from google.genai import types
import turso_serverless

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it").strip()
# Vercel functions have a read-only deployment filesystem; /tmp is writable but ephemeral.
DEFAULT_DATABASE_PATH = "/tmp/video_history.db" if os.getenv("VERCEL") else str(BASE_DIR / "video_history.db")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH))
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()
IS_VERCEL = os.getenv("VERCEL", "").lower() == "1"
SYSTEM_PROMPT_PATH = Path(os.getenv("SYSTEM_PROMPT_PATH", str(BASE_DIR / "system_prompt.txt")))

app = FastAPI(
    title="Kannada Kids Video Generator",
    description="Generate Kannada kids-video scripts and social metadata using Google AI Studio models.",
    version="1.0.0",
)

class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    video_settings: dict[str, Any] = Field(default_factory=lambda: {
        "duration_seconds": 15,
        "aspect_ratio": "9:16",
        "visual_style": "cinematic_realistic",
        "language": "Kannada",
        "platform": ["Instagram Reels", "YouTube Shorts"],
    })
    characters: dict[str, Any] = Field(default_factory=lambda: {
        "boy_age": 3,
        "girl_age": 4,
        "boy_outfit": "AUTO",
        "girl_outfit": "AUTO",
        "boy_accessories": "AUTO",
        "girl_accessories": "AUTO",
    })
    story_settings: dict[str, Any] = Field(default_factory=lambda: {
        "theme": "AUTO",
        "district": "AUTO",
        "location": "AUTO",
        "regional_dialect": "AUTO",
        "comedy_style": "AUTO",
    })
    custom_instructions: str = ""

@contextmanager
def db_connect():
    """Open a remote Turso connection, or local SQLite for local development."""
    if bool(TURSO_DATABASE_URL) != bool(TURSO_AUTH_TOKEN):
        raise RuntimeError(
            "Set both TURSO_DATABASE_URL and TURSO_AUTH_TOKEN, or remove both for local SQLite."
        )

    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        # HTTP-based DB-API driver: suitable for stateless/serverless hosting such as Vercel.
        conn = turso_serverless.connect(
            TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
    else:
        if IS_VERCEL:
            raise RuntimeError(
                "Turso is not configured. Add TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in Vercel Environment Variables."
            )
        conn = sqlite3.connect(DATABASE_PATH, timeout=30)
        conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()


def row_value(row, column_name: str, column_index: int = 0):
    """Read a value from either sqlite3.Row or a DB-API tuple row."""
    if hasattr(row, "keys"):
        return row[column_name]
    return row[column_index]


def row_to_dict(row, description) -> dict[str, Any]:
    """Convert sqlite3.Row or a DB-API tuple row into a normal dictionary."""
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    column_names = [column[0] for column in description]
    return dict(zip(column_names, row))


def init_db():
    if not (TURSO_DATABASE_URL and TURSO_AUTH_TOKEN) and not IS_VERCEL:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                title_normalized TEXT NOT NULL,
                theme TEXT,
                district TEXT,
                location TEXT,
                model_name TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                response_hash TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_title ON generations(title_normalized)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_created ON generations(created_at)")
        conn.commit()


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise HTTPException(status_code=500, detail=f"System prompt file not found: {SYSTEM_PROMPT_PATH}")
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise HTTPException(status_code=500, detail="system_prompt.txt is empty.")
    return prompt

def normalize_title(title: str) -> str:
    title = title.casefold().strip()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
    return title

def recent_titles(limit: int = 40) -> list[str]:
    with db_connect() as conn:
        cursor = conn.execute(
            "SELECT title FROM generations ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
    return [row_value(row, "title") for row in rows]

def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    # Remove common markdown fences if the model ignores the JSON-only instruction.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON object.")
        parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")
    return parsed

def validate_output(data: dict[str, Any], duration: int) -> None:
    required = ["video_metadata", "regional_settings", "characters", "scenes",
                "punchline", "complete_video_generation", "social_media"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required output keys: {', '.join(missing)}")
    if not isinstance(data["scenes"], list) or not data["scenes"]:
        raise ValueError("scenes must be a non-empty array.")
    if not isinstance(data["social_media"], dict):
        raise ValueError("social_media must be an object.")
    if not isinstance(data["complete_video_generation"], dict):
        raise ValueError("complete_video_generation must be an object.")
    # Validate scene timings without assuming every model uses identical optional fields.
    for index, scene in enumerate(data["scenes"], start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {index} must be an object.")
        start = scene.get("start_time_seconds")
        end = scene.get("end_time_seconds")
        if start is None or end is None:
            raise ValueError(f"Scene {index} needs start_time_seconds and end_time_seconds.")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"Scene {index} timestamps must be numbers.")
        if start < 0 or end <= start or end > duration:
            raise ValueError(f"Scene {index} has invalid timestamps for {duration}s video.")
    metadata_duration = data["video_metadata"].get("duration_seconds")
    if metadata_duration is not None and int(metadata_duration) != duration:
        raise ValueError("video_metadata.duration_seconds must match requested duration.")
    title = data["video_metadata"].get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("video_metadata.title must be a non-empty string.")
    master_prompt = data["complete_video_generation"].get("master_prompt")
    if not isinstance(master_prompt, str) or not master_prompt.strip():
        raise ValueError("complete_video_generation.master_prompt must be a non-empty string.")

def generate_with_google(system_prompt: str, request_data: dict[str, Any], excluded_titles: list[str]) -> dict[str, Any]:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing. Add it to your .env file.")
    client = genai.Client(api_key=API_KEY)
    duration = int(request_data.get("video_settings", {}).get("duration_seconds", 15))
    user_prompt = f"""
Generate one complete Kannada kids video package using the system instructions.

USER REQUEST JSON:
{json.dumps(request_data, ensure_ascii=False, indent=2)}

RECENT TITLES TO AVOID:
{json.dumps(excluded_titles, ensure_ascii=False)}

IMPORTANT:
- Return exactly one valid JSON object and nothing else.
- Follow the output keys and structure specified in the system prompt.
- All spoken dialogue must be in Kannada script.
- Scene timestamps must cover the requested {duration} seconds without overlaps or gaps where practical.
- The final scene must end at exactly {duration} seconds.
- Do not reuse any title in RECENT TITLES TO AVOID.
- Include a complete copy-paste-ready video-generation master_prompt with the actual dialogue.
- Include Instagram and YouTube title, description, hashtags, SEO keywords, and thumbnail text.
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.85,
            max_output_tokens=8192,
        ),
    )
    return extract_json(response.text or "")

def save_generation(request_data: dict[str, Any], output: dict[str, Any]) -> int:
    metadata = output.get("video_metadata", {})
    regional = output.get("regional_settings", {})
    title = str(metadata.get("title", "Untitled")).strip()
    normalized = normalize_title(title)
    response_json = json.dumps(output, ensure_ascii=False)
    digest = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()
    with db_connect() as conn:
        cursor = conn.execute("""
            INSERT INTO generations
            (created_at, title, title_normalized, theme, district, location,
             model_name, request_json, response_json, response_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """, (
            created_at, title, normalized,
            str(metadata.get("theme", "")),
            str(regional.get("district", "")),
            str(regional.get("location", "")),
            MODEL_NAME,
            json.dumps(request_data, ensure_ascii=False),
            response_json,
            digest,
        ))
        inserted_row = cursor.fetchone()
        conn.commit()
        if inserted_row is None:
            raise RuntimeError("Database did not return the new generation ID.")
        return int(row_value(inserted_row, "id"))

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    database_name = "Turso Cloud" if TURSO_DATABASE_URL else str(DATABASE_PATH)
    return {"status": "ok", "model": MODEL_NAME, "database": database_name}

@app.post("/generate")
async def generate_video(payload: VideoRequest):
    request_data = payload.model_dump()
    duration = int(request_data.get("video_settings", {}).get("duration_seconds", 15))
    if duration < 5 or duration > 60:
        raise HTTPException(status_code=422, detail="duration_seconds must be between 5 and 60.")
    system_prompt = load_system_prompt()
    excluded = recent_titles(limit=40)

    try:
        output = await run_in_threadpool(generate_with_google, system_prompt, request_data, excluded)
        validate_output(output, duration)

        # Simple duplicate-title check: if the title already exists, ask for one fresh version.
        existing = {normalize_title(x) for x in excluded}
        generated_title = normalize_title(output["video_metadata"]["title"])
        if generated_title in existing:
            retry_note = excluded + [output["video_metadata"]["title"]]
            retry_request = dict(request_data)
            retry_request["custom_instructions"] = (
                str(retry_request.get("custom_instructions", "")) +
                "\nIMPORTANT: Generate a substantially different story and a new title. "
                "Do not reuse the previous title or story."
            )
            output = await run_in_threadpool(generate_with_google, system_prompt, retry_request, retry_note)
            validate_output(output, duration)

        generation_id = save_generation(request_data, output)
        output["generation_record"] = {
            "id": generation_id,
            "stored_in_database": True,
            "database": "Turso Cloud" if TURSO_DATABASE_URL else "Local SQLite",
            "duplicate_title_check": "checked_recent_40_titles",
            "model": MODEL_NAME,
        }
        return output

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {type(exc).__name__}: {exc}")

@app.get("/history")
def get_history(limit: int = 20):
    limit = max(1, min(limit, 100))
    with db_connect() as conn:
        cursor = conn.execute("""
            SELECT id, created_at, title, theme, district, location, model_name
            FROM generations ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        items = [row_to_dict(row, cursor.description) for row in rows]
    return {"count": len(items), "items": items}

@app.get("/history/{generation_id}")
def get_generation(generation_id: int):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT response_json FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    return json.loads(row_value(row, "response_json"))

@app.get("/", response_class=HTMLResponse)
def simple_gui():
    return HTMLResponse(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kannada Kids Video Generator</title>
<style>
body{font-family:Arial,sans-serif;max-width:1000px;margin:28px auto;padding:0 16px;background:#f6f7fb;color:#202434}
.card{background:white;padding:20px;border-radius:12px;margin:14px 0;box-shadow:0 2px 12px #0000000d}
label{display:block;font-weight:600;margin:12px 0 5px}
input,textarea,select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccd1dc;border-radius:7px}
button{padding:11px 18px;border:0;border-radius:8px;background:#2457d6;color:white;font-weight:700;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:#e5e7eb;padding:16px;border-radius:8px;max-height:650px;overflow:auto}
small{color:#5c6475}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:650px){.grid{grid-template-columns:1fr}}
</style></head>
<body>
<h1>Kannada Kids Video Generator</h1>
<p>Enter only the details you want to customize. Leave fields as AUTO for automatic generation.</p>
<div class="card">
<div class="grid">
<div><label>Video duration (seconds)</label><input id="duration" type="number" min="5" max="60" value="15"></div>
<div><label>Aspect ratio</label><select id="ratio"><option>9:16</option><option>16:9</option><option>1:1</option></select></div>
<div><label>Boy outfit</label><input id="boy" value="AUTO"></div>
<div><label>Girl outfit</label><input id="girl" value="AUTO"></div>
<div><label>Theme</label><input id="theme" value="AUTO"></div>
<div><label>District</label><input id="district" value="AUTO"></div>
<div><label>Location</label><input id="location" value="AUTO"></div>
<div><label>Regional Kannada dialect</label><input id="dialect" value="AUTO"></div>
</div>
<label>Additional instructions</label><textarea id="extra" rows="3" placeholder="Example: funny conversation, festival setting, strong unexpected punchline"></textarea>
<br><br><button onclick="generate()">Generate video JSON</button>
<small id="status"></small>
</div>
<div class="card"><h2>Generated JSON</h2><button onclick="copyOutput()">Copy JSON</button><pre id="result">Your generated JSON will appear here.</pre></div>
<div class="card"><h2>Recent generations</h2><button onclick="loadHistory()">Refresh history</button><div id="history"></div></div>
<script>
let lastOutput = "";
async function generate(){
 const status=document.getElementById("status"); status.textContent=" Generating...";
 const payload={
  video_settings:{duration_seconds:Number(document.getElementById("duration").value),aspect_ratio:document.getElementById("ratio").value,visual_style:"cinematic_realistic",language:"Kannada",platform:["Instagram Reels","YouTube Shorts"]},
  characters:{boy_age:3,girl_age:4,boy_outfit:document.getElementById("boy").value,girl_outfit:document.getElementById("girl").value,boy_accessories:"AUTO",girl_accessories:"AUTO"},
  story_settings:{theme:document.getElementById("theme").value,district:document.getElementById("district").value,location:document.getElementById("location").value,regional_dialect:document.getElementById("dialect").value,comedy_style:"AUTO"},
  custom_instructions:document.getElementById("extra").value
 };
 try{
  const res=await fetch("/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await res.json();
  if(!res.ok) throw new Error(data.detail||"Request failed");
  lastOutput=JSON.stringify(data,null,2);document.getElementById("result").textContent=lastOutput;
  status.textContent=" Done. Saved to database.";loadHistory();
 }catch(e){status.textContent=" Error: "+e.message;}
}
async function copyOutput(){if(lastOutput){await navigator.clipboard.writeText(lastOutput);alert("JSON copied");}}
async function loadHistory(){
 try{const res=await fetch("/history?limit=20");const data=await res.json();
 document.getElementById("history").innerHTML=data.items.map(x=>`<p><b>#${x.id}</b> ${x.title} <small>(${x.district||"district"})</small> <a href="/history/${x.id}" target="_blank">Open JSON</a></p>`).join("")||"No history yet.";
 }catch(e){document.getElementById("history").textContent=e.message;}
}
loadHistory();
</script></body></html>
""")
