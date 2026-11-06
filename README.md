# Kannada Kids Video Generator (Vercel + Turso)

## Redeploy
1. Replace the files in your GitHub/Vercel project with the contents of this folder.
2. Keep `main.py`, `system_prompt.txt`, `requirements.txt`, `vercel.json`, and `.python-version` at the project root.
3. In Vercel → Project → Settings → Environment Variables, set `GEMINI_API_KEY`, `GEMINI_MODEL`, `TURSO_DATABASE_URL`, and `TURSO_AUTH_TOKEN`. Do not commit real keys to GitHub.
4. Redeploy from Vercel → Deployments.

## Google Flow prompt copy feature
After a successful generation, the page displays a separate **Complete prompt for Google Flow** section. Click **Copy complete prompt for Google Flow** and paste the copied standalone prompt into Google Flow. The full JSON response remains available below and the API response keeps `complete_video_generation.master_prompt`.

## Endpoints
- `/` — generator interface
- `/docs` — API documentation
- `/health` — health check
- `/generate` — generate a script and prompt
- `/history` — generation history

The application generates the script and prompt; it does not automatically submit or generate a video inside Google Flow.
