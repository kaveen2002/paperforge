# PaperForge (Gemini free-tier) — deploy to Render

One server that serves the web app AND does conversion + cropping + export,
using Google Gemini Flash (free tier). Works from any phone/browser once deployed.

## Files
- server.py         FastAPI app: serves the page + /convert + /export (uses Gemini)
- index.html        the web UI (served at the site root)
- requirements.txt  Python dependencies

## Get a FREE Gemini API key
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account, click "Create API key".
3. Copy the key (used as GEMINI_API_KEY below). The free tier has daily limits
   that comfortably cover a few papers a day.

## Run locally (optional test)
    pip install -r requirements.txt
    export GEMINI_API_KEY=your-key
    uvicorn server:app --reload --port 8000
    # open http://localhost:8000

## Deploy to Render (use from any phone)
1. Put these 3 files in a GitHub repo.
2. render.com -> New + -> Web Service -> connect the repo.
3. Settings:
   - Runtime:        Python 3
   - Build command:  pip install -r requirements.txt
   - Start command:  uvicorn server:app --host 0.0.0.0 --port $PORT
   - Instance type:  Free
4. Environment -> add variable:
   - GEMINI_API_KEY = your-google-ai-studio-key
5. Create Web Service. You get a URL like https://paperforge.onrender.com
6. Open that URL on any device — the app loads and works.

## Keep it awake (avoid free-tier sleep)
Add the URL to UptimeRobot or cron-job.org, ping https://YOUR-URL/health every ~10 min.

## Cost
- Gemini Flash: FREE within Google AI Studio's daily limits. No card needed to start.
- Render: free tier (sleeps when idle; keep-alive above avoids that).

## Notes / still-to-harden
- FIGURE_STORE is in-memory: figures lost on restart. Fine for single-session use.
  For durability, move to cloud storage keyed by paper id.
- Conversion quality lives in SYSTEM_PROMPT in server.py — tune against real screenshots.
- Crop boxes are ~accurate; a drag-to-adjust UI is the next polish step.
- To switch models later (e.g. gemini-2.0-flash-lite for higher limits, or back to
  Claude/Sonnet for max accuracy), only the /convert call needs changing.
