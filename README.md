# PaperForge — deploy to Render

One server that serves the web app AND does conversion + cropping + export.
Works from any phone/browser once deployed.

## Files
- server.py         FastAPI app: serves the page + /convert + /export
- index.html        the web UI (served at the site root)
- requirements.txt  Python dependencies

## Run locally (optional test)
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
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
   - ANTHROPIC_API_KEY = sk-ant-your-key
5. Create Web Service. You get a URL like https://paperforge.onrender.com
6. Open that URL on any device — the app loads and works.

## Keep it awake (avoid free-tier sleep)
Add the URL to UptimeRobot or cron-job.org, ping https://YOUR-URL/health every ~10 min.

## Notes / still-to-harden for real production
- FIGURE_STORE is in-memory: figures are lost on restart. For durability move to
  S3/GCS keyed by paper id. Fine for single-session use as-is.
- Tune SYSTEM_PROMPT in server.py against your real screenshots to improve LaTeX quality.
- Crop boxes from the model are ~accurate; a manual drag-to-adjust UI is the next polish.
