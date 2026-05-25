"""
PaperForge backend — Gemini (free-tier) edition
================================================
Same app, but the conversion call uses Google's Gemini Flash, which has a
generous FREE tier through Google AI Studio. Everything else — cropping,
numbering, export/zip — is identical and model-independent.

  POST /convert  ->  image + question number  =>  { latex, totalMarks, figures, crops }
  POST /export   ->  full paper state          =>  zip of paper.tex + all N.png figures
  GET  /         ->  serves the web UI (index.html beside this file)

Run locally:
  pip install -r requirements.txt
  export GEMINI_API_KEY=your-google-ai-studio-key      # free key, stays server-side
  uvicorn server:app --reload --port 8000
  # open http://localhost:8000

Get a free key: https://aistudio.google.com/apikey
"""

import os, io, json, base64, zipfile, re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# ---- model config ----------------------------------------------------------
MODEL = "gemini-2.0-flash"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

app = FastAPI(title="PaperForge (Gemini)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def home():
    index = os.path.join(HERE, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"ok": True, "model": MODEL, "note": "index.html not found beside server.py"}

FIGURE_STORE = {}   # tempName -> PNG bytes

SYSTEM_PROMPT = r"""
You convert a screenshot of ONE exam question into Edexcel-style LaTeX.

Reply with STRICT JSON ONLY (no prose, no markdown fences), shape:
{
  "latex": "<the \\item body, WITHOUT the \\item line and WITHOUT the total-marks line>",
  "totalMarks": <integer, or null if no total is printed in the image>,
  "figures": [ {"x":<0-1>,"y":<0-1>,"w":<0-1>,"h":<0-1>} ]
}

LaTeX rules:
- Preserve wording, punctuation, symbols, bold, italics, fractions, surds, powers EXACTLY.
- Sub-parts use nested \begin{enumerate} ... \end{enumerate}.
- Inline maths \( \); display maths \[ \].
- Marks per part as:  \hfill (2)
- Leave working space with \vspace: 2-3cm short, 4-5cm medium, 6-8cm long.
- For each figure, at the correct position insert:
    \begin{center}
    \includegraphics[width=0.7\textwidth]{__FIGURE_n__}
    \end{center}
  where n = 1,2,3... in order of appearance. The server replaces __FIGURE_n__ with the real filename.
- Do NOT add the \item line; do NOT add the (Total for Question ...) line.

figures:
- A bounding box for EVERY diagram/photo/figure (NOT tables, NOT text).
- Coordinates are fractions of the image (x,y = top-left corner; w,h = size).
- Box only the figure artwork, excluding caption and surrounding text.
- Same order as the __FIGURE_n__ placeholders.
"""

# ============================ /convert =====================================
@app.post("/convert")
async def convert(image: UploadFile = File(...), questionNumber: int = Form(...)):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    W, H = img.size

    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY not set on the server")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        resp = model.generate_content(
            [
                {"mime_type": image.content_type or "image/png", "data": raw},
                f"Convert this as Question {questionNumber}. JSON only.",
            ],
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:400]}")

    latex = data.get("latex", "")
    total = data.get("totalMarks")
    figures = data.get("figures", []) or []

    crops = []
    for i, box in enumerate(figures, start=1):
        try:
            rect = _frac_to_px(box, W, H, pad=0.01)
            crop = img.crop(rect)
            png = io.BytesIO(); crop.save(png, format="PNG"); png = png.getvalue()
        except Exception:
            continue
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box": box, "rect": rect, "dataUrl": preview})

    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "figures": figures, "crops": crops}


def _frac_to_px(box, W, H, pad=0.0):
    x = int(box["x"] * W); y = int(box["y"] * H)
    w = int(box["w"] * W); h = int(box["h"] * H)
    px = int(pad * W); py = int(pad * H)
    return (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))


# ============================ /export ======================================
class QIn(BaseModel):
    body: str
    marks: Optional[int] = None
    tempImageNames: List[str] = []

class PaperIn(BaseModel):
    title: str; author: str; cred: str; inst: str; contact: str; date: str
    questions: List[QIn]

@app.post("/export")
def export(paper: PaperIn):
    global_idx = 0
    items, packaged = [], {}
    for qi, q in enumerate(paper.questions, start=1):
        body = q.body
        for local_i, tmp in enumerate(q.tempImageNames, start=1):
            global_idx += 1
            final = f"{global_idx}.png"
            body = body.replace(f"__FIGURE_{local_i}__", final)
            if tmp in FIGURE_STORE:
                packaged[final] = FIGURE_STORE[tmp]
        total = (f"\n\\hfill \\textbf{{(Total for Question {qi} is {q.marks} marks)}}"
                 if q.marks is not None else "")
        items.append(f"\\item\n{body}\n{total}\n\\hline")

    tex = _doc(paper, "\n\n".join(items))

    refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    missing = [r for r in refs if r not in packaged]
    if missing:
        raise HTTPException(409, f"Missing figures for: {missing}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("paper.tex", tex)
        for name, png in packaged.items():
            z.writestr(name, png)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="paper_export.zip"'})


def _doc(p: PaperIn, items: str) -> str:
    return (
        "\\documentclass[a4paper,12pt]{article}\n"
        "\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{geometry}\\usepackage{array}\\usepackage{graphicx}\\usepackage{xcolor}\n"
        "\\geometry{margin=1in}\n\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n\\maketitle\n\\hline\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\\end{{document}}"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "figures_cached": len(FIGURE_STORE)}
